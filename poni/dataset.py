import gc
import cv2
import bz2
import math
import json
import tqdm
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp
import _pickle as cPickle
import multiprocessing as mp
import skimage.morphology as skmp

from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
from poni.geometry import (
    spatial_transform_map,
    crop_map,
    get_frontiers_np,
)
from poni.constants import (
    SPLIT_SCENES,
    OBJECT_CATEGORIES,
    INV_OBJECT_CATEGORY_MAP,
    NUM_OBJECT_CATEGORIES,
    # General constants
    CAT_OFFSET,
    FLOOR_ID,
    # Coloring
    d3_40_colors_rgb,
    gibson_palette,
    MIN_OBJECTS_THRESH,
)
from poni.fmm_planner import FMMPlanner
from einops import asnumpy, repeat
from matplotlib import font_manager

MIN_OBJECTS_THRESH = 4
EPS = 1e-10


def is_int(s):
    """
    Check if a string can be converted to an integer.
    
    Args:
        s: The string to check
        
    Returns:
        bool: True if the string can be converted to an integer, False otherwise
    """
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def load_data(save_path):
    with bz2.BZ2File(save_path, 'rb') as fp:
        data = cPickle.load(fp)
    return data


class SemanticMapDataset(Dataset):
    """
    Loads semantic maps and associated information for training or evaluation.
    Handles different datasets like Gibson, MP3D, and HM3D.
    """
    grid_size = 0.05  # m
    object_boundary = 1.0  # m

    def __init__(
        self,
        cfg,
        split='train',
        scf_name=None,  # scene_floor name e.g. Allensville_0
        seed=None,
    ):
        self.cfg = cfg
        self.dset = cfg.dset_name
        # Seed the dataset
        if seed is None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)

        # Load maps
        maps_path = sorted(glob.glob(osp.join(cfg.root, "*.h5")))
        maps_info_path = osp.join(cfg.root, 'semmap_GT_info.json')

        # Check if the required JSON info file exists
        if not osp.exists(maps_info_path):
            raise FileNotFoundError(
                f"Error: semmap_GT_info.json not found in {cfg.root}. "
                f"Please ensure you have run the preprocessing script "
                f"(e.g., preprocess_hm3d.py) for the {self.dset} dataset."
            )

        # Load json info
        try:
            with open(maps_info_path, 'r') as f:
                maps_info = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {maps_info_path}: {e}")

        maps = {}
        names = []
        maps_xyz_info = {}
        fmm_dists = {}  # Store FMM distances here

        print(f"Loading {split} split for {self.dset} dataset from {cfg.root}...")
        num_skipped_split = 0
        num_skipped_scf = 0
        num_skipped_json_scene = 0
        num_skipped_json_floor = 0
        num_skipped_objects = 0
        num_loaded = 0

        for path in tqdm.tqdm(maps_path, desc="Processing maps"):
            scene_name = path.split('/')[-1].split('.')[0]

            # Check if scene belongs to the specified split
            if scene_name not in SPLIT_SCENES.get(self.dset, {}).get(split, []):
                num_skipped_split += 1
                continue

            # Filter by specific scene_floor name if provided
            if (scf_name is not None) and (not scf_name.startswith(scene_name)):
                # Check prefix only, floor check happens later
                num_skipped_scf += 1
                continue

            # Check if scene metadata exists in the JSON file
            if scene_name not in maps_info:
                print(f"Warning: Metadata for scene '{scene_name}' not found in {maps_info_path}. Skipping.")
                num_skipped_json_scene += 1
                continue

            try:
                with h5py.File(path, 'r') as fp:
                    floor_ids = sorted([key for key in fp.keys() if is_int(key)])
                    for floor_id in floor_ids:
                        name = f'{scene_name}_{floor_id}'  # scene_floor name

                        # Filter by specific scene_floor name if provided (exact match)
                        if (scf_name is not None) and (name != scf_name):
                            num_skipped_scf += 1  # Increment here for exact floor mismatch
                            continue

                        # Check if floor metadata exists in the JSON file for this scene
                        if floor_id not in maps_info[scene_name]:
                            num_skipped_json_floor += 1
                            continue

                        # Safely access metadata now that checks passed
                        map_world_shift = maps_info[scene_name].get('map_world_shift', [0.0, 0.0, 0.0])  # Provide default
                        map_y = maps_info[scene_name][floor_id].get('y_min', 0.0)  # Provide default
                        resolution = maps_info[scene_name].get('resolution', 0.05)  # Provide default

                        map_semantic = np.array(fp[floor_id]['map_semantic'])
                        nuniq = len(np.unique(map_semantic))

                        # Check if map has enough unique objects (excluding background/walls)
                        if nuniq < MIN_OBJECTS_THRESH + 2:  # +2 for floor and wall/out-of-bounds
                            num_skipped_objects += 1
                            continue

                        maps[name] = map_semantic
                        names.append(name)
                        maps_xyz_info[name] = {
                            'map_world_shift': map_world_shift,
                            'y_min': map_y,
                            'resolution': resolution,
                        }
                        num_loaded += 1

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue  # Skip this file on error

        print(f"Finished loading maps. Summary:")
        print(f"  - Total HDF5 files found: {len(maps_path)}")
        print(f"  - Skipped (wrong split): {num_skipped_split}")
        if scf_name is not None:
            print(f"  - Skipped (scf mismatch): {num_skipped_scf}")
        print(f"  - Skipped (missing scene in JSON): {num_skipped_json_scene}")
        print(f"  - Skipped (missing floor in JSON): {num_skipped_json_floor}")
        print(f"  - Skipped (insufficient objects): {num_skipped_objects}")
        print(f"  - Successfully loaded scene_floors: {num_loaded}")

        if num_loaded == 0:
            print(f"Warning: No maps were loaded for split '{split}'. Check dataset paths, split definitions, and preprocessing output.")

        self.maps = maps
        self.names = sorted(names)
        self.maps_xyz_info = maps_xyz_info
        self.visibility_size = cfg.visibility_size

        # Pre-compute or load FMM distances for each semantic map
        if not self.names:  # If no maps were loaded, skip FMM
            print("Skipping FMM distance calculation as no maps were loaded.")
            self.fmm_dists = {}
            self.navigable_locs = {}
            return

        if self.cfg.fmm_dists_saved_root == '':
            print("Computing FMM distances...")
            pool = mp.Pool(8)  # Use multiprocessing for speed
            inputs = [(self.maps[name], self.maps_xyz_info[name]['resolution']) for name in self.names]
            fmm_dists_list = list(tqdm.tqdm(pool.imap(self._compute_fmm_dist, inputs), total=len(inputs)))
            self.fmm_dists = {name: dist for name, dist in zip(self.names, fmm_dists_list)}
            pool.close()
            pool.join()
        else:
            print(f"Loading precomputed FMM distances from {self.cfg.fmm_dists_saved_root}...")
            self.fmm_dists = {}
            missing_fmm = 0
            for name in tqdm.tqdm(self.names, desc="Loading FMM"):
                fmm_path = osp.join(self.cfg.fmm_dists_saved_root, f"{name}.pbz2")
                if osp.exists(fmm_path):
                    try:
                        self.fmm_dists[name] = load_data(fmm_path)
                    except Exception as e:
                        print(f"Warning: Error loading FMM data for {name} from {fmm_path}: {e}. Skipping.")
                        missing_fmm += 1
                else:
                    print(f"Warning: Precomputed FMM distance file not found for {name} at {fmm_path}. Skipping.")
                    missing_fmm += 1
            if missing_fmm > 0:
                print(f"Warning: Failed to load FMM distances for {missing_fmm} maps.")
            if len(self.fmm_dists) == 0 and len(self.names) > 0:
                print("ERROR: No FMM distances could be loaded. Cannot proceed without FMM data.")

        # Pre-compute navigable locations for each map using FMM distances
        print("Computing navigable locations...")
        self.navigable_locs = {}
        for name in tqdm.tqdm(self.names, desc="Computing Navigable Locs"):
            if name in self.fmm_dists:
                dist = self.fmm_dists[name]
                self.navigable_locs[name] = np.array(np.where(dist > 0)).T
            else:
                self.navigable_locs[name] = np.empty((0, 2), dtype=int)
                print(f"Warning: Cannot compute navigable locations for {name} due to missing FMM data.")

        print(f"Dataset initialization complete for split '{split}'. Found {len(self.names)} valid scene_floors.")

    def _compute_fmm_dist(self, args):
        """Helper function to compute FMM distance for a single map."""
        map_semantic, resolution = args
        traversable = map_semantic > 0
        planner = FMMPlanner(traversable, resolution)
        fmm_dist = planner.fmm_dist
        return fmm_dist

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        map_semantic = self.maps[name]
        fmm_dist = self.fmm_dists.get(name, None)
        navigable_locs = self.navigable_locs.get(name, np.empty((0, 2), dtype=int))
        map_xyz_info = self.maps_xyz_info[name]

        if fmm_dist is None:
            print(f"Warning: Missing FMM data for {name} in __getitem__.")
            fmm_dist = np.zeros_like(map_semantic, dtype=np.float32)

        if len(navigable_locs) > 0:
            start_loc = navigable_locs[np.random.randint(len(navigable_locs))]
        else:
            print(f"Warning: No navigable locations found for {name}. Using default [0,0].")
            start_loc = np.array([0, 0])

        return {
            'name': name,
            'map_semantic': map_semantic,
            'fmm_dist': fmm_dist,
            'start_loc': start_loc,
            'map_xyz_info': map_xyz_info,
            'navigable_locs': navigable_locs
        }