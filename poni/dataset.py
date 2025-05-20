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
    OBJECT_CATEGORIES, # Used to get category names for visualization titles
    INV_OBJECT_CATEGORY_MAP,
    NUM_OBJECT_CATEGORIES,
    # Dataset specific palettes if defined, or general ones
    gibson_palette, # Assuming this is a list of RGB float tuples or list of floats
    d3_40_colors_rgb,
    # Potentially add HM3D specific palettes here if defined in constants.py
    # e.g., HM3D_PALETTE from poni.constants
    # For now, we will add a placeholder for hm3d_palette that can be refined.
    CAT_OFFSET, # This was 1, its usage in convert_maps_to_oh was potentially problematic.
                # Assuming it means object categories (excluding floor/wall) start at this index
                # in some external system, but for PONI's one-hot maps, we align with OBJECT_CATEGORY_MAP.
    FLOOR_ID,   # Should be 0 if 'floor' is the first category in OBJECT_CATEGORIES[dset]
)
from poni.fmm_planner import FMMPlanner
from einops import asnumpy, repeat
from matplotlib import font_manager

MIN_OBJECTS_THRESH = 4
EPS = 1e-10


def is_int(s):
    try:
        int(s)
        return True
    except:
        return False


class SemanticMapDataset(Dataset):
    grid_size = 0.05 # m
    object_boundary = 1.0 # m
    def __init__(
        self,
        cfg,
        split='train',
        scf_name=None,
        seed=None,
    ):
        self.cfg = cfg
        self.dset = cfg.dset_name
        if self.dset not in NUM_OBJECT_CATEGORIES:
            raise ValueError(f"Dataset name '{self.dset}' not found in PONI constants (NUM_OBJECT_CATEGORIES). Please define it in poni/constants.py.")

        # Seed the dataset
        if seed is None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)

        # Load maps
        maps_path = sorted(glob.glob(osp.join(cfg.root, "*.h5")))
        # Load json info
        maps_info = json.load(open(osp.join(cfg.root, 'semmap_GT_info.json')))
        maps = {}
        names = []
        maps_xyz_info = {}
        for path in maps_path:
            scene_name = path.split('/')[-1].split('.')[0]
            if scene_name not in SPLIT_SCENES.get(self.dset, {}).get(split, []):
                continue
            if (scf_name is not None) and (scene_name not in scf_name): # Ensure scf_name can be a list/set
                continue
            with h5py.File(path, 'r') as fp:
                floor_ids = sorted([key for key in fp.keys() if is_int(key)])
                for floor_id in floor_ids:
                    name = f'{scene_name}_{floor_id}'
                    if (scf_name is not None) and (name != scf_name): # If scf_name is a single map name
                        continue
                    map_world_shift = maps_info[scene_name]['map_world_shift']
                    if floor_id not in maps_info[scene_name]:
                        continue
                    map_y = maps_info[scene_name][floor_id]['y_min']
                    resolution = maps_info[scene_name]['resolution']
                    map_semantic_ids = np.array(fp[floor_id]['map_semantic']) # This has category IDs
                    nuniq = len(np.unique(map_semantic_ids))
                    # +2 for floor and wall
                    if nuniq >= MIN_OBJECTS_THRESH + 2:
                        names.append(name)
                        # Directly use map_semantic_ids for one-hot conversion
                        maps[name] = self.convert_maps_to_oh(map_semantic_ids)
                        maps_xyz_info[name] = {
                            'world_shift': map_world_shift,
                            'resolution': resolution,
                            'y': map_y,
                            'scene_name': scene_name,
                        }
        self.maps = maps
        self.names = sorted(names)
        self.maps_xyz_info = maps_xyz_info
        self.visibility_size = cfg.visibility_size
        # Pre-compute FMM dists for each semmap
        if self.cfg.fmm_dists_saved_root == '':
            self.fmm_dists = self.compute_fmm_dists()
        else:
            self.fmm_dists = {}
            print(f"Loading FMM distances for {self.dset} split {split}...")
            for name in tqdm.tqdm(self.names):
                fname = f'{cfg.fmm_dists_saved_root}/{name}.pbz2'
                if not osp.exists(fname):
                    print(f"Warning: FMM dist file not found: {fname}. Skipping map {name}.")
                    # Remove map if FMM data is missing, or handle appropriately
                    if name in self.maps: del self.maps[name]
                    if name in self.maps_xyz_info: del self.maps_xyz_info[name]
                    continue # Skip this map from further processing
                try:
                    with bz2.BZ2File(fname, 'rb') as fp:
                        self.fmm_dists[name] = (cPickle.load(fp)).astype(np.float32)
                except Exception as e:
                    print(f"Error loading FMM dist for {name}: {e}. Skipping.")
                    if name in self.maps: del self.maps[name]
                    if name in self.maps_xyz_info: del self.maps_xyz_info[name]
                    continue
            self.names = sorted([n for n in self.names if n in self.maps]) # Update names list

        # Pre-compute navigable locations for each map
        self.nav_locs = self.compute_navigable_locations()

    def __len__(self):
        return len(self.names) # Use updated names list

    def __getitem__(self, idx):
        name = self.names[idx]
        semmap_oh = self.maps[name] # This is already one-hot
        fmm_dists = self.fmm_dists[name]
        map_xyz_info = self.maps_xyz_info[name]

        # The FLOOR_ID from constants.py (expected to be 0) refers to the floor channel
        # in the one-hot encoded map.
        nav_space = semmap_oh[FLOOR_ID] # Assuming FLOOR_ID is 0 and is the floor channel
        nav_locs = self.nav_locs[name]
        # Create input and output maps
        if self.cfg.masking_mode == 'spath':
            spath = self.get_random_shortest_path(nav_space, nav_locs)
            input_map, label = self.create_spath_based_input_output_pairs(
                semmap_oh, fmm_dists, spath, map_xyz_info,
            )
        else:
            raise ValueError(f"Masking mode {self.cfg.masking_mode} is not implemented!")
        return input_map, label

    def get_item_by_name(self, name):
        if name not in self.names:
            raise ValueError(f"Map name {name} not found in dataset.")
        idx = self.names.index(name)
        return self[idx]

    def convert_maps_to_oh(self, semmap_with_cat_ids):
        """
        Converts a 2D map with category IDs to a one-hot encoded map.
        The channel order in semmap_oh will correspond to the category ID order.
        (e.g., channel 0 for category ID 0 (floor), channel 1 for ID 1 (wall), etc.)
        """
        n_total_categories = NUM_OBJECT_CATEGORIES[self.dset]
        semmap_oh = np.zeros((n_total_categories, *semmap_with_cat_ids.shape), dtype=np.float32)
        for cat_id_val in range(n_total_categories):
            semmap_oh[cat_id_val] = (semmap_with_cat_ids == cat_id_val).astype(np.float32)
        return semmap_oh

    def plan_path(self, nav_space, start_loc, end_loc):
        planner = FMMPlanner(nav_space)
        planner.set_goal(end_loc)
        curr_loc = start_loc
        spath = [curr_loc]
        ctr = 0
        while True:
            ctr += 1
            if ctr > 10000: # Increased safety break
                print(f"plan_path() --- Run into infinite loop for start:{start_loc}, end:{end_loc}!")
                # Fallback: return just the start and end if pathing fails badly
                # Or, if end_loc is not navigable, this might loop.
                # A better FMMPlanner might return failure.
                if not nav_space[end_loc[0], end_loc[1]]: # if end is not navigable
                    return [start_loc] if nav_space[start_loc[0], start_loc[1]] else []
                return [start_loc, end_loc] if nav_space[start_loc[0], start_loc[1]] and nav_space[end_loc[0], end_loc[1]] else [start_loc] if nav_space[start_loc[0], start_loc[1]] else []


            next_y, next_x, _, stop = planner.get_short_term_goal(curr_loc)
            if stop:
                # Ensure the goal is actually reached or very close if stop is triggered
                if np.linalg.norm(np.array(curr_loc) - np.array(end_loc)) > planner.step_size * 1.5 : # Check if curr_loc is near end_loc
                    spath.append(end_loc) # Add end_loc if stop happens far from it
                break
            curr_loc = (int(round(next_y)), int(round(next_x))) # Ensure integer coords
            spath.append(curr_loc)
        return spath

    def get_random_shortest_path(self, nav_space, nav_locs):
        planner = FMMPlanner(nav_space)
        ys, xs = nav_locs
        if xs.shape[0] == 0: # No navigable locations
            return []

        num_outer_trials = 0
        while True:
            num_outer_trials += 1
            if num_outer_trials > 1000:
                print(f"=======> GetRandomShortestPath: Stuck in infinite outer loop!")
                # Fallback: pick two random navigable points if possible
                if xs.shape[0] >= 2:
                    idx1, idx2 = np.random.choice(xs.shape[0], 2, replace=False)
                    return [(ys[idx1], xs[idx1]), (ys[idx2], xs[idx2])]
                elif xs.shape[0] == 1:
                    return [(ys[0], xs[0])]
                return []


            rnd_ix = np.random.randint(0, xs.shape[0])
            start_x, start_y = xs[rnd_ix], ys[rnd_ix]
            try:
                planner.set_goal((start_y, start_x))
            except IndexError: # Goal might be out of bounds if nav_space is too small/problematic
                continue

            rchble_mask = planner.fmm_dist < planner.fmm_dist.max().item()
            if np.count_nonzero(rchble_mask) < 2: # Need at least two points for a path
                continue

            rchble_y, rchble_x = np.where(rchble_mask)
            if rchble_x.shape[0] == 0:
                continue

            rnd_ix = np.random.randint(0, rchble_x.shape[0])
            end_x, end_y = rchble_x[rnd_ix], rchble_y[rnd_ix]

            if start_y == end_y and start_x == end_x: # Start and end are same
                if xs.shape[0] > 1 : # try to pick different point
                    continue
                else: # only one navigable point on the map
                    return [(start_y, start_x)]
            break
        spath = self.plan_path(nav_space, (start_y, start_x), (end_y, end_x))
        return spath

    def compute_fmm_dists(self):
        fmm_dists_all_maps = {}
        # Assuming self.grid_size and self.object_boundary are in meters
        # Convert object_boundary to grid cells
        selem_radius_cells = int(round(self.object_boundary / self.grid_size))
        selem = skmp.disk(selem_radius_cells)

        print(f"Computing FMM distances for {self.dset}...")
        for name in tqdm.tqdm(self.names):
            semmap_oh = self.maps[name] # This is one-hot (C, H, W)
            navmap = semmap_oh[FLOOR_ID] # Floor channel
            dists_for_map = []

            n_categories_for_dset = NUM_OBJECT_CATEGORIES[self.dset]

            for cat_idx in range(n_categories_for_dset):
                catmap = semmap_oh[cat_idx] # Binary map for this category

                if np.count_nonzero(catmap) == 0:
                    fmm_dist = np.full_like(catmap, np.inf, dtype=np.float32)
                else:
                    # Create traversible map for this specific category's FMM:
                    # Free space is traversible (navmap)
                    # Goal regions (catmap) are also considered traversible for distance calculation
                    # This ensures FMM can reach points within the object extent.
                    current_traversible = np.logical_or(navmap, catmap).astype(np.float32)
                    planner = FMMPlanner(current_traversible)

                    # Dilate category map to define goal region for FMM
                    # Points within object_boundary of the category are considered goal.
                    dilated_catmap = skmp.binary_dilation(catmap, selem).astype(np.float32)
                    planner.set_multi_goal(dilated_catmap, validate_goal=True) # validate_goal will ensure goals are on traversible parts
                    fmm_dist = np.copy(planner.fmm_dist).astype(np.float32)
                dists_for_map.append(fmm_dist)
            fmm_dists_all_maps[name] = np.stack(dists_for_map, axis=0)
        return fmm_dists_all_maps

    def compute_object_pfs(self, fmm_dists):
        cutoff = self.cfg.object_pf_cutoff_dist
        # fmm_dists is already in meters here if loaded from precomputed or * self.grid_size
        opfs = torch.clamp((cutoff - fmm_dists) / (cutoff + EPS), 0.0, 1.0) # Added EPS to avoid division by zero if cutoff is 0
        return opfs

    def compute_navigable_locations(self):
        nav_locs = {}
        print(f"Computing navigable locations for {self.dset}...")
        for name in tqdm.tqdm(self.names):
            semmap_oh = self.maps[name]
            navmap = semmap_oh[FLOOR_ID] # Assuming FLOOR_ID is 0 and is the floor channel
            ys, xs = np.where(navmap > 0.5) # Use a threshold for binary maps
            nav_locs[name] = (ys, xs)
        return nav_locs

    # ... (keep get_world_coordinates, get_visibility_map, create_spath_based_input_output_pairs, transform_input_output_pairs as they are,
    # they seem general enough if semmap_oh structure is (TotalCategories, H, W))

    def create_spath_based_input_output_pairs(
        self, semmap_oh, fmm_dists_meters, spath, map_xyz_info
    ): # fmm_dists are already in meters
        out_semmap = torch.from_numpy(semmap_oh)
        out_fmm_dists = torch.from_numpy(fmm_dists_meters) # Already in meters
        in_semmap = out_semmap.clone()

        if not spath: # If spath is empty (no navigable path found)
            # Create a dummy visibility map (e.g., all zeros or a small region)
            # This prevents errors in get_visibility_map and subsequent operations
            # Or, decide to skip this sample (might be better)
            # For now, let's create a minimal vis_map if spath is empty
            # This case should ideally be rare if maps are well-formed.
             print(f"Warning: Empty spath for map {map_xyz_info['scene_name']}. Using a dummy visibility region.")
             dummy_loc = [(out_semmap.shape[1] // 2, out_semmap.shape[2] // 2)]
             vis_map = self.get_visibility_map(in_semmap, dummy_loc)

        else:
            vis_map = self.get_visibility_map(in_semmap, spath)

        in_semmap *= vis_map
        # Transform the maps about a random center and rotate by a random angle
        center = random.choice(spath) if spath else (out_semmap.shape[1] // 2, out_semmap.shape[2] // 2)

        rot = random.uniform(-math.pi, math.pi)
        Wby2, Hby2 = out_semmap.shape[2] // 2, out_semmap.shape[1] // 2
        # tform_trans expects (x,y,theta) where x is right, y is down.
        # spath points are (row, col) which is (y, x)
        tform_trans = torch.Tensor([[center[1] - Wby2, center[0] - Hby2, 0]])
        tform_rot = torch.Tensor([[0, 0, rot]])
        (
            in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks
        ) = self.transform_input_output_pairs(
            in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot)
        # Get real-world position and orientation of agent
        world_xyz = self.get_world_coordinates(center, map_xyz_info) # center is (y,x) map coords
        world_heading = -rot # Agent turning leftward is positive in habitat
        scene_name = map_xyz_info['scene_name']
        object_pfs = self.compute_object_pfs(out_fmm_dists) # out_fmm_dists are in meters
        return in_semmap, {
            'semmap': out_semmap,
            'fmm_dists': out_fmm_dists, # meters
            'agent_fmm_dist': agent_fmm_dist, # meters
            'object_pfs': object_pfs, # unitless (0-1)
            'masks': out_masks,
            'world_xyz': world_xyz,
            'world_heading': world_heading,
            'scene_name': scene_name,
        }

    def get_world_coordinates(self, map_yx, world_xyz_info): # map_yx is (y,x)
        shift_xyz = world_xyz_info['world_shift'] # This is (x_shift, y_floor_level, z_shift)
        resolution = world_xyz_info['resolution'] # meters/pixel
        world_y_level = world_xyz_info['y'] # The actual y height of the floor in world coordinates

        # map_yx[0] is map_y (row), map_yx[1] is map_x (col)
        # world_x = map_x * resolution + x_shift
        # world_z = map_y * resolution + z_shift (if map y corresponds to world z)
        # The shifts are usually to bring map origin (0,0) to world origin, then add actual coords.
        # Or, shifts are min_world_coord for that map.
        # Based on create_semantic_maps.py:
        # map_world_shift = central_pos - world_dim / 2
        # vertices -= map_world_shift
        # vertex_to_map_x = (vertices[:, 0] / resolution).round() (world x to map x)
        # vertex_to_map_z = (vertices[:, 2] / resolution).round() (world z to map y/row)
        # So, to go from map (row,col) to world (X,Z):
        # world_x = map_col * resolution + map_world_shift[0]
        # world_z = map_row * resolution + map_world_shift[2]

        world_x_coord = map_yx[1] * resolution + shift_xyz[0]
        world_z_coord = map_yx[0] * resolution + shift_xyz[2]

        world_xyz = (
            world_x_coord,
            world_y_level, # Use the floor's actual y height
            world_z_coord,
        )
        return world_xyz

    def get_visibility_map(self, in_semmap_oh, locations_yx):
        """
        locations_yx - list of [y, x] coordinates
        in_semmap_oh - (C, H, W) tensor
        """
        vis_map = np.zeros(in_semmap_oh.shape[1:], dtype=np.uint8) # (H,W)
        if not locations_yx: # Handle empty list of locations
            return torch.from_numpy(vis_map).float()

        for i in range(len(locations_yx)):
            y, x = locations_yx[i]
            y, x = int(round(y)), int(round(x)) # Ensure integer coords
            if self.cfg.masking_shape == 'square':
                S_cells = int(round(self.visibility_size / self.grid_size / 2.0))
                # Ensure slicing does not go out of bounds
                y_start, y_end = max(0, y - S_cells), min(vis_map.shape[0], y + S_cells + 1) # +1 for exclusive upper bound
                x_start, x_end = max(0, x - S_cells), min(vis_map.shape[1], x + S_cells + 1)
                vis_map[y_start:y_end, x_start:x_end] = 1
            else:
                raise ValueError(f'Masking shape {self.cfg.masking_shape} not defined!')

        vis_map = torch.from_numpy(vis_map).float()
        return vis_map # (H,W)

    def transform_input_output_pairs(
        self, in_semmap_oh, out_semmap_oh, out_fmm_dists_meters, tform_trans, tform_rot
    ):
        # fmm_dists are already in meters
        max_dist = out_fmm_dists_meters[out_fmm_dists_meters != math.inf].max().item() + 1.0 # Add 1 meter margin
        if math.isinf(max_dist): max_dist = 100.0 # A large fallback if all are inf

        # Invert fmm_dists for transformations to handle padding correctly (pad with far away)
        # We want 0 in padding for fmm_dist to become a very large number after inversion,
        # and inf values to become 0.
        inv_fmm_dists = 1.0 / (out_fmm_dists_meters + EPS) # inf becomes 0, small becomes large

        # Expand to add batch dim
        in_semmap = in_semmap_oh.unsqueeze(0)
        out_semmap = out_semmap_oh.unsqueeze(0)
        inv_fmm_dists_b = inv_fmm_dists.unsqueeze(0)

        # Crop a large-enough map around agent
        _, N, H, W = in_semmap.shape
        # tform_trans is (map_x_center - W/2, map_y_center - H/2, 0)
        # crop_center needs to be (map_x_center, map_y_center)
        crop_center_x = tform_trans[:, 0] + W / 2.0
        crop_center_y = tform_trans[:, 1] + H / 2.0
        crop_center = torch.stack([crop_center_x, crop_center_y], dim=1)

        map_size_intermediate_crop = int(round(2.0 * self.cfg.output_map_size / self.grid_size)) # Number of pixels for intermediate crop
        map_size_intermediate_crop = max(map_size_intermediate_crop, int(self.cfg.output_map_size / self.grid_size) + 2) # Ensure it's at least final + buffer

        in_semmap = crop_map(in_semmap, crop_center, map_size_intermediate_crop, mode='nearest')
        out_semmap = crop_map(out_semmap, crop_center, map_size_intermediate_crop, mode='nearest')
        inv_fmm_dists_b = crop_map(inv_fmm_dists_b, crop_center, map_size_intermediate_crop, mode='nearest')

        # Rotate the map
        # spatial_transform_map inverts by default, which is what we want if tform_rot is agent's rotation
        in_semmap = spatial_transform_map(in_semmap, tform_rot, mode='nearest')
        out_semmap = spatial_transform_map(out_semmap, tform_rot, mode='nearest')
        inv_fmm_dists_b = spatial_transform_map(inv_fmm_dists_b, tform_rot, mode='nearest')

        # Crop out the final appropriate size of the map, centered.
        _, N_final, H_final, W_final = in_semmap.shape
        map_center_final = torch.Tensor([[W_final / 2.0, H_final / 2.0]])
        map_size_final_pixels = int(round(self.cfg.output_map_size / self.grid_size))

        in_semmap = crop_map(in_semmap, map_center_final, map_size_final_pixels, mode='nearest')
        out_semmap = crop_map(out_semmap, map_center_final, map_size_final_pixels, mode='nearest')
        inv_fmm_dists_b = crop_map(inv_fmm_dists_b, map_center_final, map_size_final_pixels, mode='nearest')

        # Create loss masks
        # By default, select only regions present in the original semantic map (non-padded)
        # out_semmap is (1, C, H_final, W_final)
        # A simple mask could be where any category is present in the ground truth output map
        out_base_masks = torch.any(out_semmap > 0.5, dim=1, keepdim=True).float() # (1, 1, H_final, W_final)
        out_base_masks = repeat(out_base_masks, '() () h w -> () n h w', n=N_final).float() # N_final is C

        out_masks_for_loss = out_base_masks.clone() # Start with base mask

        if self.cfg.potential_function_masking:
            free_map_input = in_semmap[0, FLOOR_ID].clone() # (H_final, W_final)
            if self.cfg.dilate_free_map:
                free_map_input_b = free_map_input.float().unsqueeze(0).unsqueeze(1) # (1,1,H,W)
                for _ in range(self.cfg.dilate_iters):
                    free_map_input_b = torch.nn.functional.max_pool2d(
                        free_map_input_b, kernel_size=7, stride=1, padding=3
                    )
                free_map_input = free_map_input_b.bool().squeeze(0).squeeze(0) # (H,W)

            exp_map_input = torch.any(in_semmap[0] > 0.5, dim=0) # (H_final, W_final)
            exp_map_input = exp_map_input | free_map_input # Explored is anything seen OR dilated free space
            unk_map_input = (~exp_map_input).numpy()
            frontiers = get_frontiers_np(unk_map_input, free_map_input.numpy())
            frontiers = torch.from_numpy(frontiers).unsqueeze(0).unsqueeze(0).float() # (1,1,H,W)

            frontiers_mask = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3) # (1,1,H,W)
            frontiers_mask = (frontiers_mask > 0.5).float() # Make it binary

            alpha = self.cfg.potential_function_frontier_scaling
            beta = self.cfg.potential_function_non_visible_scaling
            gamma = self.cfg.potential_function_non_frontier_scaling

            visibility_mask = (torch.sum(in_semmap[0], dim=0, keepdim=True) > 0.5).float().unsqueeze(0) # (1,1,H,W)
            visibility_mask = repeat(visibility_mask, '() () h w -> () n h w', n=N_final)
            frontiers_mask_rep = repeat(frontiers_mask, '() () h w -> () n h w', n=N_final)


            not_frontier_and_not_visible = (1.0 - visibility_mask) * (1.0 - frontiers_mask_rep)
            visible_and_not_frontier = visibility_mask * (1.0 - frontiers_mask_rep)

            # Apply scalings. This mask will multiply the loss.
            # Base mask (out_base_masks) ensures we only consider valid map areas.
            loss_region_scalings = (
                visible_and_not_frontier * gamma +
                not_frontier_and_not_visible * beta +
                frontiers_mask_rep * alpha
            )
            out_masks_for_loss = out_base_masks * loss_region_scalings


        # Remove batch dim
        in_semmap = in_semmap.squeeze(0)
        out_semmap = out_semmap.squeeze(0)
        inv_fmm_dists_final = inv_fmm_dists_b.squeeze(0)
        out_masks_for_loss = out_masks_for_loss.squeeze(0) # (C, H, W)

        # Re-invert fmm_dists. inv_fmm_dists had 0 for original inf values.
        # So 1/0 will be inf. Values that were small (large inv_fmm_dist) will become small again.
        out_fmm_dists_meters_transformed = torch.clamp(1.0 / (inv_fmm_dists_final + EPS), 0.0, max_dist)
        # Fill any remaining pure zeros (from perfect inversion of inf) with inf again.
        out_fmm_dists_meters_transformed[inv_fmm_dists_final < EPS/2] = math.inf


        # Compute distance from agent (center of the map patch) to all locations
        nav_map_transformed = out_semmap[FLOOR_ID].cpu().numpy() # (H_final, W_final)
        planner = FMMPlanner(nav_map_transformed)
        agent_map_center = np.zeros(nav_map_transformed.shape, dtype=np.float32)
        H_center, W_center = nav_map_transformed.shape[0] // 2, nav_map_transformed.shape[1] // 2
        # Make a small region around center as start for FMM
        agent_map_center[H_center - 1 : H_center + 2, W_center - 1 : W_center + 2] = 1.0

        # Dilate this small region to ensure it's robustly on navigable space for FMM
        # object_boundary is in meters, convert to pixels
        # selem_radius_agent = int(round((self.object_boundary / 2.0) / self.grid_size))
        # selem_agent = skmp.disk(max(1, selem_radius_agent)) # Ensure radius is at least 1
        # agent_goal_map = skmp.binary_dilation(agent_map_center, selem_agent).astype(np.float32)
        # Instead of dilation, ensure the planner goal is valid
        if not nav_map_transformed[H_center, W_center]: # If center is not navigable, find nearest
            dist_to_nav = skfmm.distance(nav_map_transformed < 0.5, dx=1)
            min_dist_indices = np.unravel_index(np.argmin(dist_to_nav), dist_to_nav.shape)
            agent_map_center = np.zeros_like(nav_map_transformed)
            agent_map_center[min_dist_indices[0], min_dist_indices[1]] = 1.0


        planner.set_multi_goal(agent_map_center, validate_goal=True) # validate_goal to ensure it's on traversible
        agent_fmm_dist_pixels = torch.from_numpy(planner.fmm_dist)
        agent_fmm_dist_meters = agent_fmm_dist_pixels * self.grid_size

        return in_semmap, out_semmap, out_fmm_dists_meters_transformed, agent_fmm_dist_meters, out_masks_for_loss

    @staticmethod
    def _get_palette(dataset_name):
        # Try to get a specific palette
        # This part assumes poni.constants might have HM3D_PALETTE or similar
        # For now, we use a conditional logic based on existing ones.
        if dataset_name == 'gibson':
            return gibson_palette # This is a flat list of RGB values
        elif dataset_name == 'mp3d':
            # Create mp3d palette from d3_40_colors_rgb
            # This should match NUM_OBJECT_CATEGORIES['mp3d']
            # The original code had specific indices for bg, obstacle, free space, etc.
            # For simplicity, let's ensure constants.py can provide a direct palette list for mp3d too
            # Or, use d3_40_colors_rgb directly if categories align
            n_cat_mp3d = NUM_OBJECT_CATEGORIES.get('mp3d', 23) # default to 23 if not found
            palette = [255,255,255, 153,153,153, 242,242,242] # out-of-bounds, obstacle, free
            for color in d3_40_colors_rgb[:n_cat_mp3d-3]: # -3 for the special ones above
                 palette.extend(color.tolist())
            return palette
        elif dataset_name == 'hm3d':
            # !!! ACTION REQUIRED: Define hm3d_palette in poni/constants.py !!!
            # Fallback to a generic palette if HM3D_PALETTE is not defined
            try:
                from poni.constants import hm3d_palette as hm3d_specific_palette
                return hm3d_specific_palette
            except ImportError:
                print(f"Warning: hm3d_palette not found in poni.constants. Falling back to d3_40_colors for HM3D visualization.")
                n_cat_hm3d = NUM_OBJECT_CATEGORIES.get('hm3d', 20) # Example default
                palette = [255,255,255, 153,153,153, 242,242,242]
                for color in d3_40_colors_rgb[:n_cat_hm3d-3]:
                     palette.extend(color.tolist())
                return palette
        else:
            # Fallback for unknown datasets
            print(f"Warning: Unknown dataset '{dataset_name}' for visualization. Using default d3_40 palette.")
            n_cat_default = 20
            palette = [255,255,255, 153,153,153, 242,242,242]
            for color in d3_40_colors_rgb[:n_cat_default-3]:
                 palette.extend(color.tolist())
            return palette

    @staticmethod
    def visualize_map(semmap_oh, bg=1.0, dataset_name='gibson'):
        # semmap_oh is (C, H, W) one-hot tensor/numpy array
        semmap_oh_np = asnumpy(semmap_oh)

        # Compress one-hot map to a category ID map for PIL
        # Assumes channel i corresponds to category ID i
        c_map = np.argmax(semmap_oh_np, axis=0) # Get category ID with max probability
        # If multiple channels have same max prob (e.g. all zeros), argmax gives first.
        # We also need a mask for where *any* category is present vs background.
        any_object_mask = np.sum(semmap_oh_np, axis=0) > 0.5
        c_map[~any_object_mask] = 0 # Assign a background/out-of-bounds ID if nothing is present

        # Get the appropriate palette for the dataset
        # The palette should be a flat list [R1,G1,B1, R2,G2,B2, ...]
        # Palette index 0 = out-of-bounds, 1 = floor, 2 = wall, 3+ = objects
        # The c_map has IDs: 0=floor, 1=wall, 2=chair etc.
        # We need to map these PONI internal IDs to palette indices.
        # A simple way if palette is structured for these IDs directly:
        # Palette index for category `k` is `k`.
        # The special colors for bg, obstacle, free space in the old palette were for a different compression.
        # Let's use OBJECT_CATEGORY_MAP to get colors.

        # Define colors: 0: out-of-bounds, 1: floor, 2: wall, 3..N+2: objects
        # This requires a palette where index matches category ID.
        display_palette = SemanticMapDataset._get_palette(dataset_name)
        # Ensure palette is long enough
        max_cat_id = NUM_OBJECT_CATEGORIES.get(dataset_name,1) -1
        needed_palette_length = (max_cat_id + 1) * 3
        if len(display_palette) < needed_palette_length:
            # Extend palette if too short (e.g. with black or repeating colors)
            print(f"Warning: Palette for {dataset_name} is shorter than needed. Extending with black.")
            display_palette.extend([0,0,0] * (max_cat_id + 1 - len(display_palette)//3))


        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(display_palette) # Palette expects flat list
        semantic_img.putdata((c_map.flatten()).astype(np.uint8)) # Ensure IDs are used as indices
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)

        return semantic_img

    @staticmethod
    def visualize_object_pfs(
        in_semmap_oh, semmap_oh, object_pfs, dirs=None, locs=None, dataset_name='gibson'
    ):
        in_semmap_oh_np = asnumpy(in_semmap_oh)
        semmap_oh_np = asnumpy(semmap_oh)
        # Use the consistent visualize_map for background
        semmap_rgb = SemanticMapDataset.visualize_map(in_semmap_oh, bg=1.0, dataset_name=dataset_name)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255 # Red color for PF
        object_pfs_np = asnumpy(object_pfs)
        vis_ims = []

        num_obj_categories = object_pfs_np.shape[0] # Should match relevant object channels

        for i in range(num_obj_categories):
            # What category ID does channel 'i' of object_pfs correspond to?
            # Assuming object_pfs channels align with actual objects,
            # starting after floor and wall.
            # PONI's CAT_OFFSET = 2 in constants.py (if used consistently) would mean
            # object_pfs channel 0 is for original category ID 2 (e.g. chair).
            # Let's assume object_pfs has C_obj channels, and these map to category IDs
            # CAT_OFFSET through CAT_OFFSET + C_obj - 1.

            actual_category_id = i + CAT_OFFSET # This assumes CAT_OFFSET is correctly set up
                                                # to align object_pfs channels with global category IDs

            opf = object_pfs_np[i][..., np.newaxis]
            sm = np.copy(semmap_rgb)
            smpf = red_image * opf + sm * (1 - opf)
            smpf = smpf.astype(np.uint8)

            # Highlight ground truth object locations for this category
            # semmap_oh_np[actual_category_id] would be the GT mask for this object.
            if actual_category_id < semmap_oh_np.shape[0]: # Check if ID is valid for the GT map
                 gt_object_mask = semmap_oh_np[actual_category_id] > 0.5
                 smpf[gt_object_mask, :] = np.array([0, 0, 255]) # Blue for GT object

            # Highlight directions (if provided)
            if dirs is not None and i < len(dirs) and dirs[i] is not None:
                # Assuming dirs corresponds to the channels of object_pfs
                direction_angle_degrees = dirs[i] # Assuming this is already in degrees
                dir_rad = math.radians(direction_angle_degrees)
                center_x, center_y = sm.shape[1] // 2, sm.shape[0] // 2
                end_x = int(center_x + 50 * math.cos(dir_rad)) # Length 50 pixels
                end_y = int(center_y + 50 * math.sin(dir_rad))
                cv2.line(smpf, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2) # Green line

            # Highlight location targets (if provided)
            if locs is not None and i < len(locs) and locs[i] is not None:
                # Assuming locs corresponds to the channels of object_pfs
                # locs are (x, y) normalized between 0-1
                H_viz, W_viz = semmap_rgb.shape[:2]
                loc_x_norm, loc_y_norm = locs[i]
                if loc_x_norm >= 0 and loc_y_norm >= 0: # Check for valid loc (-1 means no object)
                    target_x = int(loc_x_norm * W_viz)
                    target_y = int(loc_y_norm * H_viz)
                    cv2.circle(smpf, (target_x, target_y), 3, (0, 255, 0), -1) # Green circle

            vis_ims.append(smpf)
        return vis_ims

    @staticmethod
    def visualize_object_category_pf(semmap_oh, object_pfs, cat_display_idx, dset_name):
        # semmap_oh is one-hot (C, H, W)
        # object_pfs is one-hot (C_obj, H, W) or (C_total, H, W)
        # cat_display_idx is the index relative to the list of *displayed* object categories
        # (i.e., after floor and wall are excluded).
        # We need to map this back to the actual category ID in OBJECT_CATEGORY_MAP

        semmap_oh_np = asnumpy(semmap_oh)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap_oh, bg=1.0, dataset_name=dset_name)

        # This is tricky: object_pfs might have C_total channels or C_obj channels.
        # Let's assume object_pfs has channels corresponding to actual objects,
        # so channel `k` in object_pfs corresponds to category ID `k + CAT_OFFSET`.
        # cat_display_idx refers to the k-th object. So the channel in object_pfs is cat_display_idx.
        # The actual category ID is cat_display_idx + CAT_OFFSET.
        object_pf_channel_idx = cat_display_idx
        actual_category_id = cat_display_idx + CAT_OFFSET

        object_pfs_np = asnumpy(object_pfs)
        if object_pf_channel_idx >= object_pfs_np.shape[0]:
            print(f"Warning: cat_display_idx {cat_display_idx} is out of bounds for object_pfs with shape {object_pfs_np.shape}")
            return semmap_rgb # Return base map if index is bad

        pf_for_category = object_pfs_np[object_pf_channel_idx][..., np.newaxis] # (H, W, 1)

        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        smpf = red_image * pf_for_category + semmap_rgb * (1 - pf_for_category)
        smpf = smpf.astype(np.uint8)

        # Highlight the GT location of this specific category
        if actual_category_id < semmap_oh_np.shape[0]:
            gt_object_mask = semmap_oh_np[actual_category_id] > 0.5
            smpf[gt_object_mask, :] = np.array([0, 0, 255]) # Blue for GT object

        return smpf

    @staticmethod
    def visualize_area_pf(semmap_oh, area_pfs, dset_name='gibson'):
        # semmap_oh is (C,H,W), area_pfs is (1,H,W) or (H,W)
        semmap_oh_np = asnumpy(semmap_oh)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap_oh_np, bg=1.0, dataset_name=dset_name)
        pfs_np = asnumpy(area_pfs).squeeze() # Ensure it's (H, W)
        pfs_np = pfs_np[..., np.newaxis] # (H, W, 1)

        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255 # Red for PF
        smpf = red_image * pfs_np + semmap_rgb * (1 - pfs_np)
        smpf = smpf.astype(np.uint8)

        return smpf

    @staticmethod
    def combine_image_grid(
        in_semmap_oh, out_semmap_oh, gt_object_pfs, pred_object_pfs=None,
        gt_acts=None, gt_area_pfs=None, pred_area_pfs=None, dset_name=None,
        n_per_row=8, pad=2, border_color=200, output_width=1024,
    ):
        # Ensure dset_name is provided for category name lookups
        if dset_name is None:
            raise ValueError("dset_name must be provided for combine_image_grid")

        # Visualize input and output maps
        in_map_rgb = SemanticMapDataset.visualize_map(in_semmap_oh, dataset_name=dset_name)
        out_map_rgb = SemanticMapDataset.visualize_map(out_semmap_oh, dataset_name=dset_name)

        img_and_titles = [
            (in_map_rgb, 'Input map'), (out_map_rgb, 'Full output map')
        ]

        if gt_area_pfs is not None:
            gt_area_rgb = SemanticMapDataset.visualize_area_pf(in_semmap_oh, gt_area_pfs, dset_name=dset_name)
            img_and_titles.append((gt_area_rgb, 'GT Area map'))
        if pred_area_pfs is not None:
            pred_area_rgb = SemanticMapDataset.visualize_area_pf(in_semmap_oh, pred_area_pfs, dset_name=dset_name)
            img_and_titles.append((pred_area_rgb, 'Pred Area map'))

        # Visualize object PFs
        # gt_object_pfs has C_obj channels. Channel i corresponds to category (i + CAT_OFFSET)
        num_obj_channels_in_pfs = gt_object_pfs.shape[0]

        for obj_channel_idx in range(num_obj_channels_in_pfs):
            actual_cat_id = obj_channel_idx + CAT_OFFSET
            if actual_cat_id not in INV_OBJECT_CATEGORY_MAP[dset_name]:
                cat_name = f"UnknownCatID_{actual_cat_id}"
            else:
                cat_name = INV_OBJECT_CATEGORY_MAP[dset_name][actual_cat_id]

            if cat_name in ['floor', 'wall']: # Skip floor and wall PFs if not meaningful
                continue

            acts_suffix = ''
            if gt_acts is not None and obj_channel_idx < len(gt_acts):
                acts_suffix = f'(act: {gt_acts[obj_channel_idx].item():d})' # Assuming gt_acts aligns with obj_pf channels

            # Visualize GT PF for this object channel
            # visualize_object_category_pf expects cat_display_idx to be the index for obj after floor/wall
            gt_pf_vis = SemanticMapDataset.visualize_object_category_pf(
                in_semmap_oh, gt_object_pfs, obj_channel_idx, dset_name
            )
            img_and_titles.append((gt_pf_vis, 'GT PF for ' + cat_name + acts_suffix))

            if pred_object_pfs is not None:
                pred_pf_vis = SemanticMapDataset.visualize_object_category_pf(
                    in_semmap_oh, pred_object_pfs, obj_channel_idx, dset_name
                )
                img_and_titles.append((pred_pf_vis, 'Pred PF for ' + cat_name + acts_suffix))


        imgs = []
        for img_data, title in img_and_titles:
            cimg = SemanticMapDataset.add_title_to_image(img_data, title)
            # Pad image
            cimg = np.pad(cimg, ((pad, pad), (pad, pad), (0, 0)),
                          mode='constant', constant_values=border_color)
            imgs.append(cimg)

        # ... (rest of the grid combination logic remains the same)
        n_rows = (len(imgs) + n_per_row - 1) // n_per_row # ceiling division
        n_cols = min(len(imgs), n_per_row)
        if not imgs: # Handle case with no images
            return np.zeros((100,100,3), dtype=np.uint8) # Return small black image

        H, W = imgs[0].shape[:2]
        grid_img = np.full((n_rows * H, n_cols * W, 3), border_color, dtype=np.uint8) # Fill with border color
        for i, img in enumerate(imgs):
            r = i // n_per_row
            c = i % n_per_row
            grid_img[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
        # Rescale image
        if output_width is not None and grid_img.shape[1] > 0 : # Add check for W > 0
            output_height = int(
                output_width * grid_img.shape[0] / grid_img.shape[1]
            )
            grid_img = cv2.resize(grid_img, (output_width, output_height))
        return grid_img

    @staticmethod
    def add_title_to_image(
        img: np.ndarray, title: str, font_size: int = 50, bg_color=200,
        fg_color=(0, 0, 0) # Black text on light BG
    ):
        font_img = np.full((font_size, img.shape[1], 3), bg_color, dtype=np.uint8)
        font_img_pil = Image.fromarray(font_img)
        draw = ImageDraw.Draw(font_img_pil)
        try:
            mpl_font = font_manager.FontProperties(family="sans-serif", weight="normal") # Changed to normal weight
            font_path = font_manager.findfont(mpl_font)
            font = ImageFont.truetype(font=font_path, size=20) # Adjusted size
        except: # Fallback font
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0,0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center text
        text_x = (img.shape[1] - text_width) // 2
        text_y = (font_size - text_height) // 2

        draw.text((text_x, text_y), title, fill=fg_color, font=font)
        font_img_np = np.array(font_img_pil)
        return np.concatenate([font_img_np, img], axis=0)


class SemanticMapPrecomputedDataset(SemanticMapDataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.dset = cfg.dset_name
        if self.dset not in NUM_OBJECT_CATEGORIES: # Ensure dset name is valid early
            raise ValueError(f"Dataset name '{self.dset}' not found in PONI constants. Please define it in poni/constants.py.")

        # Seed the dataset
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        # Load map paths from precomputed dataset directory
        # The root in cfg for PrecomputedDataset should point to the SAVE_ROOT of create_poni_dataset.py
        self.map_paths = sorted(
            glob.glob(osp.join(cfg.root, split, "**/*.pbz2"), recursive=True)
        )
        if not self.map_paths:
            print(f"Warning: No precomputed data found for {self.dset} split {split} at {osp.join(cfg.root, split)}")

        # Both locations and directions cannot be enabled at the same time.
        assert not (self.cfg.enable_locations and self.cfg.enable_directions)

    def __len__(self):
        return len(self.map_paths)

    # compute_object_pfs is inherited from SemanticMapDataset

    def __getitem__(self, idx):
        try:
            with bz2.BZ2File(self.map_paths[idx], 'rb') as fp:
                data = cPickle.load(fp)
        except Exception as e:
            print(f"Error loading precomputed file {self.map_paths[idx]}: {e}")
            # Return a dummy sample or raise error, depending on desired behavior
            # For now, let's try to get another sample if this one fails, or raise error if too many fails
            if len(self.map_paths) > 1:
                print("Attempting to load a different sample.")
                return self.__getitem__((idx + 1) % len(self.map_paths)) # Try next, wrap around
            else:
                raise e


        # fmm_dists in precomputed data was saved as int32 after *100.0, convert back to float meters
        fmm_dists_meters = data['fmm_dists'].astype(np.float32) / 100.0

        # in_semmap and semmap are already one-hot boolean, convert to float
        in_semmap_oh = torch.from_numpy(data['in_semmap'].astype(np.float32))
        out_semmap_oh = torch.from_numpy(data['semmap'].astype(np.float32))
        fmm_dists_tensor = torch.from_numpy(fmm_dists_meters)

        # Compute object_pfs (unitless, 0-1 range)
        # This uses self.cfg.object_pf_cutoff_dist which should be in meters
        object_pfs_unitless = self.compute_object_pfs(fmm_dists_tensor) # Expects fmm_dists in meters

        # Get masks and labels (directions, locations, area_pfs, actions)
        # This needs the one-hot input map, GT output map, and GT FMM distances (in meters)
        loss_masks, masks_for_pf_scaling, dirs, locs, area_pfs_unitless, acts, _ = self.get_masks_and_labels(
            in_semmap_oh, out_semmap_oh, fmm_dists_tensor
        ) # _ is contours, not used here

        if self.cfg.potential_function_masking:
            # Apply the scaling mask to the PFs.
            # object_pfs is already (0-1). Multiplying by mask_for_pf_scaling which has values (alpha, beta, gamma)
            # effectively scales the PFs themselves, not just the loss.
            object_pfs_scaled = torch.clamp(object_pfs_unitless * masks_for_pf_scaling, 0.0, 1.0)
        else:
            object_pfs_scaled = object_pfs_unitless


        input_data = {'semmap': in_semmap_oh} # Model expects 'semmap' key for input

        # Prepare labels for training
        # Convert PFs to int16 (0 -> 1000) for memory optimization if needed, as done in original.
        # Loss calculation will need to convert them back to float / 1000.0.
        # Current train.py already does this: labels["object_pfs"].float() / 1000.0
        label_data = {
            'semmap': out_semmap_oh, # GT full semantic map (one-hot)
            'object_pfs': (object_pfs_scaled * 1000.0).short(), # Store as short int
            'loss_masks': loss_masks, # This mask is used in train.py to scale the loss per pixel/category
        }

        if area_pfs_unitless is not None:
            # area_pfs_unitless is also 0-1. Scale and store as short int.
            label_data['area_pfs'] = (area_pfs_unitless * 1000.0).short()
        if dirs is not None:
            label_data['dirs'] = dirs.long() # Ensure it's LongTensor for CrossEntropy
        if locs is not None:
            label_data['locs'] = locs # FloatTensor (normalized x,y)
        if acts is not None:
            label_data['acts'] = acts.long() # Ensure it's LongTensor

        # Free memory
        del data
        # gc.collect() # gc.collect can be slow, use sparingly or if memory issues persist
        return input_data, label_data

    # get_masks_and_labels is inherited and should work if constants are set up for HM3D
    # and if `convert_maps_to_oh` produces maps correctly.