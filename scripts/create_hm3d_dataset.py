#!/usr/bin/env python3
"""
Script to create HM3D dataset in the format expected by PONI.
This extends the existing dataset creation pipeline for HM3D data.
"""

import os
import argparse
import glob
import multiprocessing as mp
import bz2
import _pickle as cPickle
import numpy as np
import tqdm
from typing import Dict, List

from poni.default import get_cfg
from poni.dataset import SemanticMapDataset
from poni.fmm_planner import FMMPlanner

# Set environment variable for the active dataset
os.environ["ACTIVE_DATASET"] = "hm3d"
DATASET = "hm3d"
OUTPUT_MAP_SIZE = 24.0
MASKING_MODE = "spath"
MASKING_SHAPE = "square"
SEED = 123
DATA_ROOT = f"data/semantic_maps/{DATASET}/semantic_maps"
FMM_DISTS_SAVED_ROOT = f"data/semantic_maps/{DATASET}/fmm_dists_{SEED}"
NUM_SAMPLES = {'train': 400000, 'val': 1000}
SAVE_ROOT = f"data/semantic_maps/{DATASET}/precomputed_dataset_{OUTPUT_MAP_SIZE}_{SEED}_{MASKING_MODE}_{MASKING_SHAPE}"


def setup_parser():
    parser = argparse.ArgumentParser(description="Create HM3D dataset for PONI")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--map-id', type=int, default=-1,
                        help="Process only a specific map ID")
    parser.add_argument('--map-id-range', type=int, nargs="+", default=None,
                        help="Process maps in a specific ID range")
    parser.add_argument('--num-workers', type=int, default=8,
                        help="Number of parallel workers")
    return parser


def precompute_fmm_dists():
    """Pre-compute FMM distances for all maps."""
    cfg = get_cfg()
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.fmm_dists_saved_root = ""
    cfg.freeze()
    
    os.makedirs(FMM_DISTS_SAVED_ROOT, exist_ok=True)
    pool = mp.Pool(8)  # Adjust based on your system
    
    for split in ["val", "train"]:
        print(f"=====> Computing FMM dists for {split} split")
        dataset = SemanticMapDataset(cfg.DATASET, split=split)
        print("--> Saving FMM dists")
        inputs = []
        for name in dataset.names:
            save_path = os.path.join(FMM_DISTS_SAVED_ROOT, f"{name}.pbz2")
            data = dataset.fmm_dists[name]
            inputs.append((data, save_path))
        
        _ = list(tqdm.tqdm(pool.imap(save_data, inputs), total=len(inputs)))


def save_data(inputs):
    """Helper function to save FMM data."""
    data, path = inputs
    with bz2.BZ2File(path, "w") as fp:
        cPickle.dump(data, fp)


def precompute_dataset_for_map(kwargs):
    """Process a single map to create the PONI dataset."""
    cfg = kwargs["cfg"]
    split = kwargs["split"]
    name = kwargs["name"]
    n_samples_per_map = kwargs["n_samples_per_map"]
    save_root = kwargs["save_root"]

    dataset = SemanticMapDataset(
        cfg.DATASET, split=split, scf_name=name, seed=SEED
    )
    print(f'====> Pre-computing for map {name}')
    os.makedirs(f'{save_root}/{name}', exist_ok=True)
    
    for i in range(n_samples_per_map):
        input_data, label = dataset.get_item_by_name(name)
        save_path = f'{save_root}/{name}/sample_{i:05d}.pbz2'
        in_semmap = np.array(input_data) > 0.5  # (N, H, W)
        semmap = np.array(label['semmap']) > 0.5  # (N, H, W)
        fmm_dists = np.array(label['fmm_dists']).astype(np.float32)  # (N, H, W)
        world_xyz = np.array(label['world_xyz'])  # (3, )
        world_heading = np.array([label['world_heading']])  # (1, )
        scene_name = np.array(label['scene_name'])
        
        # Convert to int maps to save space
        fmm_dists = (fmm_dists * 100.0).astype(np.int32)
        
        with bz2.BZ2File(save_path, 'w') as fp:
            cPickle.dump(
                {
                    'in_semmap': in_semmap,
                    'semmap': semmap,
                    'fmm_dists': fmm_dists,
                    'scene_name': scene_name,
                    'world_xyz': world_xyz,
                    'world_heading': world_heading,
                },
                fp
            )


def precompute_dataset(args):
    """Precompute dataset for all maps or selected maps."""
    cfg = get_cfg()
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.output_map_size = OUTPUT_MAP_SIZE
    cfg.DATASET.fmm_dists_saved_root = FMM_DISTS_SAVED_ROOT
    cfg.DATASET.masking_mode = MASKING_MODE
    cfg.DATASET.masking_shape = MASKING_SHAPE
    cfg.DATASET.visibility_size = 3.0  # m
    cfg.freeze()

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, args.split), exist_ok=True)
    
    dataset = SemanticMapDataset(cfg.DATASET, split=args.split)
    n_maps = len(dataset)
    print(f'Maps: {n_maps}')
    n_samples_per_map = (NUM_SAMPLES[args.split] // n_maps) + 1

    if args.map_id != -1:
        map_names = [dataset.names[args.map_id]]
    elif args.map_id_range is not None:
        assert len(args.map_id_range) == 2
        map_names = [
            dataset.names[i]
            for i in range(args.map_id_range[0], args.map_id_range[1] + 1)
        ]
    else:
        map_names = dataset.names

    pool = mp.Pool(processes=args.num_workers)
    inputs = []
    for name in map_names:
        kwargs = {
            "cfg": cfg,
            "split": args.split,
            "name": name,
            "n_samples_per_map": n_samples_per_map,
            "save_root": f'{SAVE_ROOT}/{args.split}',
        }
        inputs.append(kwargs)
    
    with tqdm.tqdm(total=len(inputs)) as pbar:
        for _ in pool.imap_unordered(precompute_dataset_for_map, inputs):
            pbar.update()


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Ensure that only one of map-id or map-id-range is specified
    assert (args.map_id == -1) or (args.map_id_range is None), \
        "Cannot specify both map-id and map-id-range"
    
    # Check if FMM distances are already computed
    fmm_files = glob.glob(os.path.join(FMM_DISTS_SAVED_ROOT, "*.pbz2"))
    if len(fmm_files) == 0:
        print("FMM distances not found. Computing them first...")
        precompute_fmm_dists()
    
    # Precompute the dataset
    precompute_dataset(args)
    print(f"Dataset creation complete! Saved to {SAVE_ROOT}")


if __name__ == "__main__":
    main()