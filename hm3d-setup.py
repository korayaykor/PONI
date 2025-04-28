#!/usr/bin/env python3
"""
Modified version of create_semantic_maps.py with added debugging
"""

import os
import glob
import json
import multiprocessing as mp
import tqdm
import re
import sys

# Debug function to check directories and files
def debug_directories():
    poni_root = os.environ.get('PONI_ROOT', '/app/PONI')
    print("DEBUG: PONI_ROOT =", poni_root)
    
    # Check all relevant directories
    dirs_to_check = [
        "data/scene_datasets/hm3d",
        "data/scene_datasets/hm3d/train",
        "data/scene_datasets/hm3d/val",
        "data/scene_datasets/hm3d_uncompressed"
    ]
    
    for dir_path in dirs_to_check:
        full_path = os.path.join(poni_root, dir_path)
        if os.path.exists(full_path):
            print(f"✓ Directory exists: {full_path}")
            # List first few files
            files = os.listdir(full_path)
            print(f"  Contains {len(files)} files/directories")
            for i, file in enumerate(files[:5]):
                print(f"    - {file}")
            if len(files) > 5:
                print(f"    - ... and {len(files)-5} more")
        else:
            print(f"✗ Directory MISSING: {full_path}")
    
    # Check if HM3D is in constants.py
    constants_path = os.path.join(poni_root, "poni/constants.py")
    if os.path.exists(constants_path):
        with open(constants_path, 'r') as f:
            content = f.read()
            if "\"hm3d\"" in content:
                print("✓ Found 'hm3d' entry in constants.py")
            else:
                print("✗ NO 'hm3d' entry found in constants.py!")
    else:
        print(f"✗ Constants file not found: {constants_path}")

# Run debugging
print("="*50)
print("DEBUGGING INFORMATION")
print("="*50)
debug_directories()
print("="*50)

# Now run a modified version of the original script
import glob
import json
import multiprocessing as mp
import os
import random

from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
import tqdm
import trimesh
from PIL import Image, ImageDraw, ImageFont
from torch_scatter import scatter_max

Image.MAX_IMAGE_PIXELS = 1000000000
from matplotlib import font_manager
from plyfile import PlyData

# Make sure we're working with the correct dataset
assert 'ACTIVE_DATASET' in os.environ
ACTIVE_DATASET = os.environ['ACTIVE_DATASET']
print(f"Active Dataset: {ACTIVE_DATASET}")

# Check constants import
try:
    from poni.constants import d3_40_colors_rgb, OBJECT_CATEGORIES, SPLIT_SCENES
    print(f"✓ Successfully imported from poni.constants")
    print(f"  - SPLIT_SCENES keys: {list(SPLIT_SCENES.keys())}")
    if 'hm3d' in SPLIT_SCENES:
        print(f"  - HM3D train scenes: {len(SPLIT_SCENES['hm3d']['train'])}")
        print(f"  - HM3D val scenes: {len(SPLIT_SCENES['hm3d']['val'])}")
    else:
        print(f"✗ 'hm3d' not found in SPLIT_SCENES!")
except ImportError as e:
    print(f"✗ Error importing from poni.constants: {e}")
    sys.exit(1)

# Original script constants
HM3D_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["hm3d"]
HM3D_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(HM3D_CATEGORIES)}
HM3D_OBJECT_COLORS = []  # Excludes 'out-of-bounds', 'floor', and 'wall'
for color in d3_40_colors_rgb[: len(HM3D_CATEGORIES) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    HM3D_OBJECT_COLORS.append(color)

# General constants
OBJECT_COLORS = HM3D_OBJECT_COLORS
OBJECT_CATEGORIES = HM3D_CATEGORIES
OBJECT_CATEGORY_MAP = HM3D_CATEGORY_MAP

# Check if PONI_ROOT is set correctly
PONI_ROOT = os.environ.get('PONI_ROOT', '/app/PONI')
print(f"PONI_ROOT = {PONI_ROOT}")

# Check various possible scene root directories
possible_scene_roots = [
    "data/scene_datasets/hm3d_uncompressed",
    "data/scene_datasets/hm3d",
    "data/scene_datasets/hm3d/train",
    "data/scene_dataset/hm3d/train"
]

for root in possible_scene_roots:
    full_path = os.path.join(PONI_ROOT, root)
    if os.path.exists(full_path):
        print(f"Checking for .glb files in: {full_path}")
        glb_files = glob.glob(os.path.join(full_path, "**/*.glb"), recursive=True)
        print(f"  Found {len(glb_files)} .glb files")
        if len(glb_files) > 0:
            print(f"  First few GLB files:")
            for file in glb_files[:3]:
                print(f"    - {file}")

# Try each possible scene root and find GLB files
SCENES_ROOT = os.path.join(PONI_ROOT, "data/scene_datasets/hm3d_uncompressed")
SB_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/scene_boundaries")
PC_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/point_clouds")
SEM_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/semantic_maps")

# Find all GLB files in all possible roots
all_glb_files = []
for root in possible_scene_roots:
    full_path = os.path.join(PONI_ROOT, root)
    if os.path.exists(full_path):
        glb_files = glob.glob(os.path.join(full_path, "**/*.glb"), recursive=True)
        all_glb_files.extend(glb_files)

print(f"Total GLB files found: {len(all_glb_files)}")

# We need to manually populate valid_scenes since we're not sure which directories work
valid_scenes = []
for scene_path in all_glb_files:
    scene_name = os.path.basename(scene_path).split(".")[0]
    if scene_name in SPLIT_SCENES.get('hm3d', {}).get('train', []) or scene_name in SPLIT_SCENES.get('hm3d', {}).get('val', []):
        valid_scenes.append(scene_name)
        print(f"✓ Found valid scene: {scene_name}")

if not valid_scenes:
    print("ERROR: No valid HM3D scenes found that match entries in SPLIT_SCENES!")
    print("You need to update poni/constants.py with your HM3D scene names.")
    sys.exit(1)

print(f"Number of available scenes: {len(valid_scenes)}")
print("===========> Extracting scene boundaries")
print("Stopping here - add proper scene entries to constants.py and try again")