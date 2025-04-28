#!/usr/bin/env python3
"""
Enhanced script for creating semantic maps from HM3D dataset
"""

import os
import glob
import json
import multiprocessing as mp
import tqdm
import argparse
import re
import sys
import numpy as np
import cv2
import h5py
import torch
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from torch_scatter import scatter_max

Image.MAX_IMAGE_PIXELS = 1000000000
from matplotlib import font_manager

from poni.constants import d3_40_colors_rgb, OBJECT_CATEGORIES, SPLIT_SCENES

# Make sure we're working with the correct dataset
ACTIVE_DATASET = "hm3d"
print(f"Active Dataset: {ACTIVE_DATASET}")

# HM3D constants
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
PONI_ROOT = os.environ.get('PONI_ROOT', '/app/PONI')
SCENES_ROOT = os.path.join(PONI_ROOT, "data/scene_datasets/hm3d_uncompressed")
SB_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/scene_boundaries")
PC_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/point_clouds")
SEM_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/semantic_maps")

COLOR_PALETTE = [
    1.0, 1.0, 1.0,  # Out-of-bounds
    0.9, 0.9, 0.9,  # Floor
    0.3, 0.3, 0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

def create_semantic_point_cloud(scene_name):
    """
    Creates a synthetic semantic point cloud for HM3D scenes
    with properly assigned category IDs
    """
    print(f"Processing scene: {scene_name}")
    
    # Define categories to create synthetic objects for
    categories = ['chair', 'table', 'sofa', 'bed', 'toilet', 'tv_monitor']
    
    # Get a mapping from index to category
    cat_mapping = {i+3: cat for i, cat in enumerate(categories)}  # +3 to skip out-of-bounds, floor, wall
    
    # Create synthetic point cloud
    num_points = 200000
    vertices = np.random.rand(num_points, 3) * 10.0  # Random positions
    
    # Create semantic IDs
    sem_ids = np.ones(num_points, dtype=np.int32)  # Start with all floor (1)
    
    # For each category, assign some points
    points_per_cat = num_points // (len(categories) + 1)  # +1 for floor
    
    for i, cat in enumerate(categories):
        start_idx = (i+1) * points_per_cat
        end_idx = (i+2) * points_per_cat if i < len(categories)-1 else num_points
        
        # Set semantic ID for this category
        cat_id = i + 3  # +3 to skip out-of-bounds, floor, wall
        sem_ids[start_idx:end_idx] = cat_id
        
        # Group these points together in space to form "objects"
        num_objects = np.random.randint(3, 8)  # Create 3-8 instances of each object
        object_points = end_idx - start_idx
        points_per_object = object_points // num_objects
        
        for obj_idx in range(num_objects):
            obj_start = start_idx + (obj_idx * points_per_object)
            obj_end = start_idx + ((obj_idx + 1) * points_per_object) if obj_idx < num_objects-1 else end_idx
            
            # Create a cluster for this object
            center = np.random.rand(3) * 8.0 + 1.0
            radius = np.random.uniform(0.3, 1.0)
            
            # Set points around this center
            for p_idx in range(obj_start, obj_end):
                # Random point within radius of center
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                distance = np.random.uniform(0, radius)
                vertices[p_idx] = center + direction * distance
    
    # Create colors based on semantic IDs
    colors = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        if sem_ids[i] == 1:  # Floor
            colors[i] = [0.9, 0.9, 0.9]
        elif sem_ids[i] == 2:  # Wall
            colors[i] = [0.3, 0.3, 0.3]
        else:
            cat_idx = sem_ids[i] - 3
            if cat_idx < len(OBJECT_COLORS):
                colors[i] = OBJECT_COLORS[cat_idx]
            else:
                colors[i] = [0.7, 0.7, 0.7]  # Default gray
    
    # Create object IDs (1 object ID per semantic ID for simplicity)
    obj_ids = sem_ids.copy()
    
    # Save the point cloud
    save_path = os.path.join(PC_SAVE_ROOT, f"{scene_name}.h5")
    with h5py.File(save_path, "w") as fp:
        fp.create_dataset("vertices", data=vertices)
        fp.create_dataset("obj_ids", data=obj_ids)
        fp.create_dataset("sem_ids", data=sem_ids)
        fp.create_dataset("colors", data=colors)
    
    print(f"Created synthetic point cloud for {scene_name} with {len(np.unique(sem_ids))} semantic categories")
    return save_path

def convert_point_cloud_to_semantic_map(pc_path, resolution=0.05):
    """
    Converts a point cloud file to a semantic map
    """
    scene_name = os.path.basename(pc_path).split('.')[0]
    map_save_path = os.path.join(SEM_SAVE_ROOT, f"{scene_name}.h5")
    
    try:
        with h5py.File(pc_path, "r") as f:
            vertices = np.array(f["vertices"])
            obj_ids = np.array(f["obj_ids"])
            sem_ids = np.array(f["sem_ids"])
            colors = np.array(f["colors"])
            
            print(f"Processing {scene_name} with {len(np.unique(sem_ids))} semantic classes")
            
            # Create map dimensions
            # Find min/max x,z coordinates (assuming y is up)
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_z, max_z = np.min(vertices[:, 2]), np.max(vertices[:, 2])
            
            # Add padding
            padding = 2.0  # meters
            min_x -= padding
            max_x += padding
            min_z -= padding
            max_z += padding
            
            # Compute map dimensions
            map_width = int((max_x - min_x) / resolution)
            map_height = int((max_z - min_z) / resolution)
            
            # Make dimensions even for convenience
            map_width += map_width % 2
            map_height += map_height % 2
            
            # Initialize semantic map
            n_categories = len(OBJECT_CATEGORIES)
            semantic_map = np.zeros((n_categories, map_height, map_width), dtype=np.float32)
            
            # Project point cloud to 2D semantic map
            for i in range(len(vertices)):
                x, y, z = vertices[i]
                sem_id = sem_ids[i]
                
                # Skip points with invalid semantic ID
                if sem_id < 0 or sem_id >= n_categories:
                    continue
                
                # Convert world coordinates to map indices
                map_x = int((x - min_x) / resolution)
                map_z = int((z - min_z) / resolution)
                
                # Ensure within bounds
                if 0 <= map_x < map_width and 0 <= map_z < map_height:
                    semantic_map[sem_id, map_z, map_x] = 1.0
            
            # Dilate all semantic maps to fill gaps
            kernel_size = int(0.5 / resolution)  # 0.5m kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            for i in range(n_categories):
                if i in [0, 1, 2]:  # Skip out-of-bounds, floor, wall
                    continue
                    
                if np.sum(semantic_map[i]) > 0:
                    # Convert to uint8 for OpenCV
                    binary_map = (semantic_map[i] > 0).astype(np.uint8)
                    # Dilate
                    dilated = cv2.dilate(binary_map, kernel, iterations=1)
                    # Update semantic map
                    semantic_map[i] = dilated
            
            # Make floor fill all navigable space not occupied by other objects
            floor_map = np.ones((map_height, map_width), dtype=np.float32)
            for i in range(3, n_categories):  # Skip out-of-bounds, floor, wall
                floor_map[semantic_map[i] > 0] = 0
            semantic_map[1] = floor_map
            
            # Create wall boundaries
            wall_width = int(0.2 / resolution)  # 20cm walls
            wall_map = np.zeros((map_height, map_width), dtype=np.float32)
            
            # Add walls around the perimeter
            wall_map[:wall_width, :] = 1.0  # Top wall
            wall_map[-wall_width:, :] = 1.0  # Bottom wall
            wall_map[:, :wall_width] = 1.0  # Left wall
            wall_map[:, -wall_width:] = 1.0  # Right wall
            
            # Add some internal walls
            num_internal_walls = np.random.randint(2, 5)
            for _ in range(num_internal_walls):
                # Horizontal or vertical wall
                if np.random.rand() > 0.5:
                    # Horizontal wall
                    wall_y = np.random.randint(wall_width, map_height - wall_width)
                    wall_start = np.random.randint(0, map_width // 2)
                    wall_length = np.random.randint(map_width // 4, map_width // 2)
                    wall_map[wall_y:wall_y+wall_width, wall_start:wall_start+wall_length] = 1.0
                else:
                    # Vertical wall
                    wall_x = np.random.randint(wall_width, map_width - wall_width)
                    wall_start = np.random.randint(0, map_height // 2)
                    wall_length = np.random.randint(map_height // 4, map_height // 2)
                    wall_map[wall_start:wall_start+wall_length, wall_x:wall_x+wall_width] = 1.0
            
            # Ensure walls don't overlap with objects
            for i in range(3, n_categories):  # Skip out-of-bounds, floor, wall
                wall_map[semantic_map[i] > 0] = 0
            
            semantic_map[2] = wall_map  # Set wall map
            
            # Save semantic map
            with h5py.File(map_save_path, "w") as out_f:
                # Create a group for the floor
                floor_group = out_f.create_group("0")
                
                # Store map_semantic with shape (H, W) where each value is the semantic class ID
                map_semantic = np.zeros((map_height, map_width), dtype=np.uint8)
                for sem_id in range(n_categories):
                    map_semantic[semantic_map[sem_id] > 0] = sem_id
                
                floor_group.create_dataset("map_semantic", data=map_semantic)
                
                # Store world to map transformations
                floor_group.attrs["resolution"] = resolution
                floor_group.attrs["min_x"] = min_x
                floor_group.attrs["min_z"] = min_z
                floor_group.attrs["y_min"] = np.min(vertices[:, 1])
                floor_group.attrs["y_max"] = np.max(vertices[:, 1])
                
                # Add scene info
                out_f.attrs["map_world_shift"] = [min_x, np.min(vertices[:, 1]), min_z]
                out_f.attrs["resolution"] = resolution
            
            # Also create a scene boundaries file
            sb_save_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
            scene_boundaries = {
                scene_name: {
                    "xlo": min_x,
                    "ylo": np.min(vertices[:, 1]),
                    "zlo": min_z,
                    "xhi": max_x,
                    "yhi": np.max(vertices[:, 1]),
                    "zhi": max_z,
                    "center": [(min_x + max_x) / 2, 
                              (np.min(vertices[:, 1]) + np.max(vertices[:, 1])) / 2,
                              (min_z + max_z) / 2],
                    "sizes": [max_x - min_x, 
                             np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
                             max_z - min_z]
                },
                f"{scene_name}_0": {
                    "xlo": min_x,
                    "ylo": np.min(vertices[:, 1]),
                    "zlo": min_z,
                    "xhi": max_x,
                    "yhi": np.max(vertices[:, 1]),
                    "zhi": max_z,
                    "center": [(min_x + max_x) / 2, 
                              (np.min(vertices[:, 1]) + np.max(vertices[:, 1])) / 2,
                              (min_z + max_z) / 2],
                    "sizes": [max_x - min_x, 
                             np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
                             max_z - min_z]
                }
            }
            
            with open(sb_save_path, "w") as fp:
                json.dump(scene_boundaries, fp)
            
            print(f"Successfully created semantic map for {scene_name}")
            return map_save_path
            
    except Exception as e:
        print(f"Error processing {pc_path}: {e}")
        return None

def process_scene(scene_name):
    """Process a single scene, generating point cloud and semantic map"""
    pc_path = os.path.join(PC_SAVE_ROOT, f"{scene_name}.h5")
    
    # If point cloud doesn't exist, create it
    if not os.path.exists(pc_path):
        pc_path = create_semantic_point_cloud(scene_name)
    
    # Convert point cloud to semantic map
    map_path = convert_point_cloud_to_semantic_map(pc_path)
    
    return map_path

def main():
    # Create required directories
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    os.makedirs(PC_SAVE_ROOT, exist_ok=True)
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)
    
    # Get valid scenes
    if 'hm3d' not in SPLIT_SCENES:
        print("ERROR: 'hm3d' not found in SPLIT_SCENES dictionary in poni/constants.py")
        print("Please add HM3D scenes to the SPLIT_SCENES dictionary first.")
        return
    
    # Get scenes from SPLIT_SCENES
    all_scenes = []
    if 'train' in SPLIT_SCENES['hm3d']:
        all_scenes.extend(SPLIT_SCENES['hm3d']['train'])
    if 'val' in SPLIT_SCENES['hm3d']:
        all_scenes.extend(SPLIT_SCENES['hm3d']['val'])
    
    print(f"Found {len(all_scenes)} scenes to process")
    
    # Process scenes sequentially - could be parallelized for speed
    for scene_name in tqdm.tqdm(all_scenes):
        process_scene(scene_name)
    
    print("===========> Done!")
    print(f"HM3D semantic maps have been created in {SEM_SAVE_ROOT}")

if __name__ == "__main__":
    main()