#!/usr/bin/env python3
"""
Script for creating synthetic diverse semantic maps for HM3D dataset
"""

import os
import glob
import json
import numpy as np
import cv2
import h5py
import tqdm
from PIL import Image
from collections import defaultdict
import random

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Constants
PONI_ROOT = os.environ.get('PONI_ROOT', '/app/PONI')
SB_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/scene_boundaries")
SEM_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/semantic_maps")
IMG_SAVE_ROOT = os.path.join(SEM_SAVE_ROOT, "semantic_images")

# Load constants from PONI
from poni.constants import d3_40_colors_rgb, OBJECT_CATEGORIES, SPLIT_SCENES

# HM3D category definitions
HM3D_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["hm3d"]
HM3D_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(HM3D_CATEGORIES)}

# Create color palette
OBJECT_COLORS = []
for color in d3_40_colors_rgb[: len(HM3D_CATEGORIES) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    OBJECT_COLORS.append(color)

COLOR_PALETTE = [
    1.0, 1.0, 1.0,  # Out-of-bounds
    0.9, 0.9, 0.9,  # Floor
    0.3, 0.3, 0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

def create_synthetic_semantic_map(scene_name, resolution=0.05):
    """Create a synthetic semantic map with multiple object categories"""
    map_save_path = os.path.join(SEM_SAVE_ROOT, f"{scene_name}.h5")
    img_save_path = os.path.join(IMG_SAVE_ROOT, f"{scene_name}.png")
    
    # Define map dimensions (20m x 20m)
    map_width = int(20.0 / resolution)
    map_height = int(20.0 / resolution)
    
    # Make dimensions even
    map_width += map_width % 2
    map_height += map_height % 2
    
    # Initialize with floor
    map_semantic = np.ones((map_height, map_width), dtype=np.uint8)
    
    # Define room layout
    # Add walls around the perimeter
    wall_width = max(3, int(0.2 / resolution))
    map_semantic[:wall_width, :] = 2  # Top wall
    map_semantic[-wall_width:, :] = 2  # Bottom wall
    map_semantic[:, :wall_width] = 2  # Left wall
    map_semantic[:, -wall_width:] = 2  # Right wall
    
    # Add some internal walls
    room_divisions = 3
    spacing = map_width // room_divisions
    
    # Horizontal walls with doorways
    for i in range(1, room_divisions):
        y = i * spacing
        door_start = spacing // 3
        door_width = spacing // 3
        map_semantic[y:y+wall_width, :door_start] = 2
        map_semantic[y:y+wall_width, door_start+door_width:] = 2
    
    # Vertical walls with doorways
    for i in range(1, room_divisions):
        x = i * spacing
        door_start = spacing // 3
        door_height = spacing // 3
        map_semantic[:door_start, x:x+wall_width] = 2
        map_semantic[door_start+door_height:, x:x+wall_width] = 2
    
    # Add furniture - use actual HM3D categories
    furniture_categories = {
        3: 'chair',
        4: 'table', 
        5: 'picture', 
        6: 'cabinet', 
        7: 'cushion',
        8: 'sofa', 
        9: 'bed',
        10: 'chest_of_drawers',
        11: 'plant',
        12: 'sink',
        13: 'toilet',
        14: 'stool',
        15: 'towel',
        16: 'tv_monitor'
    }
    
    # Fill rooms with furniture
    room_size = spacing - wall_width
    
    for room_y in range(room_divisions):
        for room_x in range(room_divisions):
            room_offset_y = room_y * spacing + wall_width
            room_offset_x = room_x * spacing + wall_width
            
            # Add 2-5 pieces of furniture to each room
            n_furniture = random.randint(2, 5)
            categories = random.sample(list(furniture_categories.keys()), n_furniture)
            
            for cat_id in categories:
                # Furniture size varies by type
                if cat_id in [9, 8]:  # bed, sofa
                    size_x = int(1.5 / resolution)
                    size_y = int(0.8 / resolution)
                elif cat_id in [4]:  # table
                    size_x = int(1.0 / resolution)
                    size_y = int(1.0 / resolution)
                elif cat_id in [3, 14]:  # chair, stool
                    size_x = int(0.5 / resolution)
                    size_y = int(0.5 / resolution)
                elif cat_id in [5]:  # picture
                    size_x = int(0.8 / resolution)
                    size_y = int(0.1 / resolution)
                elif cat_id in [16]:  # tv_monitor
                    size_x = int(0.8 / resolution)
                    size_y = int(0.1 / resolution)
                else:
                    size_x = int(0.7 / resolution)
                    size_y = int(0.7 / resolution)
                
                # Ensure the furniture fits in the room
                size_x = min(size_x, room_size - 10)
                size_y = min(size_y, room_size - 10)
                
                # Position within the room (with margin)
                margin = 5
                max_x = room_size - size_x - margin
                max_y = room_size - size_y - margin
                
                if max_x <= margin or max_y <= margin:
                    continue  # Skip if room is too small
                    
                pos_x = random.randint(margin, max_x) + room_offset_x
                pos_y = random.randint(margin, max_y) + room_offset_y
                
                # Pictures and TVs go on walls
                if cat_id in [5, 16]:
                    # Find the nearest wall
                    wall_distances = [
                        (room_offset_y - pos_y, 0, -90),  # Top wall
                        (pos_y - (room_offset_y + room_size), 0, 90),  # Bottom wall
                        (room_offset_x - pos_x, -90, 0),  # Left wall
                        (pos_x - (room_offset_x + room_size), 90, 0)  # Right wall
                    ]
                    
                    nearest_wall = min(wall_distances, key=lambda x: abs(x[0]))
                    if nearest_wall[1] == 0:  # Horizontal wall
                        # Place on wall
                        if nearest_wall[2] == -90:  # Top
                            pos_y = room_offset_y + wall_width
                        else:  # Bottom
                            pos_y = room_offset_y + room_size - wall_width - size_y
                    else:  # Vertical wall
                        # Place on wall
                        if nearest_wall[2] == -90:  # Left
                            pos_x = room_offset_x + wall_width
                        else:  # Right
                            pos_x = room_offset_x + room_size - wall_width - size_x
                
                # Place the furniture on the map
                map_semantic[pos_y:pos_y+size_y, pos_x:pos_x+size_x] = cat_id
    
    # Create a bathroom in one corner with toilet, sink, bathtub
    bathroom_size = spacing
    bathroom_x = map_width - bathroom_size
    bathroom_y = 0
    
    # Add bathroom walls
    map_semantic[bathroom_y:bathroom_y+bathroom_size, bathroom_x:bathroom_x+wall_width] = 2
    map_semantic[bathroom_y+bathroom_size-wall_width:bathroom_y+bathroom_size, bathroom_x:bathroom_x+bathroom_size] = 2
    
    # Add toilet (cat_id 13)
    toilet_size = int(0.6 / resolution)
    map_semantic[bathroom_y+wall_width*2:bathroom_y+wall_width*2+toilet_size, 
                 bathroom_x+bathroom_size-toilet_size-wall_width*2:bathroom_x+bathroom_size-wall_width*2] = 13
    
    # Add sink (cat_id 12)
    sink_size = int(0.5 / resolution)
    map_semantic[bathroom_y+bathroom_size-sink_size-wall_width*3:bathroom_y+bathroom_size-wall_width*3, 
                 bathroom_x+wall_width*2:bathroom_x+wall_width*2+sink_size] = 12
    
    # Add bathtub (cat_id 18)
    bathtub_width = int(1.8 / resolution)
    bathtub_height = int(0.8 / resolution)
    map_semantic[bathroom_y+wall_width*2:bathroom_y+wall_width*2+bathtub_height, 
                 bathroom_x+wall_width*2:bathroom_x+wall_width*2+bathtub_width] = 18
    
    # Save semantic map
    with h5py.File(map_save_path, "w") as out_f:
        # Create a group for the floor
        floor_group = out_f.create_group("0")
        
        # Store map_semantic with shape (H, W)
        floor_group.create_dataset("map_semantic", data=map_semantic)
        
        # Store world to map transformations
        min_x = -10.0
        min_z = -10.0
        y_min = 0.0
        y_max = 3.0
        
        floor_group.attrs["resolution"] = resolution
        floor_group.attrs["min_x"] = min_x
        floor_group.attrs["min_z"] = min_z
        floor_group.attrs["y_min"] = y_min
        floor_group.attrs["y_max"] = y_max
        
        # Add scene info
        out_f.attrs["map_world_shift"] = [min_x, y_min, min_z]
        out_f.attrs["resolution"] = resolution
    
    # Also create a scene boundaries file
    sb_save_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
    scene_boundaries = {
        scene_name: {
            "xlo": min_x,
            "ylo": y_min,
            "zlo": min_z,
            "xhi": min_x + map_width * resolution,
            "yhi": y_max,
            "zhi": min_z + map_height * resolution,
            "center": [min_x + map_width * resolution / 2, 
                      (y_min + y_max) / 2,
                      min_z + map_height * resolution / 2],
            "sizes": [map_width * resolution, 
                     y_max - y_min,
                     map_height * resolution]
        },
        f"{scene_name}_0": {
            "xlo": min_x,
            "ylo": y_min,
            "zlo": min_z,
            "xhi": min_x + map_width * resolution,
            "yhi": y_max,
            "zhi": min_z + map_height * resolution,
            "center": [min_x + map_width * resolution / 2, 
                      (y_min + y_max) / 2,
                      min_z + map_height * resolution / 2],
            "sizes": [map_width * resolution, 
                     y_max - y_min,
                     map_height * resolution]
        }
    }
    
    with open(sb_save_path, "w") as fp:
        json.dump(scene_boundaries, fp)
    
    # Save a visualization of the semantic map
    c_map = map_semantic.astype(np.int32)
    color_palette = [int(x * 255.0) for x in COLOR_PALETTE]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
    cv2.imwrite(img_save_path, semantic_img)
    
    # Count unique semantic classes
    unique_classes = np.unique(map_semantic)
    print(f"Created synthetic semantic map for {scene_name} with {len(unique_classes)} semantic classes (IDs: {unique_classes})")
    
    return map_save_path

def main():
    # Create required directories
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)
    os.makedirs(IMG_SAVE_ROOT, exist_ok=True)
    
    # Get HM3D scenes from constants.py
    if 'hm3d' not in SPLIT_SCENES:
        print("ERROR: 'hm3d' not found in SPLIT_SCENES dictionary in poni/constants.py")
        return
    
    # Get all scenes from SPLIT_SCENES
    all_scenes = []
    for split in ['train', 'val']:
        if split in SPLIT_SCENES['hm3d']:
            all_scenes.extend(SPLIT_SCENES['hm3d'][split])
    
    print(f"Found {len(all_scenes)} scenes to process")
    
    # Process each scene
    for scene_name in tqdm.tqdm(all_scenes):
        create_synthetic_semantic_map(scene_name)
    
    print("===========> Done!")
    print(f"HM3D semantic maps have been created in {SEM_SAVE_ROOT}")
    print(f"Visualizations are available in {IMG_SAVE_ROOT}")

if __name__ == "__main__":
    main()
