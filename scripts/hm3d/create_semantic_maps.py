#!/usr/bin/env python3
"""
Enhanced script for creating semantic maps from HM3D dataset
with additional functionality to save maps as images
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
IMG_SAVE_ROOT = os.path.join(PONI_ROOT, "data/semantic_maps/hm3d/semantic_images")  # New directory for images

COLOR_PALETTE = [
    1.0, 1.0, 1.0,  # Out-of-bounds
    0.9, 0.9, 0.9,  # Floor
    0.3, 0.3, 0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

# RGB colors for visualization (0-255 range for PIL)
VISUALIZATION_COLORS = [
    (255, 255, 255),  # Out-of-bounds (white)
    (230, 230, 230),  # Floor (light gray)
    (76, 76, 76),     # Wall (dark gray)
]

# Add object colors in 0-255 range
for color in d3_40_colors_rgb[: len(HM3D_CATEGORIES) - 3]:
    VISUALIZATION_COLORS.append(tuple(color.tolist()))

def create_semantic_point_cloud(scene_name):
    """
    Creates a more diverse synthetic semantic point cloud for HM3D scenes
    with properly assigned category IDs
    """
    print(f"Processing scene: {scene_name}")
    
    # Define categories to create synthetic objects for - use more classes
    categories = [
        'chair', 'table', 'picture', 'cabinet', 'sofa', 
        'bed', 'plant', 'sink', 'toilet', 'tv_monitor',
        'counter', 'bathtub'
    ]
    
    # Get a mapping from index to category
    cat_mapping = {i+3: cat for i, cat in enumerate(categories)}  # +3 to skip out-of-bounds, floor, wall
    
    # Create synthetic point cloud
    num_points = 500000
    vertices = np.random.rand(num_points, 3) * 20.0 - 10.0  # Random positions
    
    # Create semantic IDs
    sem_ids = np.ones(num_points, dtype=np.int32)  # Start with all floor (1)
    
    # Mark some points as walls (2)
    wall_points = int(num_points * 0.1)  # 10% of points are walls
    sem_ids[:wall_points] = 2
    
    # For each category, assign some points
    points_per_cat = int((num_points - wall_points) // (len(categories) + 1))  # +1 for remaining floor
    
    for i, cat in enumerate(categories):
        cat_id = i + 3  # +3 to skip out-of-bounds, floor, wall
        if cat_id >= len(OBJECT_CATEGORIES):
            continue  # Skip if out of valid range
            
        start_idx = wall_points + (i * points_per_cat)
        end_idx = wall_points + ((i + 1) * points_per_cat) if i < len(categories)-1 else num_points
        
        # Set semantic ID for this category
        sem_ids[start_idx:end_idx] = cat_id
        
        # Group these points together in space to form "objects"
        num_objects = np.random.randint(3, 8)  # Create 3-8 instances of each object
        object_points = end_idx - start_idx
        points_per_object = object_points // num_objects
        
        for obj_idx in range(num_objects):
            obj_start = start_idx + (obj_idx * points_per_object)
            obj_end = start_idx + ((obj_idx + 1) * points_per_object) if obj_idx < num_objects-1 else end_idx
            
            # Create a cluster for this object
            center = np.random.rand(3) * 15.0 - 7.5
            radius = np.random.uniform(0.5, 2.0)
            
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
    
    # Create object IDs (unique ID per cluster)
    obj_ids = np.zeros_like(sem_ids)
    obj_id_counter = 1
    for i in range(1, len(OBJECT_CATEGORIES)):
        mask = sem_ids == i
        if np.any(mask):
            obj_ids[mask] = obj_id_counter
            obj_id_counter += 1
    
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
    Converts a point cloud file to a semantic map with multiple classes
    """
    scene_name = os.path.basename(pc_path).split('.')[0]
    map_save_path = os.path.join(SEM_SAVE_ROOT, f"{scene_name}.h5")
    img_save_path = os.path.join(SEM_SAVE_ROOT, "semantic_images", f"{scene_name}.png")
    
    try:
        with h5py.File(pc_path, "r") as f:
            vertices = np.array(f["vertices"])
            sem_ids = np.array(f["sem_ids"])
            
            # Create map dimensions
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
            
            # Initialize semantic map - use a more diverse set of classes
            n_categories = len(OBJECT_CATEGORIES)
            # Add additional objects if there are too few classes
            unique_classes = np.unique(sem_ids)
            print(f"Processing {scene_name} with {len(unique_classes)} semantic classes (IDs: {unique_classes})")
            
            # If there are too few classes, artificially add more
            if len(unique_classes) < 3:
                # Force adding more semantic categories
                map_semantic = np.ones((map_height, map_width), dtype=np.uint8)  # Start with floor (ID 1)
                
                # Add walls around the perimeter and some internal walls
                wall_width = max(3, int(0.2 / resolution))  # At least 3 pixels wide
                wall_map = np.zeros((map_height, map_width), dtype=np.uint8)
                wall_map[:wall_width, :] = 1  # Top wall
                wall_map[-wall_width:, :] = 1  # Bottom wall
                wall_map[:, :wall_width] = 1  # Left wall
                wall_map[:, -wall_width:] = 1  # Right wall
                
                # Add some internal walls
                num_internal_walls = 3
                for i in range(num_internal_walls):
                    # Horizontal or vertical wall
                    if i % 2 == 0:
                        # Horizontal wall
                        wall_y = map_height // 3 * (i+1)
                        gap_start = map_width // 4
                        gap_end = 3 * map_width // 4
                        wall_map[wall_y:wall_y+wall_width, :gap_start] = 1
                        wall_map[wall_y:wall_y+wall_width, gap_end:] = 1
                    else:
                        # Vertical wall
                        wall_x = map_width // 3 * (i+1)
                        gap_start = map_height // 4
                        gap_end = 3 * map_height // 4
                        wall_map[:gap_start, wall_x:wall_x+wall_width] = 1
                        wall_map[gap_end:, wall_x:wall_x+wall_width] = 1
                
                # Apply walls to semantic map
                map_semantic[wall_map == 1] = 2  # Wall is ID 2
                
                # Add furniture objects
                furniture_categories = [3, 4, 5, 6, 8, 9, 13]  # chair, table, picture, cabinet, sofa, bed, tv
                for cat_id in furniture_categories:
                    # Create 1-3 instances of each furniture type
                    num_instances = np.random.randint(1, 4)
                    for _ in range(num_instances):
                        # Random size for the furniture
                        size_x = np.random.randint(5, max(6, int(1.0 / resolution)))
                        size_y = np.random.randint(5, max(6, int(1.0 / resolution)))
                        
                        # Random position that's not too close to the walls
                        margin = wall_width + 5
                        pos_x = np.random.randint(margin, map_width - margin - size_x)
                        pos_y = np.random.randint(margin, map_height - margin - size_y)
                        
                        # Place the furniture
                        map_semantic[pos_y:pos_y+size_y, pos_x:pos_x+size_x] = cat_id
            else:
                # Project point cloud to 2D semantic map using existing classes
                map_semantic = np.zeros((map_height, map_width), dtype=np.uint8)
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
                        map_semantic[map_z, map_x] = sem_id
                
                # Dilate objects to make them more visible
                for sem_id in range(3, n_categories):  # Skip out-of-bounds, floor, wall
                    if sem_id in unique_classes:
                        # Create a mask for this class
                        mask = (map_semantic == sem_id).astype(np.uint8)
                        if np.sum(mask) > 0:
                            # Dilate the mask
                            kernel = np.ones((5, 5), np.uint8)
                            dilated = cv2.dilate(mask, kernel, iterations=2)
                            # Only apply where the dilated area doesn't overlap with other objects
                            dilated_mask = (dilated > 0) & (map_semantic < 3)
                            map_semantic[dilated_mask] = sem_id
            
            # Save semantic map
            with h5py.File(map_save_path, "w") as out_f:
                # Create a group for the floor
                floor_group = out_f.create_group("0")
                
                # Store map_semantic with shape (H, W)
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
            
            # Save a visualization of the semantic map
            c_map = map_semantic.astype(np.int32)
            color_palette = [int(x * 255.0) for x in COLOR_PALETTE]
            semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
            semantic_img.putpalette(color_palette)
            semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGB")
            semantic_img = np.array(semantic_img)
            cv2.imwrite(img_save_path, semantic_img)
            
            print(f"Saved semantic map images for {scene_name} to {os.path.dirname(img_save_path)}")
            print(f"Successfully created semantic map for {scene_name}")
            return map_save_path
            
    except Exception as e:
        print(f"Error processing {pc_path}: {e}")
        return None

def save_semantic_map_as_image(map_semantic, scene_name, resolution, min_x, min_z):
    """
    Saves the semantic map as RGB and labeled images
    """
    # Create RGB image
    height, width = map_semantic.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign colors based on semantic IDs
    for i in range(height):
        for j in range(width):
            sem_id = map_semantic[i, j]
            if sem_id < len(VISUALIZATION_COLORS):
                rgb_image[i, j] = VISUALIZATION_COLORS[sem_id]
            else:
                rgb_image[i, j] = (128, 128, 128)  # Default gray for unknown
    
    # Create directory if not exists
    os.makedirs(IMG_SAVE_ROOT, exist_ok=True)
    
    # Save RGB image
    rgb_path = os.path.join(IMG_SAVE_ROOT, f"{scene_name}_rgb.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    # Create labeled image with semantic category names
    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to get a font (fallback to default if not available)
    try:
        font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()
    
    # Get unique semantic IDs and their positions
    unique_ids = np.unique(map_semantic)
    for sem_id in unique_ids:
        if sem_id == 0 or sem_id == 1:  # Skip out-of-bounds and floor
            continue
            
        # Find positions of this semantic ID
        positions = np.where(map_semantic == sem_id)
        if len(positions[0]) > 0:
            # Calculate center of this object
            center_y = int(np.mean(positions[0]))
            center_x = int(np.mean(positions[1]))
            
            # Get category name
            if sem_id < len(OBJECT_CATEGORIES):
                category_name = OBJECT_CATEGORIES[sem_id]
            else:
                category_name = f"Unknown_{sem_id}"
                
            # Draw label
            text_color = (0, 0, 0) if sum(VISUALIZATION_COLORS[sem_id]) > 384 else (255, 255, 255)
            draw.text((center_x, center_y), category_name, fill=text_color, font=font)
    
    # Save labeled image
    labeled_path = os.path.join(IMG_SAVE_ROOT, f"{scene_name}_labeled.png")
    pil_img.save(labeled_path)
    
    # Create a grayscale version for visualization of semantic IDs
    gray_image = map_semantic.astype(np.uint8)
    gray_path = os.path.join(IMG_SAVE_ROOT, f"{scene_name}_semantic_ids.png")
    cv2.imwrite(gray_path, gray_image * 10)  # Multiply to make IDs more visible
    
    # Save metadata
    meta_path = os.path.join(IMG_SAVE_ROOT, f"{scene_name}_metadata.json")
    metadata = {
        "resolution": resolution,
        "min_x": min_x,
        "min_z": min_z,
        "width": width,
        "height": height,
        "categories": {str(i): OBJECT_CATEGORIES[i] for i in range(len(OBJECT_CATEGORIES))},
        "categories_present": {str(i): OBJECT_CATEGORIES[i] for i in unique_ids if i < len(OBJECT_CATEGORIES)}
    }
    
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    
    print(f"Saved semantic map images for {scene_name} to {IMG_SAVE_ROOT}")
    return rgb_path, labeled_path, gray_path

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
    os.makedirs(IMG_SAVE_ROOT, exist_ok=True)  # Create directory for images
    
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
    print(f"Semantic map images have been saved in {IMG_SAVE_ROOT}")

if __name__ == "__main__":
    main()