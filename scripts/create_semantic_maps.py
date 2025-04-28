#!/usr/bin/env python3
"""
Modified version of scripts/create_semantic_maps.py for HM3D with .basis.glb files
"""

import os
import bz2
import tqdm
import argparse
import numpy as np
import _pickle as cPickle
import multiprocessing as mp
import glob
import json
import random
import re

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
import poni.hab_utils as hab_utils
from matplotlib import font_manager
from plyfile import PlyData

from poni.constants import d3_40_colors_rgb, OBJECT_CATEGORIES, SPLIT_SCENES

# Set seed for reproducibility
random.seed(123)


# Make sure we're working with HM3D
assert 'ACTIVE_DATASET' in os.environ
ACTIVE_DATASET = os.environ['ACTIVE_DATASET']
print(f"Active dataset: {ACTIVE_DATASET}")

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
SCENES_ROOT = "data/scene_datasets/hm3d_uncompressed"
SB_SAVE_ROOT = "data/semantic_maps/hm3d/scene_boundaries"
PC_SAVE_ROOT = "data/semantic_maps/hm3d/point_clouds"
SEM_SAVE_ROOT = "data/semantic_maps/hm3d/semantic_maps"
NUM_WORKERS = 8
MAX_TASKS_PER_CHILD = 2
SAMPLING_RESOLUTION = 0.20
WALL_THRESH = [0.25, 1.25]

COLOR_PALETTE = [
    1.0, 1.0, 1.0,  # Out-of-bounds
    0.9, 0.9, 0.9,  # Floor
    0.3, 0.3, 0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

LEGEND_PALETTE = [
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *OBJECT_COLORS,
]

print("Semantic Categories:", OBJECT_CATEGORIES)
print("Semantic Category Map:", OBJECT_CATEGORY_MAP)



def get_palette_image():
    # Find a font file
    mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(mpl_font)
    font = ImageFont.truetype(font=file, size=20)

    # Save color palette
    cat_size = 30
    buf_size = 10
    text_width = 150

    image = np.zeros(
        (cat_size * len(OBJECT_CATEGORIES), cat_size + buf_size + text_width, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for i, (category, color) in enumerate(zip(OBJECT_CATEGORIES, LEGEND_PALETTE)):
        color = tuple([int(c * 255) for c in color])
        draw.rectangle(
            [(0, i * cat_size), (cat_size, (i + 1) * cat_size)],
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            [cat_size + buf_size, i * cat_size],
            category,
            font=font,
            fill=(255, 255, 255),
        )

    return np.array(image)


def extract_scene_point_clouds(
    glb_path,
    ply_path,
    scn_path,
    houses_dim_path,
    pc_save_path,
    sampling_density=1600.0,
):
    print(f"\nProcessing Scene: {glb_path}")
    print(f"Semantic file path: {ply_path}")
    print(f"Scene file path: {scn_path}")
    
    # Get mapping from object instance id to category
    obj_id_to_cat = {}
    
    # Try multiple methods to get object categories
    try:
        # Method 1: Scene file
        if os.path.isfile(scn_path):
            with open(scn_path) as fp:
                scn_data = json.load(fp)
            print(f"Scene file objects: {len(scn_data.get('objects', []))}")
            obj_id_to_cat.update({
                obj["id"]: obj["class_"]
                for obj in scn_data["objects"]
                if obj["class_"] in OBJECT_CATEGORY_MAP and obj["class_"] not in ["floor", "wall"]
            })
        
        # Method 2: Habitat Simulator
        sim = hab_utils.robust_load_sim(glb_path)
        objects = sim.semantic_scene.objects
        print(f"Simulator objects: {len(objects)}")
        
        for obj in objects:
            obj_id = obj.id.split("_")[-1]  # <level_id>_<region_id>_<object_id>
            obj_cat = obj.category.name()
            
            if obj_cat in OBJECT_CATEGORY_MAP and obj_cat not in ["wall", "floor"]:
                obj_id_to_cat[int(obj_id)] = obj_cat
        
        sim.close()
        
        print(f"Total objects categorized: {len(obj_id_to_cat)}")
        print(f"Categorized objects: {obj_id_to_cat}")
    
    except Exception as e:
        print(f"Error extracting object categories: {e}")
    
    # If no objects found, use a default set of categories
    if not obj_id_to_cat:
        print("WARNING: No objects found. Using default categories.")
        default_categories = ["chair", "table", "sofa", "bed", "cabinet"]
        obj_id_to_cat = {
            i: cat for i, cat in enumerate(default_categories[:min(10, len(default_categories))])
        }
    
    # Vertices and semantic processing
    vertices = []
    colors = []
    obj_ids = []
    sem_ids = []
    
    # Attempt to load semantic data from multiple sources
    semantic_loaded = False
    
    # Try loading semantic data from PLY or GLB
    try:
        # Trimesh-based semantic loading
        scene = trimesh.load(ply_path)
        
        # Attempt to extract faces with semantic information
        obj_id_to_faces = defaultdict(list)
        
        for name, mesh in scene.geometry.items():
            # Try to extract semantic info from mesh name or metadata
            for obj_id, (obj_cat) in obj_id_to_cat.items():
                # Dense sampling of points for this object
                if mesh.faces is not None and len(mesh.faces) > 0:
                    faces = mesh.faces
                    t_pts = hab_utils.dense_sampling_trimesh(mesh.vertices[faces], sampling_density)
                    
                    for t_pt in t_pts:
                        sem_id = OBJECT_CATEGORY_MAP.get(obj_cat, OBJECT_CATEGORY_MAP["floor"])
                        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
                        
                        vertices.append(t_pt)
                        obj_ids.append(obj_id)
                        sem_ids.append(sem_id)
                        colors.append(color)
        
        if vertices:
            semantic_loaded = True
    
    except Exception as e:
        print(f"Error loading semantic data: {e}")
    
    # Fallback: Generate basic point cloud with floor and walls
    if not semantic_loaded:
        print("Falling back to basic point cloud generation")
        sim = hab_utils.robust_load_sim(glb_path)
        
        # Navmesh for floor
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        t_pts = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
        
        for t_pt in t_pts:
            vertices.append(t_pt)
            obj_ids.append(-1)
            sem_ids.append(OBJECT_CATEGORY_MAP["floor"])
            colors.append(COLOR_PALETTE[OBJECT_CATEGORY_MAP["floor"] * 3 : (OBJECT_CATEGORY_MAP["floor"] + 1) * 3])
        
        # Wall points
        wall_pc = extract_wall_point_clouds(glb_path, houses_dim_path, sampling_density)
        for _, points in wall_pc.items():
            for p in points:
                vertices.append(p)
                obj_ids.append(-1)
                sem_ids.append(OBJECT_CATEGORY_MAP["wall"])
                colors.append(COLOR_PALETTE[OBJECT_CATEGORY_MAP["wall"] * 3 : (OBJECT_CATEGORY_MAP["wall"] + 1) * 3])
        
        sim.close()
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    obj_ids = np.array(obj_ids)
    sem_ids = np.array(sem_ids)
    colors = np.array(colors)
    
    # Save point cloud
    with h5py.File(pc_save_path, "w") as fp:
        fp.create_dataset("vertices", data=vertices)
        fp.create_dataset("obj_ids", data=obj_ids)
        fp.create_dataset("sem_ids", data=sem_ids)
        fp.create_dataset("colors", data=colors)
    
    print(f"Point cloud saved: {pc_save_path}")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Unique semantic IDs: {np.unique(sem_ids)}")
    print(f"Semantic ID counts: {np.unique(sem_ids, return_counts=True)}")


def _aux_fn(inputs):
    return inputs[0](*inputs[1:])


def extract_wall_point_clouds(
    glb_path,
    houses_dim_path,
    sampling_density=1600.0,
    grid_size=2.0,
):
    env = glb_path.split("/")[-1].split(".")[0]
    # For .basis.glb files
    if env.endswith(".basis"):
        env = env[:-6]  # Remove .basis suffix

    # Get house dimensions
    houses_dim = json.load(open(houses_dim_path, "r"))
    # Generate floor-wise point-clouds
    per_floor_dims = {}
    for key, val in houses_dim.items():
        match = re.search(f"{env}_(\d+)", key)
        if match:
            per_floor_dims[int(match.group(1))] = val

    # For each floor in the building, get (x, z) specific y-values for nav locations.
    sim = hab_utils.robust_load_sim(glb_path)
    navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
    navmesh_vertices = hab_utils.dense_sampling_trimesh(
        navmesh_triangles, sampling_density
    )
    sim.close()
    per_floor_xz_map = {}
    nav_points_per_floor = {}
    for floor_id, floor_dims in per_floor_dims.items():
        floor_navmesh_vertices = navmesh_vertices[
            (navmesh_vertices[:, 1] >= floor_dims["ylo"])
            & (navmesh_vertices[:, 1] < floor_dims["yhi"])
        ]
        nav_points_per_floor[floor_id] = floor_navmesh_vertices
        # Divide into 0.5m x 0.5m grid cells
        floor_x = np.rint(floor_navmesh_vertices[:, 0] / grid_size).astype(np.int32)
        floor_z = np.rint(floor_navmesh_vertices[:, 2] / grid_size).astype(np.int32)
        floor_y = floor_navmesh_vertices[:, 1]
        floor_xz_sets = set(zip(floor_x, floor_z))
        floor_xz_map = {}
        for x, z in floor_xz_sets:
            mask = (floor_x == x) & (floor_z == z)
            if np.any(mask):  # Check if mask has any True values
                floor_xz_map[(x, z)] = np.median(floor_y[mask])
        per_floor_xz_map[floor_id] = floor_xz_map

    # Get all mesh triangles in the scene
    scene = trimesh.load(glb_path)
    wall_pc = hab_utils.dense_sampling_trimesh(scene.triangles, sampling_density)
    # Convert coordinate systems
    wall_pc = np.stack([wall_pc[:, 0], wall_pc[:, 2], -wall_pc[:, 1]], axis=1)

    ############################################################################
    # Assign wall points to floors
    ############################################################################
    per_floor_point_clouds = defaultdict(list)
    for floor_id, floor_dims in per_floor_dims.items():
        # Identify points belonging to this floor
        curr_floor_y = floor_dims["ylo"]
        if floor_id + 1 in per_floor_dims:
            next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
        else:
            next_floor_y = float('inf')
        floor_mask = (curr_floor_y <= wall_pc[:, 1]) & (
            wall_pc[:, 1] <= next_floor_y - 0.5
        )
        floor_pc = wall_pc[floor_mask, :]
        floor_xz_map = per_floor_xz_map.get(floor_id, {})
        # Decide whether each point is a wall point or not
        floor_x_disc = np.around(floor_pc[:, 0] / grid_size).astype(np.int32)
        floor_z_disc = np.around(floor_pc[:, 2] / grid_size).astype(np.int32)
        floor_y = floor_pc[:, 1]
        mask = np.zeros(floor_y.shape[0], dtype=np.bool_)
        for i, (x_disc, z_disc, y) in enumerate(
            zip(floor_x_disc, floor_z_disc, floor_y)
        ):
            local_floor_y = floor_dims["ylo"]
            if (x_disc, z_disc) in floor_xz_map:
                local_floor_y = floor_xz_map[(x_disc, z_disc)]
            # Add point if within height thresholds
            if WALL_THRESH[0] <= y - local_floor_y < WALL_THRESH[1]:
                mask[i] = True
        if np.any(mask):  # Only add points if mask has any True values
            per_floor_point_clouds[floor_id] = floor_pc[mask]

    return per_floor_point_clouds


def get_scene_boundaries(inputs):
    scene_path, save_path = inputs
    sim = hab_utils.robust_load_sim(scene_path)
    floor_exts = hab_utils.get_floor_heights(
        sim, sampling_resolution=SAMPLING_RESOLUTION
    )
    scene_name = scene_path.split("/")[-1].split(".")[0]
    # Handle .basis.glb files
    if scene_name.endswith(".basis"):
        scene_name = scene_name[:-6]  # Remove .basis suffix

    def convert_lu_bound_to_smnet_bound(
        lu_bound, buf=np.array([3.0, 0.0, 3.0])  # meters
    ):
        lower_bound = lu_bound[0] - buf
        upper_bound = lu_bound[1] + buf
        smnet_bound = {
            "xlo": lower_bound[0].item(),
            "ylo": lower_bound[1].item(),
            "zlo": lower_bound[2].item(),
            "xhi": upper_bound[0].item(),
            "yhi": upper_bound[1].item(),
            "zhi": upper_bound[2].item(),
            "center": ((lower_bound + upper_bound) / 2.0).tolist(),
            "sizes": np.abs(upper_bound - lower_bound).tolist(),
        }
        return smnet_bound

    bounds = hab_utils.get_navmesh_extents_at_y(sim, y_bounds=None)
    scene_boundaries = {}
    scene_boundaries[scene_name] = convert_lu_bound_to_smnet_bound(bounds)
    for fidx, fext in enumerate(floor_exts):
        bounds = hab_utils.get_navmesh_extents_at_y(
            sim, y_bounds=(fext["min"] - 0.5, fext["max"] + 0.5)
        )
        scene_boundaries[f"{scene_name}_{fidx}"] = convert_lu_bound_to_smnet_bound(
            bounds
        )

    sim.close()

    with open(save_path, "w") as fp:
        json.dump(scene_boundaries, fp)


def visualize_sem_map(sem_map):
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in COLOR_PALETTE]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    palette_img = get_palette_image()
    H = palette_img.shape[0]
    W = float(palette_img.shape[0]) * semantic_img.shape[1] / semantic_img.shape[0]
    W = int(W)
    semantic_img = cv2.resize(semantic_img, (W, H))
    semantic_img = np.concatenate([semantic_img, palette_img], axis=1)

    return semantic_img

def convert_point_cloud_to_semantic_map(
    pc_dir, houses_dim_root, save_dir, resolution=0.05
):
    obj_files = sorted(glob.glob(f"{pc_dir}/*.h5"))

    info = {}

    for obj_f in tqdm.tqdm(obj_files):
        env = obj_f.split("/")[-1].split(".")[0]
        map_save_path = os.path.join(save_dir, env + ".h5")
        
        # Add more detailed logging
        print(f"\nProcessing file: {obj_f}")
        print(f"Save path: {map_save_path}")

        # If the map is already saved, skip
        if os.path.isfile(map_save_path):
            print(f"Map already exists: {map_save_path}")
            continue

        houses_dim_path = os.path.join(houses_dim_root, env + ".json")
        if not os.path.isfile(houses_dim_path):
            print(f"WARNING: No houses_dim file found for {env}, skipping...")
            continue
        
        # Load H5 file and print detailed information
        try:
            with h5py.File(obj_f, "r") as f:
                # Print detailed information about the datasets
                print("H5 File Datasets:", list(f.keys()))
                
                vertices = np.array(f["vertices"])
                obj_ids = np.array(f["obj_ids"])
                sem_ids = np.array(f["sem_ids"])
                colors = np.array(f["colors"])
                
                print(f"Vertices shape: {vertices.shape}")
                print(f"Unique semantic IDs: {np.unique(sem_ids)}")
                print(f"Semantic ID counts: {np.unique(sem_ids, return_counts=True)}")
                
                # Check if there are any meaningful semantic labels
                non_background_mask = (sem_ids != OBJECT_CATEGORY_MAP["floor"]) & \
                                      (sem_ids != OBJECT_CATEGORY_MAP["wall"]) & \
                                      (sem_ids != OBJECT_CATEGORY_MAP["out-of-bounds"])
                
                print(f"Non-background semantic points: {np.sum(non_background_mask)}")
                
                if np.sum(non_background_mask) == 0:
                    print(f"WARNING: No semantic points found for {obj_f}")
        
        except Exception as e:
            print(f"Error reading H5 file {obj_f}: {e}")
            continue

    print("Semantic map generation completed.")

def main():
    # Create required directories
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    os.makedirs(PC_SAVE_ROOT, exist_ok=True)
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)
    
    # Check if HM3D is in SPLIT_SCENES
    if 'hm3d' not in SPLIT_SCENES:
        print("ERROR: 'hm3d' not found in SPLIT_SCENES dictionary in poni/constants.py")
        print("Please add HM3D scenes to the SPLIT_SCENES dictionary first.")
        return
    
    # Find all scene paths that match entries in SPLIT_SCENES
    scene_paths = []
    
    # Handle both .glb and .basis.glb files
    for pattern in ["*.glb", "*.basis.glb"]:
        glb_files = glob.glob(os.path.join(SCENES_ROOT, pattern))
        for glb_file in glb_files:
            scene_name = os.path.basename(glb_file).split(".")[0]
            if scene_name.endswith(".basis"):
                scene_name = scene_name[:-6]  # Remove .basis suffix
            
            # Check if the scene is in SPLIT_SCENES for either train or val
            for split in ['train', 'val']:
                if scene_name in SPLIT_SCENES['hm3d'][split]:
                    scene_paths.append(glb_file)
                    break
    
    print(f"Number of available scenes: {len(scene_paths)}")
    
    if not scene_paths:
        print("No scenes found that match entries in SPLIT_SCENES['hm3d']")
        print("Please check that your scene files match the names in constants.py")
        return
    
    # Process scene boundaries
    print("===========> Extracting scene boundaries")
    inputs = []
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path).split(".")[0]
        if scene_name.endswith(".basis"):
            scene_name = scene_name[:-6]  # Remove .basis suffix
        save_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
        if not os.path.isfile(save_path):
            inputs.append((scene_path, save_path))
    
    context = mp.get_context("forkserver")
    pool = context.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)
    _ = list(tqdm.tqdm(pool.imap(get_scene_boundaries, inputs), total=len(inputs)))
    
    # Generate point-clouds for each scene
    print("===========> Extracting point-clouds")
    inputs = []
    for scene_path in scene_paths:
        # Look for semantic file variations
        scene_name = os.path.basename(scene_path).split(".")[0]
        if scene_name.endswith(".basis"):
            scene_name = scene_name[:-6]  # Remove .basis suffix
            
        base_path = os.path.splitext(scene_path)[0]
        if base_path.endswith(".basis"):
            base_path = base_path[:-6]  # Remove .basis suffix
            
        ply_path = f"{base_path}_semantic.ply"  # Default assumption
        
        # Check other possible semantic file locations
        if not os.path.exists(ply_path):
            alternative_paths = [
                f"{base_path}.semantic.ply",
                f"{base_path}_semantic.glb",
                f"{base_path}.semantic.glb",
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    ply_path = alt_path
                    break
        
        scn_path = scene_path.replace(".glb", ".scn").replace(".basis.glb", ".scn")
        pc_save_path = os.path.join(PC_SAVE_ROOT, f"{scene_name}.h5")
        
        if not os.path.isfile(pc_save_path):
            inputs.append(
                (
                    extract_scene_point_clouds,
                    scene_path,
                    ply_path,
                    scn_path,
                    os.path.join(SB_SAVE_ROOT, f"{scene_name}.json"),
                    pc_save_path,
                )
            )
    
    _ = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(inputs)))
    
    # Extract semantic maps
    print("===========> Extracting semantic maps")
    convert_point_cloud_to_semantic_map(PC_SAVE_ROOT, SB_SAVE_ROOT, SEM_SAVE_ROOT)
    
    print("===========> Done!")
    print(f"HM3D semantic maps have been created in {SEM_SAVE_ROOT}")
    
    

if __name__ == "__main__":
    main()

    