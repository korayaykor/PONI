import glob
import json
import math
import multiprocessing as mp
import os
import random
import re
import collections

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

random.seed(123)

################################################################################
# Gibson constants
################################################################################
GIBSON_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["gibson"]
GIBSON_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(GIBSON_CATEGORIES)}
GIBSON_OBJECT_COLORS = [
    (0.9400000000000001, 0.7818, 0.66),
    (0.9400000000000001, 0.8868, 0.66),
    (0.8882000000000001, 0.9400000000000001, 0.66),
    (0.7832000000000001, 0.9400000000000001, 0.66),
    (0.6782000000000001, 0.9400000000000001, 0.66),
    (0.66, 0.9400000000000001, 0.7468000000000001),
    (0.66, 0.9400000000000001, 0.8518000000000001),
    (0.66, 0.9232, 0.9400000000000001),
    (0.66, 0.8182, 0.9400000000000001),
    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),
]
################################################################################
# MP3D constants
################################################################################
MP3D_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["mp3d"]
MP3D_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(MP3D_CATEGORIES)}
MP3D_OBJECT_COLORS = []  # Excludes 'out-of-bounds', 'floor', and 'wall'
for color in d3_40_colors_rgb[: len(MP3D_CATEGORIES) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    MP3D_OBJECT_COLORS.append(color)
    
################################################################################
# HM3D constants
################################################################################
HM3D_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["hm3d"]
HM3D_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(HM3D_CATEGORIES)}
HM3D_OBJECT_COLORS = []  # Excludes 'out-of-bounds', 'floor', and 'wall'
for color in d3_40_colors_rgb[: len(HM3D_CATEGORIES) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    HM3D_OBJECT_COLORS.append(color)

################################################################################
# General constants
################################################################################
assert "ACTIVE_DATASET" in os.environ
ACTIVE_DATASET = os.environ["ACTIVE_DATASET"]  # mp3d / gibson / hm3d
if ACTIVE_DATASET == "mp3d":
    OBJECT_COLORS = MP3D_OBJECT_COLORS
    OBJECT_CATEGORIES = MP3D_CATEGORIES
    OBJECT_CATEGORY_MAP = MP3D_CATEGORY_MAP
    SCENES_ROOT = "data/scene_datasets/mp3d_uncompressed"
    SB_SAVE_ROOT = "data/semantic_maps/mp3d/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/mp3d/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/mp3d/semantic_maps"
    NUM_WORKERS = 8
    MAX_TASKS_PER_CHILD = 2
    SAMPLING_RESOLUTION = 0.20
    WALL_THRESH = [0.25, 1.25]
elif ACTIVE_DATASET == "hm3d":
    OBJECT_COLORS = HM3D_OBJECT_COLORS
    OBJECT_CATEGORIES = HM3D_CATEGORIES
    OBJECT_CATEGORY_MAP = HM3D_CATEGORY_MAP
    SCENES_ROOT = "data/scene_datasets/hm3d_uncompressed"
    SB_SAVE_ROOT = "data/semantic_maps/hm3d/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/hm3d/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/hm3d/semantic_maps"
    NUM_WORKERS = 8
    MAX_TASKS_PER_CHILD = 2
    SAMPLING_RESOLUTION = 0.10
    WALL_THRESH = [0.25, 1.25]
else:
    OBJECT_COLORS = GIBSON_OBJECT_COLORS
    OBJECT_CATEGORIES = GIBSON_CATEGORIES
    OBJECT_CATEGORY_MAP = GIBSON_CATEGORY_MAP
    SCENES_ROOT = "data/scene_datasets/gibson_semantic"
    SB_SAVE_ROOT = "data/semantic_maps/gibson/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/gibson/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps"
    NUM_WORKERS = 12
    MAX_TASKS_PER_CHILD = None
    SAMPLING_RESOLUTION = 0.10
    WALL_THRESH = [0.25, 1.25]

COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]
LEGEND_PALETTE = [
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *OBJECT_COLORS,
]


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
    print(f"Scene: {glb_path}")
    # Get mapping from object instance id to category
    if os.path.isfile(scn_path):
        with open(scn_path) as fp:
            scn_data = json.load(fp)
        obj_id_to_cat = {
            obj["id"]: obj["class_"]
            for obj in scn_data["objects"]
            if obj["class_"] in OBJECT_CATEGORY_MAP
        }
    else:
        sim = hab_utils.robust_load_sim(glb_path)
        objects = sim.semantic_scene.objects
        obj_id_to_cat = {}
        for obj in objects:
            obj_id = obj.id.split("_")[-1]  # <level_id>_<region_id>_<object_id>
            obj_cat = obj.category.name()
            if obj_cat not in OBJECT_CATEGORY_MAP or obj_cat in ["wall", "floor"]:
                continue
            obj_id_to_cat[int(obj_id)] = obj_cat
        sim.close()
    ############################################################################
    # Get vertices for all objects
    ############################################################################
    vertices = []
    colors = []
    obj_ids = []
    sem_ids = []
    ply_data = PlyData.read(ply_path)
    # Get faces for each object id
    obj_id_to_faces = defaultdict(list)
    for face in ply_data["face"]:
        vids = list(face[0])
        obj_id = face[1]
        if obj_id in obj_id_to_cat:
            p1 = ply_data["vertex"][vids[0]]
            p1 = [p1[0], p1[2], -p1[1]]
            p2 = ply_data["vertex"][vids[1]]
            p2 = [p2[0], p2[2], -p2[1]]
            p3 = ply_data["vertex"][vids[2]]
            p3 = [p3[0], p3[2], -p3[1]]
            obj_id_to_faces[obj_id].append([p1, p2, p3])
    # Get dense point-clouds for each object id
    for obj_id, faces in obj_id_to_faces.items():
        ocat = obj_id_to_cat[obj_id]
        sem_id = OBJECT_CATEGORY_MAP[ocat]
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        
        # Sample more points for each object to ensure better representation
        sampling_density_local = sampling_density * 2  # Increase sampling density for objects
        t_pts = hab_utils.dense_sampling_trimesh(np.array(faces), sampling_density_local)
        
        for t_pt in t_pts:
            vertices.append(t_pt)
            obj_ids.append(obj_id)
            sem_ids.append(sem_id)
            colors.append(color)

    ############################################################################
    # Get vertices for navigable spaces
    ############################################################################
    sim = hab_utils.robust_load_sim(glb_path)
    navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
    t_pts = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
    for t_pt in t_pts:
        obj_id = -1
        sem_id = OBJECT_CATEGORY_MAP["floor"]
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        vertices.append(t_pt)
        obj_ids.append(obj_id)
        sem_ids.append(sem_id)
        colors.append(color)
    sim.close()

    ############################################################################
    # Get vertices for walls
    ############################################################################
    per_floor_wall_pc = extract_wall_point_clouds(
        glb_path, houses_dim_path, sampling_density=sampling_density
    )
    for _, points in per_floor_wall_pc.items():
        obj_id = -1
        sem_id = OBJECT_CATEGORY_MAP["wall"]
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        for p in points:
            vertices.append(p)
            obj_ids.append(obj_id)
            sem_ids.append(sem_id)
            colors.append(color)

    vertices = np.array(vertices)
    obj_ids = np.array(obj_ids)
    sem_ids = np.array(sem_ids)
    colors = np.array(colors)

    with h5py.File(pc_save_path, "w") as fp:
        fp.create_dataset("vertices", data=vertices)
        fp.create_dataset("obj_ids", data=obj_ids)
        fp.create_dataset("sem_ids", data=sem_ids)
        fp.create_dataset("colors", data=colors)


def _aux_fn(inputs):
    return inputs[0](*inputs[1:])


def extract_wall_point_clouds(
    glb_path,
    houses_dim_path,
    sampling_density=1600.0,
    grid_size=2.0,
):
    env = glb_path.split("/")[-1].split(".")[0]

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
            next_floor_y = math.inf
        floor_mask = (curr_floor_y <= wall_pc[:, 1]) & (
            wall_pc[:, 1] <= next_floor_y - 0.5
        )
        floor_pc = wall_pc[floor_mask, :]
        floor_xz_map = per_floor_xz_map[floor_id]
        # Decide whether each point is a wall point or not
        floor_x_disc = np.around(floor_pc[:, 0] / grid_size).astype(np.int32)
        floor_z_disc = np.around(floor_pc[:, 2] / grid_size).astype(np.int32)
        floor_y = floor_pc[:, 1]
        mask = np.zeros(floor_y.shape[0], dtype=bool)
        for i, (x_disc, z_disc, y) in enumerate(
            zip(floor_x_disc, floor_z_disc, floor_y)
        ):
            floor_y = per_floor_dims[floor_id]["ylo"]
            if (x_disc, z_disc) in floor_xz_map:
                floor_y = floor_xz_map[(x_disc, z_disc)]
            # Add point if within height thresholds
            if WALL_THRESH[0] <= y - floor_y < WALL_THRESH[1]:
                mask[i] = True
        per_floor_point_clouds[floor_id] = floor_pc[mask]

    return per_floor_point_clouds

def extract_hm3d_scene_point_clouds(
    glb_path,
    semantic_glb_path,
    semantic_txt_path,
    houses_dim_path,
    pc_save_path,
    sampling_density=1600.0,
):
    print(f"\n===== PROCESSING SCENE: {glb_path} =====")
    print(f"Semantic GLB: {semantic_glb_path}")
    print(f"Semantic TXT: {semantic_txt_path}")
    
    # Get mapping from object instance id to category
    obj_id_to_cat = {}
    category_id_mapping = {}
    
    # Parse the semantic.txt file to get mapping from instance IDs to categories
    if os.path.isfile(semantic_txt_path):
        try:
            with open(semantic_txt_path, 'r') as f:
                lines = f.readlines()
                print(f"Found {len(lines)} entries in semantic.txt")
                
                # Try to detect format - is it comma-separated or space-separated?
                sample_line = lines[0].strip() if lines else ""
                if ',' in sample_line:
                    print("Detected comma-separated format")
                    # Format appears to be: ID,HexColor,"Name",FloorNum
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            try:
                                obj_id = int(parts[0])
                                obj_cat = parts[2].strip('"').lower()
                                
                                # Map to our category system
                                for known_cat in OBJECT_CATEGORY_MAP.keys():
                                    if known_cat != "out-of-bounds":
                                        if known_cat == obj_cat or known_cat in obj_cat or obj_cat in known_cat:
                                            obj_id_to_cat[obj_id] = known_cat
                                            category_id_mapping[obj_id] = OBJECT_CATEGORY_MAP[known_cat]
                                            break
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing line: {line.strip()} - {e}")
                else:
                    print("Assuming space-separated format")
                    # Format might be: ID Category
                    for line in lines:
                        parts = line.strip().split(None, 1)
                        if len(parts) >= 2:
                            try:
                                obj_id = int(parts[0])
                                obj_cat = parts[1].lower()
                                
                                # Map to our category system
                                for known_cat in OBJECT_CATEGORY_MAP.keys():
                                    if known_cat != "out-of-bounds":
                                        if known_cat == obj_cat or known_cat in obj_cat or obj_cat in known_cat:
                                            obj_id_to_cat[obj_id] = known_cat
                                            category_id_mapping[obj_id] = OBJECT_CATEGORY_MAP[known_cat]
                                            break
                            except ValueError:
                                continue
                
                # Print summary of mapped categories
                print(f"Successfully mapped {len(obj_id_to_cat)} objects to categories")
                cat_counts = collections.Counter(obj_id_to_cat.values())
                for cat, count in cat_counts.most_common():
                    print(f"  {cat}: {count} instances")
                
                # Special handling for important categories
                for obj_id, parts in enumerate(lines):
                    if isinstance(parts, str):
                        parts = parts.strip().split(',')
                        if len(parts) >= 3:
                            obj_cat = parts[2].strip('"').lower()
                    else:
                        continue
                        
                    # Directly map important furniture categories
                    for furniture in ["chair", "table", "sofa", "couch", "bed", "toilet", "tv", "sink", "cabinet"]:
                        if furniture in obj_cat:
                            obj_id_to_cat[obj_id] = furniture
                            category_id_mapping[obj_id] = OBJECT_CATEGORY_MAP[furniture]
                            print(f"  Added direct mapping for {obj_cat} -> {furniture}")
                
        except Exception as e:
            print(f"ERROR parsing semantic.txt: {e}")
    
    # Load the main scene for navigation and wall extraction
    try:
        print("Loading simulator...")
        sim = hab_utils.robust_load_sim(glb_path)
        print("Simulator loaded successfully")
    except Exception as e:
        print(f"ERROR loading simulator for {glb_path}: {e}")
        # Create a minimal empty point cloud
        with h5py.File(pc_save_path, "w") as fp:
            vertices = np.array([[0, 0, 0]])
            obj_ids = np.array([-1])
            sem_ids = np.array([OBJECT_CATEGORY_MAP["floor"]])
            colors = np.array([COLOR_PALETTE[OBJECT_CATEGORY_MAP["floor"] * 3 : (OBJECT_CATEGORY_MAP["floor"] + 1) * 3]])
            
            fp.create_dataset("vertices", data=vertices)
            fp.create_dataset("obj_ids", data=obj_ids)
            fp.create_dataset("sem_ids", data=sem_ids)
            fp.create_dataset("colors", data=colors)
        
        print(f"Created minimal point cloud for {os.path.basename(glb_path)}")
        return
    
    ############################################################################
    # Get vertices for all objects
    ############################################################################
    vertices = []
    colors = []
    obj_ids = []
    sem_ids = []
    
    # Process semantic mesh if it exists
    if os.path.isfile(semantic_glb_path):
        try:
            # Load semantic mesh
            print("Loading semantic mesh...")
            semantic_mesh = trimesh.load(semantic_glb_path)
            print("Semantic mesh loaded successfully")
            
            # Debug: Print mesh details
            if hasattr(semantic_mesh, 'geometry'):
                print(f"Found {len(semantic_mesh.geometry)} sub-meshes in semantic model")
                
                # Create a dictionary of processed objects
                processed_objects = set()
                
                # Pre-assign furniture objects to be elevated more
                furniture_keywords = ["chair", "table", "sofa", "couch", "bed", "desk", "cabinet", "shelf"]
                
                # Process each submesh
                for mesh_name, mesh in semantic_mesh.geometry.items():
                    # Try to extract object ID from mesh name
                    try:
                        # Look for patterns like 'object_NNN' or similar
                        match = None
                        for pattern in [r'_(\d+)', r'(\d+)_', r'(\d+)', r'object_(\d+)']:
                            match = re.search(pattern, mesh_name)
                            if match:
                                obj_id = int(match.group(1))
                                break
                                
                        if not match:
                            obj_id = -2  # Unknown object ID
                            print(f"Could not extract ID from mesh name: {mesh_name}")
                    except:
                        obj_id = -2
                    
                    # Skip if already processed this object ID
                    if obj_id in processed_objects:
                        continue
                    processed_objects.add(obj_id)
                    
                    # Determine semantic ID
                    sem_id = None
                    obj_cat = None
                    
                    # First check the manual mapping
                    if obj_id in category_id_mapping:
                        sem_id = category_id_mapping[obj_id]
                        obj_cat = obj_id_to_cat[obj_id]
                        print(f"Found mapping for object ID {obj_id}: {obj_cat} (sem_id: {sem_id})")
                    # Then check mesh name for common categories
                    elif obj_id not in [1, 2, -1]:  # Skip floor and wall
                        mesh_name_lower = mesh_name.lower()
                        
                        # Check for furniture keywords in mesh name
                        for keyword in furniture_keywords:
                            if keyword in mesh_name_lower:
                                # Find the corresponding category
                                for cat, id in OBJECT_CATEGORY_MAP.items():
                                    if keyword in cat:
                                        sem_id = id
                                        obj_cat = cat
                                        print(f"Assigned {keyword} category to {mesh_name} (ID: {obj_id})")
                                        break
                                if sem_id is not None:
                                    break
                    
                    # If still no category found, check for wall or floor
                    if sem_id is None:
                        if "wall" in mesh_name.lower():
                            sem_id = OBJECT_CATEGORY_MAP["wall"]
                            print(f"  Assigning as wall based on name: {mesh_name}")
                        elif "floor" in mesh_name.lower():
                            sem_id = OBJECT_CATEGORY_MAP["floor"]
                            print(f"  Assigning as floor based on name: {mesh_name}")
                        else:
                            # Skip unknown objects
                            print(f"  Skipping unknown object: {mesh_name}")
                            continue
                    
                    # Get color for this semantic ID
                    color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
                    
                    # Sample points from mesh
                    if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                        try:
                            print(f"  Sampling points from mesh with {len(mesh.triangles)} triangles")
                            # For furniture, use higher sampling density
                            local_density = sampling_density
                            if obj_cat in furniture_keywords or any(kw in str(obj_cat).lower() for kw in furniture_keywords):
                                local_density = sampling_density * 2
                                print(f"  Using higher density ({local_density}) for furniture: {obj_cat}")
                                
                            t_pts = hab_utils.dense_sampling_trimesh(mesh.triangles, local_density)
                            print(f"  Generated {len(t_pts)} points")
                            
                            for t_pt in t_pts:
                                vertices.append(t_pt)
                                obj_ids.append(obj_id)
                                sem_ids.append(sem_id)
                                colors.append(color)
                        except Exception as e:
                            print(f"  ERROR sampling points from mesh {mesh_name}: {e}")
            else:
                # Try to handle flat format semantic mesh
                print("Mesh does not have submeshes, trying flat format")
                # Generate points from triangles directly if possible
                try:
                    if hasattr(semantic_mesh, 'triangles') and len(semantic_mesh.triangles) > 0:
                        print(f"Found {len(semantic_mesh.triangles)} triangles in flat mesh")
                        t_pts = hab_utils.dense_sampling_trimesh(semantic_mesh.triangles, sampling_density)
                        print(f"Generated {len(t_pts)} points")
                        
                        # For flat meshes, we need to look up colors
                        # Try to extract colors if available
                        if hasattr(semantic_mesh, 'visual') and hasattr(semantic_mesh.visual, 'face_colors'):
                            face_colors = semantic_mesh.visual.face_colors
                            print(f"Found {len(face_colors)} face colors")
                            
                            for i, t_pt in enumerate(t_pts):
                                # Determine face index for this point
                                face_idx = i % len(semantic_mesh.triangles)
                                
                                # Get the color for this face
                                if face_idx < len(face_colors):
                                    face_color = face_colors[face_idx]
                                    
                                    # Try to map color to object ID
                                    # This requires knowing the color mapping scheme for your dataset
                                    color_hex = f"{face_color[0]:02X}{face_color[1]:02X}{face_color[2]:02X}"
                                    
                                    # Check if this color appears in the semantic.txt mapping
                                    obj_id = -1
                                    sem_id = OBJECT_CATEGORY_MAP["floor"]  # Default to floor
                                    
                                    # For each line in semantic.txt, check if the hex color matches
                                    with open(semantic_txt_path, 'r') as f:
                                        for line in f:
                                            parts = line.strip().split(',')
                                            if len(parts) >= 2 and parts[1].strip().upper() == color_hex:
                                                obj_id = int(parts[0])
                                                if obj_id in category_id_mapping:
                                                    sem_id = category_id_mapping[obj_id]
                                                break
                                else:
                                    # Default to floor category for unknown colors
                                    obj_id = -1
                                    sem_id = OBJECT_CATEGORY_MAP["floor"]
                                
                                # Get proper color format for this sem_id
                                color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
                                
                                vertices.append(t_pt)
                                obj_ids.append(obj_id)
                                sem_ids.append(sem_id)
                                colors.append(color)
                        else:
                            # No color info, just default to floor
                            for t_pt in t_pts:
                                sem_id = OBJECT_CATEGORY_MAP["floor"]
                                color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
                                
                                vertices.append(t_pt)
                                obj_ids.append(-1)  # Unknown object ID
                                sem_ids.append(sem_id)
                                colors.append(color)
                except Exception as e:
                    print(f"ERROR processing flat semantic mesh: {e}")
        except Exception as e:
            print(f"ERROR processing semantic mesh: {e}")
    else:
        print("WARNING: No semantic mesh file found - objects will be missing!")
    
    ############################################################################
    # Get vertices for navigable spaces
    ############################################################################
    try:
        print("Extracting navigable space...")
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        print(f"Found {len(navmesh_triangles)} navigable triangles")
        t_pts = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
        print(f"Generated {len(t_pts)} floor points")
        
        for t_pt in t_pts:
            obj_id = -1
            sem_id = OBJECT_CATEGORY_MAP["floor"]
            color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
            vertices.append(t_pt)
            obj_ids.append(obj_id)
            sem_ids.append(sem_id)
            colors.append(color)
    except Exception as e:
        print(f"ERROR extracting navigable spaces: {e}")

    ############################################################################
    # Get vertices for walls
    ############################################################################
    try:
        print("Extracting wall point clouds...")
        per_floor_wall_pc = extract_wall_point_clouds(
            glb_path, houses_dim_path, sampling_density=sampling_density
        )
        total_wall_points = 0
        for floor_id, points in per_floor_wall_pc.items():
            print(f"  Floor {floor_id}: {len(points)} wall points")
            total_wall_points += len(points)
            obj_id = -1
            sem_id = OBJECT_CATEGORY_MAP["wall"]
            color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
            for p in points:
                vertices.append(p)
                obj_ids.append(obj_id)
                sem_ids.append(sem_id)
                colors.append(color)
        print(f"Total wall points: {total_wall_points}")
    except Exception as e:
        print(f"ERROR extracting wall point clouds: {e}")
    
    # Close simulator
    sim.close()
    
    # Convert lists to numpy arrays
    if len(vertices) == 0:
        # Handle empty case - create minimal valid data
        print(f"WARNING: No points extracted for {glb_path}")
        vertices = np.array([[0, 0, 0]])
        obj_ids = np.array([-1])
        sem_ids = np.array([OBJECT_CATEGORY_MAP["floor"]])
        colors = np.array([COLOR_PALETTE[OBJECT_CATEGORY_MAP["floor"] * 3 : (OBJECT_CATEGORY_MAP["floor"] + 1) * 3]])
    else:
        print(f"Converting {len(vertices)} points to numpy arrays")
        vertices = np.array(vertices)
        obj_ids = np.array(obj_ids)
        sem_ids = np.array(sem_ids)
        colors = np.array(colors)
        
        # Print category statistics
        category_counts = collections.Counter(sem_ids)
        print("Point cloud category distribution:")
        for sem_id, count in category_counts.items():
            cat_name = "unknown"
            for name, id in OBJECT_CATEGORY_MAP.items():
                if id == sem_id:
                    cat_name = name
                    break
            print(f"  {cat_name}: {count} points ({count/len(sem_ids)*100:.1f}%)")

    # Save point cloud data
    print(f"Saving point cloud to {pc_save_path}")
    with h5py.File(pc_save_path, "w") as fp:
        fp.create_dataset("vertices", data=vertices)
        fp.create_dataset("obj_ids", data=obj_ids)
        fp.create_dataset("sem_ids", data=sem_ids)
        fp.create_dataset("colors", data=colors)
    
    print(f"Successfully saved point cloud for {os.path.basename(glb_path)} with {len(vertices)} points")
    
def get_scene_boundaries(inputs):
    scene_path, save_path = inputs
    sim = hab_utils.robust_load_sim(scene_path)
    floor_exts = hab_utils.get_floor_heights(
        sim, sampling_resolution=SAMPLING_RESOLUTION
    )
    scene_name = scene_path.split("/")[-1].split(".")[0]

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
        print(f"\n===== Processing environment: {env} =====")
        
        if os.path.isfile(map_save_path):
            print(f"Skipping - map already exists at {map_save_path}")
            continue

        houses_dim_path = os.path.join(houses_dim_root, env + ".json")
        if not os.path.isfile(houses_dim_path):
            print(f"Missing house dimensions for {env}, skipping...")
            continue

        with open(houses_dim_path, "r") as fp:
            houses_dim = json.load(fp)
        
        print(f"Loading point cloud from {obj_f}")
        f = h5py.File(obj_f, "r")

        # Generate floor-wise maps
        per_floor_dims = {}
        for key, val in houses_dim.items():
            match = re.search(f"{env}_(\d+)", key)
            if match:
                per_floor_dims[int(match.group(1))] = val
        
        print(f"Found {len(per_floor_dims)} floors in house dimensions")

        all_vertices = np.array(f["vertices"])
        all_obj_ids = np.array(f["obj_ids"])
        all_sem_ids = np.array(f["sem_ids"])
        all_colors = np.array(f["colors"])
        
        print(f"Loaded point cloud with {len(all_vertices)} points")
        
        # Print semantic category distribution
        sem_id_counts = collections.Counter(all_sem_ids)
        print("Point cloud semantic categories:")
        for sem_id, count in sorted(sem_id_counts.items()):
            cat_name = "unknown"
            for name, id in OBJECT_CATEGORY_MAP.items():
                if id == sem_id:
                    cat_name = name
                    break
            print(f"  ID {sem_id} ({cat_name}): {count} points")

        f.close()

        # --- change coordinates to match map
        # --  set discret dimensions
        center = np.array(houses_dim[env]["center"])
        sizes = np.array(houses_dim[env]["sizes"])
        sizes += 2  # -- pad env bboxes

        world_dim = sizes.copy()
        world_dim[1] = 0

        central_pos = center.copy()
        central_pos[1] = 0

        map_world_shift = central_pos - world_dim / 2

        world_dim_discret = [
            int(np.round(world_dim[0] / resolution)),
            0,
            int(np.round(world_dim[2] / resolution)),
        ]

        info[env] = {
            "dim": world_dim_discret,
            "central_pos": [float(x) for x in central_pos],
            "map_world_shift": [float(x) for x in map_world_shift],
            "resolution": resolution,
        }
        
        print(f"World dimensions (in cells): {world_dim_discret}")

        # Pre-assign objects to different floors
        per_floor_obj_ids = {floor_id: [] for floor_id in per_floor_dims.keys()}
        obj_ids_set = set(all_obj_ids.tolist())
        ## -1 corresponds to wall and floor
        if -1 in obj_ids_set:
            obj_ids_set.remove(-1)
            
        print(f"Assigning {len(obj_ids_set)} unique objects to floors")
        
        for obj_id in obj_ids_set:
            is_obj_id = all_obj_ids == obj_id
            obj_vertices = all_vertices[is_obj_id, :]
            # Get extents
            min_y = obj_vertices[:, 1].min()
            # Assign object to floor closest to it's min_y
            best_floor_id = None
            best_diff = math.inf
            for floor_id, floor_dims in per_floor_dims.items():
                diff = abs(min_y - floor_dims["ylo"])
                if (diff < best_diff) and min_y - floor_dims["ylo"] > -0.5:
                    best_diff = diff
                    best_floor_id = floor_id
            if best_floor_id is None:
                # Skip the object if it does not belong to any floor
                print(f"Object ID {obj_id} does not belong to any floor (min_y={min_y:.2f})")
                continue
                
            # Look up the semantic ID for this object
            obj_sem_ids = set(all_sem_ids[is_obj_id].tolist())
            if len(obj_sem_ids) > 0:
                sem_id = list(obj_sem_ids)[0]  # Just take the first one
                # Skip floor and wall objects
                if sem_id == OBJECT_CATEGORY_MAP["floor"] or sem_id == OBJECT_CATEGORY_MAP["wall"]:
                    continue
                    
            per_floor_obj_ids[best_floor_id].append(obj_id)
        
        # Print objects per floor stats
        for floor_id, obj_ids in per_floor_obj_ids.items():
            print(f"Floor {floor_id}: {len(obj_ids)} objects")
            if obj_ids:
                # Count object types on this floor
                floor_obj_types = []
                for obj_id in obj_ids:
                    obj_sem_ids = all_sem_ids[all_obj_ids == obj_id]
                    if len(obj_sem_ids) > 0:
                        floor_obj_types.append(obj_sem_ids[0])
                obj_type_counts = collections.Counter(floor_obj_types)
                for sem_id, count in sorted(obj_type_counts.items()):
                    cat_name = "unknown"
                    for name, id in OBJECT_CATEGORY_MAP.items():
                        if id == sem_id:
                            cat_name = name
                            break
                    print(f"  {cat_name}: {count} objects")

        # Build maps per floor
        per_floor_maps = {}
        for floor_id, floor_dims in per_floor_dims.items():
            print(f"\nProcessing floor {floor_id}")
            curr_floor_y = floor_dims["ylo"]
            if floor_id + 1 in per_floor_dims:
                next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y = math.inf
            
            print(f"Floor {floor_id} Y range: {curr_floor_y:.2f} - {min(next_floor_y, curr_floor_y+5):.2f}")

            # Get navigable and wall vertices based on height thresholds
            is_on_floor = (all_vertices[:, 1] >= curr_floor_y) & (
                all_vertices[:, 1] <= next_floor_y - 0.5
            )
            is_floor = (all_sem_ids == OBJECT_CATEGORY_MAP["floor"]) & is_on_floor
            is_wall = (all_sem_ids == OBJECT_CATEGORY_MAP["wall"]) & is_on_floor
            
            print(f"Points on floor: {np.sum(is_on_floor)}")
            print(f"Floor points: {np.sum(is_floor)}")
            print(f"Wall points: {np.sum(is_wall)}")

            # Get object vertices based on height thresholds for individual object instances
            is_object = np.zeros_like(is_on_floor)
            for obj_id in per_floor_obj_ids[floor_id]:
                is_object = is_object | (all_obj_ids == obj_id)
            
            print(f"Object points: {np.sum(is_object & ~is_floor & ~is_wall)}")

            # Slightly elevate objects above floor to ensure they appear in projection
            vertices_copy = np.copy(all_vertices)
            obj_mask = is_object & ~is_floor & ~is_wall
            furniture_mask = np.zeros_like(obj_mask)
            for furniture_id in [OBJECT_CATEGORY_MAP.get(f, -1) for f in ["chair", "table", "sofa", "couch", "bed", "toilet", "tv", "cabinet"]]:
                if furniture_id != -1:
                    furniture_mask = furniture_mask | (all_sem_ids == furniture_id)
            furniture_mask = furniture_mask & obj_mask
            # Elevate furniture more to ensure visibility
            vertices_copy[furniture_mask, 1] += 0.5  # Add 50cm to furniture heights
            # Add regular elevation to other objects
            regular_obj_mask = obj_mask & ~furniture_mask
            vertices_copy[regular_obj_mask, 1] += 0.2  # Add 20cm to other object heights

            mask = is_floor | is_wall | is_object
            vertices = vertices_copy[mask]
            obj_ids = np.copy(all_obj_ids[mask])
            sem_ids = np.copy(all_sem_ids[mask])
            
            print(f"Total points for projection: {len(vertices)}")
            
            # Count points by category for this floor
            sem_id_counts = collections.Counter(sem_ids)
            print("Floor point categories:")
            for sem_id, count in sorted(sem_id_counts.items()):
                cat_name = "unknown"
                for name, id in OBJECT_CATEGORY_MAP.items():
                    if id == sem_id:
                        cat_name = name
                        break
                print(f"  {cat_name}: {count} points")

            # -- some maps have 0 obj of interest
            if len(vertices) == 0:
                print(f"WARNING: No points for floor {floor_id}")
                info[env][floor_id] = {"y_min": 0.0}
                dims = (world_dim_discret[2], world_dim_discret[0])
                mask = np.zeros(dims, dtype=bool)
                map_z = np.zeros(dims, dtype=np.float32)
                map_instance = np.zeros(dims, dtype=np.int32)
                map_semantic = np.zeros(dims, dtype=np.int32)
                map_semantic_rgb = np.zeros((*dims, 3), dtype=np.uint8)
                per_floor_maps[floor_id] = {
                    "mask": mask,
                    "map_z": map_z,
                    "map_instance": map_instance,
                    "map_semantic": map_semantic,
                    "map_semantic_rgb": map_semantic_rgb,
                }
                continue

            vertices -= map_world_shift
            
            print(f"Applying map world shift: {map_world_shift}")

            # Set the min_y for the floor. This will be used during episode generation to find
            # a random navigable start location.
            floor_mask = sem_ids == OBJECT_CATEGORY_MAP["floor"]
            if np.any(floor_mask):
                min_y = vertices[floor_mask, 1].min()
                info[env][floor_id] = {"y_min": float(min_y.item())}
                print(f"Floor min_y: {min_y:.2f}")
            else:
                info[env][floor_id] = {"y_min": 0.0}
                print(f"WARNING: No floor points found for {env}, floor {floor_id}")
                continue

            # Reduce heights of floor and navigable space to ensure objects are taller.
            wall_mask = sem_ids == OBJECT_CATEGORY_MAP["wall"]
            vertices[wall_mask, 1] -= 0.5
            vertices[floor_mask, 1] -= 0.5
            
            print("Applied height adjustments to floor and wall points")

            # -- discretize point cloud
            vertices = torch.FloatTensor(vertices)
            obj_ids = torch.FloatTensor(obj_ids)
            sem_ids = torch.FloatTensor(sem_ids)

            y_values = vertices[:, 1]

            vertex_to_map_x = (vertices[:, 0] / resolution).round()
            vertex_to_map_z = (vertices[:, 2] / resolution).round()

            outside_map_indices = (
                (vertex_to_map_x >= world_dim_discret[0])
                + (vertex_to_map_z >= world_dim_discret[2])
                + (vertex_to_map_x < 0)
                + (vertex_to_map_z < 0)
            )
            
            print(f"Points outside map boundaries: {outside_map_indices.sum().item()} out of {len(vertices)}")

            # Skip if too many points are outside the map
            if outside_map_indices.sum() > 0.9 * len(vertices):
                print(f"WARNING: Most points are outside the map for {env}, floor {floor_id}")
                info[env][floor_id] = {"y_min": 0.0}
                dims = (world_dim_discret[2], world_dim_discret[0])
                mask = np.zeros(dims, dtype=bool)
                map_z = np.zeros(dims, dtype=np.float32)
                map_instance = np.zeros(dims, dtype=np.int32)
                map_semantic = np.zeros(dims, dtype=np.int32)
                map_semantic_rgb = np.zeros((*dims, 3), dtype=np.uint8)
                per_floor_maps[floor_id] = {
                    "mask": mask,
                    "map_z": map_z,
                    "map_instance": map_instance,
                    "map_semantic": map_semantic,
                    "map_semantic_rgb": map_semantic_rgb,
                }
                continue

            y_values = y_values[~outside_map_indices]
            vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
            vertex_to_map_x = vertex_to_map_x[~outside_map_indices]

            obj_ids = obj_ids[~outside_map_indices]
            sem_ids = sem_ids[~outside_map_indices]
            
            print(f"Points for projection after filtering: {len(y_values)}")

            # -- get the z values for projection
            # -- shift to positive values
            y_values = y_values - min_y
            y_values += 1.0
            
            print(f"Heights adjusted for projection (min={y_values.min().item():.2f}, max={y_values.max().item():.2f})")

            # -- projection
            print("Projecting points to 2D map...")
            feat_index = (
                world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x
            ).long()
            flat_highest_z = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            
            # Apply scatter max only if we have points to project
            if len(y_values) > 0:
                flat_highest_z, argmax_flat_spatial_map = scatter_max(
                    y_values,
                    feat_index,
                    dim=0,
                    out=flat_highest_z,
                )
                # NOTE: This is needed only for torch_scatter>=2.3
                argmax_flat_spatial_map[argmax_flat_spatial_map == y_values.shape[0]] = -1
            else:
                # Handle empty case
                argmax_flat_spatial_map = torch.zeros_like(flat_highest_z, dtype=torch.long) - 1
            
            m = argmax_flat_spatial_map >= 0
            
            # Check if projection worked
            if m.sum() == 0:
                print(f"WARNING: No points were projected to the 2D map")
                info[env][floor_id] = {"y_min": 0.0}
                dims = (world_dim_discret[2], world_dim_discret[0])
                mask = np.zeros(dims, dtype=bool)
                map_z = np.zeros(dims, dtype=np.float32)
                map_instance = np.zeros(dims, dtype=np.int32)
                map_semantic = np.zeros(dims, dtype=np.int32)
                map_semantic_rgb = np.zeros((*dims, 3), dtype=np.uint8)
                per_floor_maps[floor_id] = {
                    "mask": mask,
                    "map_z": map_z,
                    "map_instance": map_instance,
                    "map_semantic": map_semantic,
                    "map_semantic_rgb": map_semantic_rgb,
                }
                continue
            
            print(f"Projected {m.sum().item()} points to 2D map")
            
            flat_map_instance = (
                torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
            )

            flat_map_instance[m.view(-1)] = obj_ids[argmax_flat_spatial_map[m]]

            flat_map_semantic = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_map_semantic[m.view(-1)] = sem_ids[argmax_flat_spatial_map[m]]
            
            # Count pixels by category in the projected map
            map_sem_ids = sem_ids[argmax_flat_spatial_map[m]].cpu().numpy()
            sem_id_counts = collections.Counter(map_sem_ids)
            
            print("Projected map semantic categories:")
            for sem_id, count in sorted(sem_id_counts.items()):
                cat_name = "unknown"
                for name, id in OBJECT_CATEGORY_MAP.items():
                    if id == sem_id:
                        cat_name = name
                        break
                percentage = count/len(map_sem_ids)*100 if len(map_sem_ids) > 0 else 0
                print(f"  {cat_name}: {count} pixels ({percentage:.1f}%)")

            # -- format data
            mask = m.reshape(world_dim_discret[2], world_dim_discret[0])
            mask = mask.numpy()
            mask = mask.astype(bool)
            map_z = flat_highest_z.reshape(world_dim_discret[2], world_dim_discret[0])
            map_z = map_z.numpy()
            map_z = map_z.astype(np.float32)
            map_instance = flat_map_instance.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_instance = map_instance.numpy()
            map_instance = map_instance.astype(np.float32)
            map_semantic = flat_map_semantic.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_semantic = map_semantic.numpy()
            map_semantic = map_semantic.astype(np.float32)
            map_semantic_rgb = visualize_sem_map(map_semantic)
            
            print(f"Created semantic map of size {map_semantic.shape}")

            per_floor_maps[floor_id] = {
                "mask": mask,
                "map_z": map_z,
                "map_instance": map_instance,
                "map_semantic": map_semantic,
                "map_semantic_rgb": map_semantic_rgb,
            }

            rgb_save_path = os.path.join(save_dir, f"{env}_{floor_id}.png")
            cv2.imwrite(rgb_save_path, map_semantic_rgb)
            print(f"Saved visualization to {rgb_save_path}")

        with h5py.File(map_save_path, "w") as f:
            f.create_dataset(f"wall_sem_id", data=OBJECT_CATEGORY_MAP["wall"])
            f.create_dataset(f"floor_sem_id", data=OBJECT_CATEGORY_MAP["floor"])
            f.create_dataset(f"out-of-bounds_sem_id", data=OBJECT_CATEGORY_MAP["out-of-bounds"])
            
            # Save an overview of what was created
            floor_summary = []
            for floor_id, floor_map in per_floor_maps.items():
                floor_summary.append(f"Floor {floor_id}")
                map_semantic = floor_map["map_semantic"]
                unique_cats = np.unique(map_semantic)
                cats_str = []
                for cat in unique_cats:
                    if cat == 0:
                        continue  # Skip out-of-bounds
                    count = np.sum(map_semantic == cat)
                    cat_name = "unknown"
                    for name, id in OBJECT_CATEGORY_MAP.items():
                        if id == cat:
                            cat_name = name
                            break
                    cats_str.append(f"{cat_name}:{count}")
                floor_summary.append(", ".join(cats_str))
                
                # Save the floor data
                f.create_dataset(f"{floor_id}/mask", data=floor_map["mask"], dtype=bool)
                f.create_dataset(
                    f"{floor_id}/map_heights", data=floor_map["map_z"], dtype=np.float32
                )
                f.create_dataset(
                    f"{floor_id}/map_instance", data=floor_map["map_instance"], dtype=np.int32
                )
                f.create_dataset(
                    f"{floor_id}/map_semantic", data=floor_map["map_semantic"], dtype=np.int32
                )
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=floor_map["map_semantic_rgb"])
            
            print("\nSummary of created maps:")
            for line in floor_summary:
                print(line)

    # Save the info file with additional metadata
    info_path = os.path.join(save_dir, "semmap_GT_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f)
    print(f"\nSaved semantic map info to {info_path}")


if __name__ == "__main__":
    # Find all scene paths
    scene_paths = []
    if ACTIVE_DATASET == "hm3d":
        # In HM3D the directory structure might be different
        # Looking for '.basis.glb' files
        scene_paths = sorted(
            glob.glob(
                os.path.join(SCENES_ROOT, "**/*.basis.glb"),
                recursive=True,
            )
        )
        
        # Debug prints
        print(f"All basis.glb files found: {len(scene_paths)}")
        for path in scene_paths[:5]:
            print(f"  - {path}")
        
        # Check for corresponding semantic files
        filtered_scene_paths = []
        for path in scene_paths:
            semantic_glb_path = path.replace(".basis.glb", ".semantic.glb")
            semantic_txt_path = path.replace(".basis.glb", ".semantic.txt")
            
            # If semantic GLB exists, add it to our filtered list
            if os.path.isfile(semantic_glb_path):
                # Create empty semantic txt file if it doesn't exist
                if not os.path.isfile(semantic_txt_path):
                    with open(semantic_txt_path, 'w') as f:
                        pass  # Create empty file
                
                # Extract the scene ID from the filename
                filename = os.path.basename(path)
                scene_id = filename.split('.')[0]
                
                # Add to our list with the scene ID
                filtered_scene_paths.append((path, scene_id))
            
        scene_paths = [p[0] for p in filtered_scene_paths]
        scene_ids = [p[1] for p in filtered_scene_paths]
        
        # Debug prints
        print(f"Paths with semantic GLB files: {len(scene_paths)}")
        for path in scene_paths[:5]:
            print(f"  - {path}")
    else:
        # For Gibson and MP3D datasets
        scene_paths = glob.glob(os.path.join(SCENES_ROOT, "*.glb"))
        scene_ids = [os.path.basename(path).split(".")[0] for path in scene_paths]
    
    print(f"Number of available scenes: {len(scene_paths)}")
    
    context = mp.get_context("forkserver")
    pool = context.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)

    # Create output directories
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    os.makedirs(PC_SAVE_ROOT, exist_ok=True)
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)

    # Extract scene_boundaries
    print("===========> Extracting scene boundaries")
    inputs = []
    for i, scene_path in enumerate(scene_paths):
        scene_name = scene_ids[i]
        save_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
        if not os.path.isfile(save_path):
            inputs.append((scene_path, save_path))
    
    if inputs:
        _ = list(tqdm.tqdm(pool.imap(get_scene_boundaries, inputs), total=len(inputs)))

    # Generate point-clouds for each scene
    print("===========> Extracting point-clouds")
    inputs = []
    for i, scene_path in enumerate(scene_paths):
        scene_name = scene_ids[i]
        
        if ACTIVE_DATASET == "hm3d":
            semantic_glb_path = scene_path.replace(".basis.glb", ".semantic.glb")
            semantic_txt_path = scene_path.replace(".basis.glb", ".semantic.txt")
            
            pc_save_path = os.path.join(PC_SAVE_ROOT, f"{scene_name}.h5")
            if not os.path.isfile(pc_save_path) and os.path.isfile(semantic_glb_path):
                # Make sure scene boundaries exist
                sb_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
                if not os.path.isfile(sb_path):
                    print(f"Missing scene boundaries for {scene_name}, skipping...")
                    continue
                    
                inputs.append(
                    (
                        extract_hm3d_scene_point_clouds,
                        scene_path,
                        semantic_glb_path,
                        semantic_txt_path,
                        sb_path,
                        pc_save_path,
                    )
                )
        else:
            ply_path = scene_path.replace(".glb", "_semantic.ply")
            scn_path = scene_path.replace(".glb", ".scn")
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
    
    if inputs:
        _ = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(inputs)))

    # Extract semantic maps
    print("===========> Extracting semantic maps")
    convert_point_cloud_to_semantic_map(PC_SAVE_ROOT, SB_SAVE_ROOT, SEM_SAVE_ROOT)