import glob
import json
import math
import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
import csv
import logging # Added for logging

import cv2
import h5py
import numpy as np
# import torch # Not directly used in this script, but torch_scatter is
import tqdm
import trimesh
from PIL import Image, ImageDraw, ImageFont
from torch_scatter import scatter_max # type: ignore

Image.MAX_IMAGE_PIXELS = 1000000000
import poni.hab_utils as hab_utils
from matplotlib import font_manager
from plyfile import PlyData

# Assuming poni.constants has been updated for HM3D
from poni.constants import (
    d3_40_colors_rgb,
    OBJECT_CATEGORIES,
    OBJECT_CATEGORY_MAP,
    SPLIT_SCENES,
)

# --- Setup Logging ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

random.seed(123)

# --- Dataset-specific configurations ---
if "ACTIVE_DATASET" not in os.environ:
    logger.error("CRITICAL: ACTIVE_DATASET environment variable is not set. Exiting.")
    logger.error("Please set it, e.g., export ACTIVE_DATASET=hm3d")
    exit(1)
ACTIVE_DATASET = os.environ["ACTIVE_DATASET"]
logger.info(f"ACTIVE_DATASET set to: {ACTIVE_DATASET}")


# Initialize current dataset variables
CURRENT_OBJECT_CATEGORIES = OBJECT_CATEGORIES.get(ACTIVE_DATASET, [])
CURRENT_OBJECT_CATEGORY_MAP = OBJECT_CATEGORY_MAP.get(ACTIVE_DATASET, {})
CURRENT_OBJECT_COLORS_VIS = []

if not CURRENT_OBJECT_CATEGORIES or not CURRENT_OBJECT_CATEGORY_MAP:
    logger.error(f"CRITICAL: Categories for dataset '{ACTIVE_DATASET}' are not defined in poni/constants.py. Exiting.")
    exit(1)


if ACTIVE_DATASET == "gibson":
    GIBSON_OBJECT_COLORS = [
        (0.9400000000000001, 0.7818, 0.66), (0.9400000000000001, 0.8868, 0.66),
        (0.8882000000000001, 0.9400000000000001, 0.66), (0.7832000000000001, 0.9400000000000001, 0.66),
        (0.6782000000000001, 0.9400000000000001, 0.66), (0.66, 0.9400000000000001, 0.7468000000000001),
        (0.66, 0.9400000000000001, 0.8518000000000001), (0.66, 0.9232, 0.9400000000000001),
        (0.66, 0.8182, 0.9400000000000001), (0.66, 0.7132, 0.9400000000000001),
        (0.7117999999999999, 0.66, 0.9400000000000001), (0.8168, 0.66, 0.9400000000000001),
        (0.9218, 0.66, 0.9400000000000001), (0.9400000000000001, 0.66, 0.8531999999999998),
        (0.9400000000000001, 0.66, 0.748199999999999),
    ]
    CURRENT_OBJECT_COLORS_VIS = GIBSON_OBJECT_COLORS
    SCENES_ROOT = "data/scene_datasets/gibson_semantic"
    SB_SAVE_ROOT = "data/semantic_maps/gibson/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/gibson/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps"
    NUM_WORKERS = 12
    MAX_TASKS_PER_CHILD = None
    SAMPLING_RESOLUTION = 0.10
    WALL_THRESH = [0.25, 1.25]
elif ACTIVE_DATASET == "mp3d":
    MP3D_OBJECT_COLORS_TEMP = []
    if len(CURRENT_OBJECT_CATEGORIES) > 2:
        for color in d3_40_colors_rgb[: len(CURRENT_OBJECT_CATEGORIES) - 2]:
            MP3D_OBJECT_COLORS_TEMP.append((color.astype(np.float32) / 255.0).tolist())
    CURRENT_OBJECT_COLORS_VIS = MP3D_OBJECT_COLORS_TEMP
    SCENES_ROOT = "data/scene_datasets/mp3d_uncompressed"
    SB_SAVE_ROOT = "data/semantic_maps/mp3d/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/mp3d/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/mp3d/semantic_maps"
    NUM_WORKERS = 8
    MAX_TASKS_PER_CHILD = 2
    SAMPLING_RESOLUTION = 0.20
    WALL_THRESH = [0.25, 1.25]
elif ACTIVE_DATASET == "hm3d":
    HM3D_OBJECT_COLORS_TEMP = []
    if len(CURRENT_OBJECT_CATEGORIES) > 2:
        for color in d3_40_colors_rgb[: len(CURRENT_OBJECT_CATEGORIES) - 2]:
            HM3D_OBJECT_COLORS_TEMP.append((color.astype(np.float32) / 255.0).tolist())
    CURRENT_OBJECT_COLORS_VIS = HM3D_OBJECT_COLORS_TEMP
    SCENES_ROOT = "data/scene_datasets/hm3d_uncompressed/"
    if not os.path.isdir(SCENES_ROOT):
        logger.error(f"CRITICAL: HM3D SCENES_ROOT does not exist or is not a directory: {SCENES_ROOT}")
        logger.error("Please ensure this path is correct relative to your PONI_ROOT.")
        exit(1)

    SB_SAVE_ROOT = "data/semantic_maps/hm3d/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/hm3d/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/hm3d/semantic_maps"
    NUM_WORKERS = int(os.cpu_count() * 0.75) if os.cpu_count() and os.cpu_count() > 1 else 1
    MAX_TASKS_PER_CHILD = None
    SAMPLING_RESOLUTION = 0.10
    WALL_THRESH = [0.25, 1.25]
else:
    logger.error(f"Unsupported ACTIVE_DATASET: {ACTIVE_DATASET}")
    exit(1)

logger.info(f"Using SCENES_ROOT: {SCENES_ROOT}")
logger.info(f"Outputting scene boundaries to: {SB_SAVE_ROOT}")
logger.info(f"Outputting point clouds to: {PC_SAVE_ROOT}")
logger.info(f"Outputting semantic maps to: {SEM_SAVE_ROOT}")


COLOR_PALETTE_VIS = [
    1.0, 1.0, 1.0,
    0.9, 0.9, 0.9,
    0.3, 0.3, 0.3,
    *[oci for oc in CURRENT_OBJECT_COLORS_VIS for oci in oc],
]
LEGEND_PALETTE_VIS = [
    (1.0, 1.0, 1.0), (0.9, 0.9, 0.9), (0.3, 0.3, 0.3),
    *CURRENT_OBJECT_COLORS_VIS,
]

HM3D_RAW_TO_PONI_CATEGORY_MAP = {
    "floor": "floor", "wall": "wall", "ceiling": "ceiling", "chair": "chair",
    "table": "table", "desk": "table", "door": "door", "window": "window",
    "window frame": "window", "door frame": "door", "sofa": "sofa", "couch": "sofa",
    "bed": "bed", "sink": "sink", "toilet": "toilet", "tv": "tv_monitor",
    "tv_monitor": "tv_monitor", "cabinet": "cabinet", "shelf": "shelving",
    "shelves": "shelving", "plant": "plant", "potted plant": "plant",
    "potted_plant": "plant", "picture": "picture", "painting": "picture",
    "mirror": "mirror", "lighting": "lighting", "lamp": "lighting",
    "ceiling lamp": "lighting", "wall lamp": "lighting", "appliance": "appliances",
    "refrigerator": "appliances", "oven": "appliances", "stove": "appliances",
    "microwave": "appliances", "dishwasher": "appliances",
    "kitchen appliance": "appliances", "heater": "appliances",
    "objects": "objects", "decoration": "objects", "cushion": "cushion",
    "pillow": "cushion", "towel": "towel", "chest_of_drawers": "chest_of_drawers",
    "stool": "stool", "bathtub": "bathtub", "shower": "shower",
    "shower cabin": "shower", "counter": "counter", "kitchen counter": "counter",
    "countertop": "counter", "fireplace": "fireplace",
    "gym_equipment": "gym_equipment", "seating": "seating", "clothes": "clothes",
    "book": "book", "books": "book", "rug": "rug", "box": "objects",
    "curtain": "curtain", "blinds": "blinds", "vent": "vent", "fire alarm": "fire alarm",
    "sculpture": "objects", "drawer": "objects",
    "sheet": "objects",
    "balustrade": "railing",
    "railing": "railing",
    "stairs": "stairs",
    "beam": "beam",
    "wall panel": "wall",
    "toilet paper": "objects",
    "tap": "objects",
    "step": "stairs",
    "display cabinet": "cabinet",
    "bottles of wine": "objects",
    "kitchen cabinet": "cabinet",
    "paper towel": "objects",
    "kitchen extractor": "appliances",
    "mini fridge": "appliances",
    "kitchen countertop item": "objects",
    "kettle": "appliances",
    "coffee machine": "appliances",
    "dishrag": "objects",
    "alarm": "objects",
    "dining table": "table",
    "dining chair": "chair",
    "air conditioner": "appliances",
    "duct": "objects",
    "unknown": "objects",
}
if ACTIVE_DATASET == "hm3d" and len(HM3D_RAW_TO_PONI_CATEGORY_MAP) < 10:
    logger.warning("HM3D_RAW_TO_PONI_CATEGORY_MAP seems sparsely populated. Please ensure it's comprehensive.")

def map_hm3d_raw_label_to_poni_category(raw_label):
    cleaned_label = raw_label.lower().strip().replace('_', ' ').replace('-', ' ')
    return HM3D_RAW_TO_PONI_CATEGORY_MAP.get(cleaned_label, None)


def get_palette_image():
    mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(mpl_font)
    font = ImageFont.truetype(font=file, size=20)
    cat_size = 30
    buf_size = 10
    text_width = 170

    num_display_categories = min(len(CURRENT_OBJECT_CATEGORIES), len(LEGEND_PALETTE_VIS))
    if num_display_categories == 0:
        logger.warning("No categories or legend palette to display in get_palette_image.")
        return np.zeros((100,100,3), dtype=np.uint8)

    image_pil = Image.new(
        "RGB",
        (cat_size + buf_size + text_width, cat_size * num_display_categories),
        (255,255,255)
    )
    draw = ImageDraw.Draw(image_pil)

    for i in range(num_display_categories):
        category = CURRENT_OBJECT_CATEGORIES[i]
        color_tuple = LEGEND_PALETTE_VIS[i]
        color_int = tuple([int(c * 255) for c in color_tuple])
        draw.rectangle(
            [(0, i * cat_size), (cat_size, (i + 1) * cat_size)],
            fill=color_int, outline=(0, 0, 0), width=1,
        )
        text_color = (0,0,0) if sum(color_int) > (255*3/2) else (255,255,255)
        try:
            text_bbox = font.getbbox(category)
            text_h = text_bbox[3] - text_bbox[1]
            text_y_pos = i * cat_size + (cat_size - text_h) // 2
            draw.text( (cat_size + buf_size, text_y_pos), category, font=font, fill=text_color,)
        except AttributeError:
             draw.text( (cat_size + buf_size, i * cat_size + 5), category, font=font, fill=text_color,)
    return np.array(image_pil)


def extract_scene_point_clouds(
    glb_path,
    semantic_file_path_or_info,
    houses_dim_path,
    pc_save_path,
    sampling_density=1600.0,
):
    logger.info(f"Processing scene for point clouds: {os.path.basename(glb_path)}")
    obj_id_to_cat = {}

    if ACTIVE_DATASET == "hm3d":
        if not semantic_file_path_or_info or not os.path.isfile(semantic_file_path_or_info):
            logger.error(f"HM3D semantic.txt file not found: {semantic_file_path_or_info} for scene {glb_path}")
            return
        try:
            with open(semantic_file_path_or_info, 'r') as f:
                first_line = f.readline().strip()
                is_header = "HM3D Semantic Annotations" in first_line or \
                            (',' in first_line and ("instance_id" in first_line.lower() or "object_id" in first_line.lower()))
                if not is_header:
                    f.seek(0)

                reader = csv.reader(f)
                if is_header and "HM3D Semantic Annotations" not in first_line :
                    try:
                        next(reader)
                    except StopIteration:
                        logger.warning(f"Empty semantic file after header: {semantic_file_path_or_info}")
                        return

                for row_idx, row in enumerate(reader):
                    if not row or len(row) < 3:
                        continue
                    instance_id_str, _, raw_category_label = row[0].strip(), row[1], row[2].strip().strip('"')
                    poni_category_name = map_hm3d_raw_label_to_poni_category(raw_category_label)
                    if poni_category_name and \
                       poni_category_name in CURRENT_OBJECT_CATEGORY_MAP and \
                       poni_category_name not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                        try:
                            instance_id = int(instance_id_str)
                            obj_id_to_cat[instance_id] = poni_category_name
                        except ValueError:
                            logger.warning(f"Could not parse instance_id '{instance_id_str}' as int in {os.path.basename(semantic_file_path_or_info)} at row {row_idx+1}")
        except Exception as e:
            logger.error(f"Error parsing HM3D semantic.txt file {os.path.basename(semantic_file_path_or_info)}: {e}", exc_info=True)
            return
        logger.info(f"HM3D: Loaded {len(obj_id_to_cat)} object instance mappings from {os.path.basename(semantic_file_path_or_info)}.")
    elif ACTIVE_DATASET == "gibson":
        ply_path_gibson = semantic_file_path_or_info
        if os.path.isfile(ply_path_gibson):
            try:
                sim = hab_utils.robust_load_sim(glb_path)
                if sim.semantic_scene and sim.semantic_scene.objects:
                    for obj in sim.semantic_scene.objects:
                        try: obj_instance_id = int(obj.id.split('_')[-1])
                        except: continue
                        raw_cat_name = obj.category.name()
                        if raw_cat_name in CURRENT_OBJECT_CATEGORY_MAP and \
                           raw_cat_name not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                            obj_id_to_cat[obj_instance_id] = raw_cat_name
                sim.close()
            except Exception as e:
                logger.error(f"Error using sim for Gibson semantics {glb_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Gibson semantic PLY file not found: {ply_path_gibson}")
    elif ACTIVE_DATASET == "mp3d":
        scn_path = semantic_file_path_or_info
        if os.path.isfile(scn_path):
            with open(scn_path) as fp: scn_data = json.load(fp)
            for obj in scn_data["objects"]:
                if obj["class_"] in CURRENT_OBJECT_CATEGORY_MAP and \
                   obj["class_"] not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                    obj_id_to_cat[obj["id"]] = obj["class_"]
        else:
            logger.warning(f"MP3D .scn file not found: {scn_path}. Consider sim fallback if needed.")


    vertices = []
    colors = []
    obj_ids_out = []
    sem_ids_out = []

    if ACTIVE_DATASET == "hm3d":
        logger.debug(f"HM3D: Extracting object meshes from GLB: {os.path.basename(glb_path)}")
        try:
            scene_trimesh_obj = trimesh.load(glb_path, force='scene', process=False)
            if not isinstance(scene_trimesh_obj, trimesh.Scene) or not hasattr(scene_trimesh_obj, 'graph'):
                logger.error(f"Could not load {os.path.basename(glb_path)} as a trimesh.Scene with a graph. Loaded as {type(scene_trimesh_obj)}. Cannot extract instances by node name.")
                scene_trimesh_obj = None

            if scene_trimesh_obj:
                for node_name in scene_trimesh_obj.graph.nodes_geometry:
                    transform, geometry_name = scene_trimesh_obj.graph[node_name]
                    
                    parsed_instance_id = -1
                    if node_name.isdigit(): 
                        parsed_instance_id = int(node_name)
                    else: 
                        patterns = [r'object_(\d+)', r'mesh_(\d+)', r'^(\d+)[-_a-zA-Z]*', r'[._-](\d+)$', r'instance_(\d+)']
                        for pat in patterns:
                            match = re.search(pat, node_name, re.IGNORECASE) 
                            if match:
                                try:
                                    parsed_instance_id = int(match.groups()[-1])
                                    break
                                except (ValueError, IndexError): continue
                    
                    if parsed_instance_id != -1 and parsed_instance_id in obj_id_to_cat:
                        poni_category_name = obj_id_to_cat[parsed_instance_id]
                        sem_id = CURRENT_OBJECT_CATEGORY_MAP[poni_category_name]
                        mesh_instance = scene_trimesh_obj.geometry[geometry_name]

                        if not isinstance(mesh_instance, trimesh.Trimesh) or not mesh_instance.vertices.shape[0] > 0:
                            continue
                        
                        mesh_instance_transformed = mesh_instance.copy()
                        mesh_instance_transformed.apply_transform(transform)
                        
                        if not hasattr(mesh_instance_transformed, 'triangles') or len(mesh_instance_transformed.triangles) == 0:
                            continue

                        t_pts = hab_utils.dense_sampling_trimesh(mesh_instance_transformed.triangles, sampling_density)
                        
                        obj_color_index = sem_id - 2 
                        color_for_obj = (0.5, 0.5, 0.5)
                        if obj_color_index >= 0 and obj_color_index < len(CURRENT_OBJECT_COLORS_VIS):
                            color_for_obj = CURRENT_OBJECT_COLORS_VIS[obj_color_index]
                        elif sem_id < len(LEGEND_PALETTE_VIS):
                             color_for_obj = LEGEND_PALETTE_VIS[sem_id]

                        for t_pt in t_pts:
                            vertices.append(t_pt); obj_ids_out.append(parsed_instance_id); sem_ids_out.append(sem_id); colors.append(list(color_for_obj))
        except Exception as e:
            logger.error(f"Error loading/processing HM3D GLB {os.path.basename(glb_path)} for object extraction: {e}", exc_info=True)
    # ... (Keep Gibson/MP3D object extraction)

    logger.info(f"Collected {len(vertices)} object vertices for {os.path.basename(glb_path)}.")

    try:
        # For HM3D, the scene_dataset_config might be needed by robust_load_sim
        hm3d_scene_dataset_config = None
        if ACTIVE_DATASET == "hm3d":
            # Path relative to SCENES_ROOT (e.g. data/objectnav_hm3d_v1/scene_datasets/hm3d_uncompressed/)
            # The config is likely one level up, in data/objectnav_hm3d_v1/scene_datasets/
            potential_cfg_path = os.path.abspath(os.path.join(SCENES_ROOT, "..", "hm3d.scene_dataset_config.json"))
            if os.path.isfile(potential_cfg_path):
                hm3d_scene_dataset_config = potential_cfg_path
                logger.info(f"Using HM3D scene dataset config: {hm3d_scene_dataset_config}")
            else:
                logger.warning(f"HM3D scene_dataset_config.json not found at expected path: {potential_cfg_path}. Sim might not load all assets correctly.")


        sim = hab_utils.robust_load_sim(glb_path, scene_dataset_config_file=hm3d_scene_dataset_config)
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        if navmesh_triangles.size > 0:
            t_pts_nav = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
            nav_sem_id = CURRENT_OBJECT_CATEGORY_MAP.get("floor", 0)
            nav_color_tuple = LEGEND_PALETTE_VIS[nav_sem_id] if nav_sem_id < len(LEGEND_PALETTE_VIS) else LEGEND_PALETTE_VIS[0]
            nav_color = [float(c) for c in nav_color_tuple]
            for t_pt in t_pts_nav:
                vertices.append(t_pt); obj_ids_out.append(-1); sem_ids_out.append(nav_sem_id); colors.append(nav_color)
            logger.debug(f"Collected {len(t_pts_nav)} navmesh vertices.")
        else:
            logger.warning(f"No navmesh vertices found for {os.path.basename(glb_path)}.")
        sim.close()
    except Exception as e:
        logger.error(f"Error processing navmesh for {os.path.basename(glb_path)}: {e}", exc_info=True)

    try:
        per_floor_wall_pc = extract_wall_point_clouds(
            glb_path, houses_dim_path, sampling_density=sampling_density,
            current_wall_thresh=WALL_THRESH
        )
        wall_sem_id = CURRENT_OBJECT_CATEGORY_MAP.get("wall", 1)
        wall_color_tuple = LEGEND_PALETTE_VIS[wall_sem_id] if wall_sem_id < len(LEGEND_PALETTE_VIS) else LEGEND_PALETTE_VIS[0]
        wall_color = [float(c) for c in wall_color_tuple]
        num_wall_pts = 0
        for _, points_on_floor in per_floor_wall_pc.items():
            if points_on_floor.shape[0] > 0:
                num_wall_pts += len(points_on_floor)
                for p_wall in points_on_floor:
                    vertices.append(p_wall); obj_ids_out.append(-1); sem_ids_out.append(wall_sem_id); colors.append(wall_color)
        logger.debug(f"Collected {num_wall_pts} wall vertices.")
    except Exception as e:
        logger.error(f"Error processing walls for {os.path.basename(glb_path)}: {e}", exc_info=True)

    if not vertices:
        logger.warning(f"No vertices extracted at all for scene {os.path.basename(glb_path)}. Skipping save of H5 point cloud.")
        return

    vertices_arr = np.array(vertices)
    obj_ids_arr = np.array(obj_ids_out)
    sem_ids_arr = np.array(sem_ids_out)
    colors_arr = np.array(colors)

    try:
        with h5py.File(pc_save_path, "w") as fp:
            fp.create_dataset("vertices", data=vertices_arr)
            fp.create_dataset("obj_ids", data=obj_ids_arr)
            fp.create_dataset("sem_ids", data=sem_ids_arr)
            fp.create_dataset("colors", data=colors_arr)
        logger.info(f"Saved point cloud for {os.path.basename(glb_path)} to {pc_save_path} with {len(vertices_arr)} points.")
    except Exception as e:
        logger.error(f"Failed to save HDF5 point cloud for {os.path.basename(glb_path)}: {e}", exc_info=True)


def get_scene_boundaries(inputs_tuple): # Modified to accept a tuple
    scene_path, save_path = inputs_tuple
    sim = None # Initialize sim to None for finally block
    try:
        logger.info(f"get_scene_boundaries: Starting for {os.path.basename(scene_path)}")
        
        scene_dataset_config = None
        if ACTIVE_DATASET == "hm3d":
            # SCENES_ROOT is like ".../hm3d_uncompressed/"
            # Config is often one level up, e.g., ".../scene_datasets/hm3d.scene_dataset_config.json"
            # Or sometimes named like "hm3d_annotated_basis.scene_dataset_config.json"
            # It's better if this path can be made more explicit or passed if needed.
            # For now, let's try a common pattern for Habitat datasets.
            # The dir containing SCENES_ROOT (e.g., .../scene_datasets/)
            scenes_root_parent_dir = os.path.abspath(os.path.join(SCENES_ROOT, ".."))
            
            potential_config_names = [
                "hm3d.scene_dataset_config.json",
                "hm3d_annotated_basis.scene_dataset_config.json", # From Adsız.jpg
                "hm3d_semantic.scene_dataset_config.json"
            ]
            for cfg_name in potential_config_names:
                potential_cfg_path = os.path.join(scenes_root_parent_dir, cfg_name)
                if os.path.isfile(potential_cfg_path):
                    scene_dataset_config = potential_cfg_path
                    logger.info(f"Found HM3D scene dataset config: {scene_dataset_config}")
                    break
            if not scene_dataset_config:
                 logger.warning(f"HM3D scene_dataset_config.json not found in {scenes_root_parent_dir} with common names. Sim might use defaults or fail for complex scenes.")


        sim = hab_utils.robust_load_sim(scene_path, scene_dataset_config_file=scene_dataset_config)
        if sim is None: # robust_load_sim might raise an error or return None if it handles internally
            logger.error(f"Simulator could not be initialized for {os.path.basename(scene_path)}. Cannot get boundaries.")
            return # Exit this task

        floor_exts = hab_utils.get_floor_heights(sim, sampling_resolution=SAMPLING_RESOLUTION)
        
        # Get scene_name from scene_path (e.g., "00000-kfPV7w3FaU5" from ".../00000-kfPV7w3FaU5.semantic.glb")
        # This should match the folder name if files are named like <folder>/<folder_short_id>.<type>.glb
        scene_name_for_json = os.path.basename(os.path.dirname(scene_path))


        scene_boundaries_data = {}
        overall_bounds = sim.pathfinder.get_bounds()
        if overall_bounds[0] is not None and overall_bounds[1] is not None:
            scene_boundaries_data[scene_name_for_json] = hab_utils.convert_lu_bound_to_smnet_bound(overall_bounds) # Assuming this helper is in hab_utils
        else:
            logger.warning(f"Could not get overall bounds for {scene_name_for_json}")

        for fidx, fext in enumerate(floor_exts):
            y_min_for_floor = fext["min"] - 0.2
            y_max_for_floor = fext["max"] + 0.2
            bounds = hab_utils.get_navmesh_extents_at_y(sim, y_bounds=(y_min_for_floor, y_max_for_floor))
            if bounds[0] is not None and bounds[1] is not None:
                scene_boundaries_data[f"{scene_name_for_json}_{fidx}"] = hab_utils.convert_lu_bound_to_smnet_bound(bounds)
            else:
                logger.warning(f"Could not get bounds for floor {fidx} of {scene_name_for_json}")
        
        with open(save_path, "w") as fp:
            json.dump(scene_boundaries_data, fp, indent=4)
        logger.info(f"Saved scene boundaries for {scene_name_for_json} to {save_path}")

    except Exception as e:
        logger.error(f"!!! ERROR in get_scene_boundaries for {os.path.basename(scene_path)}: {e}", exc_info=True)
    finally:
        if sim is not None:
            sim.close()
            logger.debug(f"Simulator closed for {os.path.basename(scene_path)}")


def extract_scene_point_clouds(
    glb_path,
    semantic_file_path_or_info,
    houses_dim_path,
    pc_save_path,
    sampling_density=1600.0,
):
    # Re-check ACTIVE_DATASET from environment within the function, especially if called by a worker
    worker_active_dataset = os.environ.get("ACTIVE_DATASET")
    if not worker_active_dataset:
        logger.error("Worker: ACTIVE_DATASET env variable not found inside extract_scene_point_clouds.")
        # Fallback to module level if absolutely necessary, but this indicates an issue
        worker_active_dataset = MODULE_LEVEL_ACTIVE_DATASET
        logger.warning(f"Worker: Falling back to module-level ACTIVE_DATASET: {worker_active_dataset}")
    
    logger.info(f"Worker for {os.path.basename(glb_path)}: ACTIVE_DATASET is '{worker_active_dataset}'. Semantic file: {semantic_file_path_or_info}")


    obj_id_to_cat = {}

    if worker_active_dataset == "hm3d":
        if not semantic_file_path_or_info or not os.path.isfile(semantic_file_path_or_info):
            logger.error(f"HM3D semantic.txt file not found: {semantic_file_path_or_info} for scene {glb_path}")
            return
        try:
            with open(semantic_file_path_or_info, 'r') as f:
                first_line = f.readline().strip()
                is_header = "HM3D Semantic Annotations" in first_line or \
                            (',' in first_line and ("instance_id" in first_line.lower() or "object_id" in first_line.lower()))
                if not is_header:
                    f.seek(0)

                reader = csv.reader(f)
                if is_header and "HM3D Semantic Annotations" not in first_line :
                    try:
                        next(reader)
                    except StopIteration:
                        logger.warning(f"Empty semantic file after header: {semantic_file_path_or_info}")
                        return

                for row_idx, row in enumerate(reader):
                    if not row or len(row) < 3:
                        continue
                    instance_id_str, _, raw_category_label = row[0].strip(), row[1], row[2].strip().strip('"')
                    poni_category_name = map_hm3d_raw_label_to_poni_category(raw_category_label)
                    if poni_category_name and \
                       poni_category_name in CURRENT_OBJECT_CATEGORY_MAP and \
                       poni_category_name not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                        try:
                            instance_id = int(instance_id_str)
                            obj_id_to_cat[instance_id] = poni_category_name
                        except ValueError:
                            logger.warning(f"Could not parse instance_id '{instance_id_str}' as int in {os.path.basename(semantic_file_path_or_info)} at row {row_idx+1}")
        except Exception as e:
            logger.error(f"Error parsing HM3D semantic.txt file {os.path.basename(semantic_file_path_or_info)}: {e}", exc_info=True)
            return
        logger.info(f"HM3D: Loaded {len(obj_id_to_cat)} object instance mappings from {os.path.basename(semantic_file_path_or_info)}.")
    
    elif worker_active_dataset == "gibson":
        logger.info(f"Gibson: Attempting to load semantic info for {os.path.basename(glb_path)}")
        sim_gibson = None
        try:
            # For Gibson, semantic_file_path_or_info is the _semantic.ply, but sim is often preferred for instance info
            # If robust_load_sim needs the ply, it should be passed as scene_dataset_config or handled internally
            sim_gibson = hab_utils.robust_load_sim(glb_path, scene_dataset_config_file=None) # Try without specific dataset config first for Gibson
            if sim_gibson is not None:
                logger.debug(f"Gibson: Simulator loaded for {os.path.basename(glb_path)}. Type of sim.semantic_scene: {type(sim_gibson.semantic_scene)}")
                if sim_gibson.semantic_scene is not None:
                    if not hasattr(sim_gibson.semantic_scene, 'objects') or sim_gibson.semantic_scene.objects is None:
                        logger.warning(f"Gibson: sim.semantic_scene for {os.path.basename(glb_path)} has no 'objects' or it's None.")
                    else:
                        try:
                            scene_objects = sim_gibson.semantic_scene.objects
                            logger.debug(f"Gibson: Found {len(scene_objects)} potential objects in sim.semantic_scene for {os.path.basename(glb_path)}.")
                            for obj_idx, obj in enumerate(scene_objects):
                                if obj is None or obj.category is None: continue
                                obj_instance_id_str = obj.id.split('_')[-1] if '_' in obj.id else obj.id
                                try:
                                    obj_instance_id = int(obj_instance_id_str)
                                except ValueError: continue
                                raw_cat_name = obj.category.name()
                                poni_category_name = raw_cat_name
                                if poni_category_name in CURRENT_OBJECT_CATEGORY_MAP and \
                                   poni_category_name not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                                    obj_id_to_cat[obj_instance_id] = poni_category_name
                        except Exception as ex_obj: # Catch specific errors like KeyError or AttributeError
                            logger.error(f"Gibson: Error accessing sim.semantic_scene.objects for {os.path.basename(glb_path)}: {ex_obj}. Type: {type(sim_gibson.semantic_scene)}", exc_info=True)
                else:
                    logger.warning(f"Gibson: sim.semantic_scene is None for {os.path.basename(glb_path)}.")
            else:
                logger.warning(f"Gibson: robust_load_sim returned None for {os.path.basename(glb_path)}.")
            
            # Fallback to PLY parsing if sim didn't yield objects or failed
            if not obj_id_to_cat and semantic_file_path_or_info and os.path.isfile(semantic_file_path_or_info):
                logger.info(f"Gibson: No objects from sim, trying to parse _semantic.ply: {semantic_file_path_or_info}")
                # ... (original PLY parsing logic for Gibson with error handling) ...
        except Exception as e:
            logger.error(f"Error processing Gibson scene {os.path.basename(glb_path)}: {e}", exc_info=True)
        finally:
            if sim_gibson is not None:
                sim_gibson.close()
        logger.info(f"Gibson: Loaded {len(obj_id_to_cat)} object instance mappings for {os.path.basename(glb_path)}.")

    elif worker_active_dataset == "mp3d":
        scn_path = semantic_file_path_or_info
        if os.path.isfile(scn_path):
            try:
                with open(scn_path) as fp: scn_data = json.load(fp)
                # Check if "objects" key exists before iterating
                if "objects" in scn_data and isinstance(scn_data["objects"], list):
                    for obj in scn_data["objects"]:
                        if obj.get("class_") in CURRENT_OBJECT_CATEGORY_MAP and \
                           obj["class_"] not in ["floor", "wall", "out-of-bounds", "ceiling"]:
                            obj_id_to_cat[obj["id"]] = obj["class_"]
                else:
                    logger.warning(f"MP3D .scn file {scn_path} does not contain an 'objects' list key.")
            except json.JSONDecodeError:
                logger.error(f"MP3D: Failed to decode JSON from .scn file: {scn_path}", exc_info=True)
            except Exception as e_scn:
                logger.error(f"MP3D: Error processing .scn file {scn_path}: {e_scn}", exc_info=True)
        else:
            logger.warning(f"MP3D .scn file not found: {scn_path}. Consider sim fallback if needed.")
        logger.info(f"MP3D: Loaded {len(obj_id_to_cat)} object instance mappings from {os.path.basename(scn_path if scn_path else 'N/A')}.")


    vertices = []
    colors = []
    obj_ids_out = []
    sem_ids_out = []

    if worker_active_dataset == "hm3d":
        logger.debug(f"HM3D: Extracting object meshes from GLB: {os.path.basename(glb_path)}")
        try:
            scene_trimesh_obj = trimesh.load(glb_path, force='scene', process=False)
            if not isinstance(scene_trimesh_obj, trimesh.Scene) or not hasattr(scene_trimesh_obj, 'graph'):
                logger.error(f"Could not load {os.path.basename(glb_path)} as a trimesh.Scene with a graph. Loaded as {type(scene_trimesh_obj)}. Cannot extract instances by node name.")
                scene_trimesh_obj = None

            if scene_trimesh_obj:
                for node_name in scene_trimesh_obj.graph.nodes_geometry:
                    transform, geometry_name = scene_trimesh_obj.graph[node_name]
                    
                    parsed_instance_id = -1
                    if node_name.isdigit(): 
                        parsed_instance_id = int(node_name)
                    else: 
                        patterns = [r'object_(\d+)', r'mesh_(\d+)', r'^(\d+)[-_a-zA-Z]*', r'[._-](\d+)$', r'instance_(\d+)']
                        for pat in patterns:
                            match = re.search(pat, node_name, re.IGNORECASE) 
                            if match:
                                try:
                                    parsed_instance_id = int(match.groups()[-1])
                                    break
                                except (ValueError, IndexError): continue
                    
                    if parsed_instance_id != -1 and parsed_instance_id in obj_id_to_cat:
                        poni_category_name = obj_id_to_cat[parsed_instance_id]
                        sem_id = CURRENT_OBJECT_CATEGORY_MAP[poni_category_name]
                        mesh_instance = scene_trimesh_obj.geometry[geometry_name]

                        if not isinstance(mesh_instance, trimesh.Trimesh) or not mesh_instance.vertices.shape[0] > 0:
                            continue
                        
                        mesh_instance_transformed = mesh_instance.copy()
                        mesh_instance_transformed.apply_transform(transform)
                        
                        if not hasattr(mesh_instance_transformed, 'triangles') or len(mesh_instance_transformed.triangles) == 0:
                            continue

                        t_pts = hab_utils.dense_sampling_trimesh(mesh_instance_transformed.triangles, sampling_density)
                        
                        obj_color_index = sem_id - 2 
                        color_for_obj = (0.5, 0.5, 0.5)
                        if obj_color_index >= 0 and obj_color_index < len(CURRENT_OBJECT_COLORS_VIS):
                            color_for_obj = CURRENT_OBJECT_COLORS_VIS[obj_color_index]
                        elif sem_id < len(LEGEND_PALETTE_VIS):
                             color_for_obj = LEGEND_PALETTE_VIS[sem_id]

                        for t_pt in t_pts:
                            vertices.append(t_pt); obj_ids_out.append(parsed_instance_id); sem_ids_out.append(sem_id); colors.append(list(color_for_obj))
        except Exception as e:
            logger.error(f"Error loading/processing HM3D GLB {os.path.basename(glb_path)} for object extraction: {e}", exc_info=True)
    # ... (Keep Gibson/MP3D object extraction)

    logger.info(f"Collected {len(vertices)} object vertices for {os.path.basename(glb_path)}.")

    try:
        hm3d_scene_dataset_config = None
        if worker_active_dataset == "hm3d": # Use worker_active_dataset
            scenes_root_parent_dir = os.path.abspath(os.path.join(SCENES_ROOT, ".."))
            potential_config_names = [
                "hm3d.scene_dataset_config.json",
                "hm3d_annotated_basis.scene_dataset_config.json",
                "hm3d_semantic.scene_dataset_config.json"
            ]
            for cfg_name in potential_config_names:
                potential_cfg_path = os.path.join(scenes_root_parent_dir, cfg_name)
                if os.path.isfile(potential_cfg_path):
                    hm3d_scene_dataset_config = potential_cfg_path
                    logger.info(f"Using HM3D scene dataset config: {hm3d_scene_dataset_config}")
                    break
            if not hm3d_scene_dataset_config:
                 logger.warning(f"HM3D scene_dataset_config.json not found in {scenes_root_parent_dir} with common names. Sim might use defaults or fail for complex scenes.")

        sim = hab_utils.robust_load_sim(glb_path, scene_dataset_config_file=hm3d_scene_dataset_config)
        if sim is None: 
            logger.error(f"Simulator could not be loaded for {os.path.basename(glb_path)}, cannot get navmesh.")
            raise RuntimeError(f"Sim load failed for {glb_path}")

        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        if navmesh_triangles.size > 0:
            t_pts_nav = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
            nav_sem_id = CURRENT_OBJECT_CATEGORY_MAP.get("floor", 0)
            nav_color_tuple = LEGEND_PALETTE_VIS[nav_sem_id] if nav_sem_id < len(LEGEND_PALETTE_VIS) else LEGEND_PALETTE_VIS[0]
            nav_color = [float(c) for c in nav_color_tuple]
            for t_pt in t_pts_nav:
                vertices.append(t_pt); obj_ids_out.append(-1); sem_ids_out.append(nav_sem_id); colors.append(nav_color)
            logger.debug(f"Collected {len(t_pts_nav)} navmesh vertices.")
        else:
            logger.warning(f"No navmesh vertices found for {os.path.basename(glb_path)}.")
        sim.close()
    except Exception as e:
        logger.error(f"Error processing navmesh for {os.path.basename(glb_path)}: {e}", exc_info=True)

    try:
        per_floor_wall_pc = extract_wall_point_clouds(
            glb_path, houses_dim_path, sampling_density=sampling_density
        )

        wall_sem_id = CURRENT_OBJECT_CATEGORY_MAP.get("wall", 1)
        wall_color_tuple = LEGEND_PALETTE_VIS[wall_sem_id] if wall_sem_id < len(LEGEND_PALETTE_VIS) else LEGEND_PALETTE_VIS[0]
        wall_color = [float(c) for c in wall_color_tuple]
        num_wall_pts = 0
        for _, points_on_floor in per_floor_wall_pc.items():
            if points_on_floor.shape[0] > 0:
                num_wall_pts += len(points_on_floor)
                for p_wall in points_on_floor:
                    vertices.append(p_wall); obj_ids_out.append(-1); sem_ids_out.append(wall_sem_id); colors.append(wall_color)
        logger.debug(f"Collected {num_wall_pts} wall vertices.")
    except Exception as e:
        logger.error(f"Error processing walls for {os.path.basename(glb_path)}: {e}", exc_info=True)

    if not vertices:
        logger.warning(f"No vertices extracted at all for scene {os.path.basename(glb_path)}. Skipping save of H5 point cloud.")
        return

    vertices_arr = np.array(vertices)
    obj_ids_arr = np.array(obj_ids_out)
    sem_ids_arr = np.array(sem_ids_out)
    colors_arr = np.array(colors)

    try:
        with h5py.File(pc_save_path, "w") as fp:
            fp.create_dataset("vertices", data=vertices_arr)
            fp.create_dataset("obj_ids", data=obj_ids_arr)
            fp.create_dataset("sem_ids", data=sem_ids_arr)
            fp.create_dataset("colors", data=colors_arr)
        logger.info(f"Saved point cloud for {os.path.basename(glb_path)} to {pc_save_path} with {len(vertices_arr)} points.")
    except Exception as e:
        logger.error(f"Failed to save HDF5 point cloud for {os.path.basename(glb_path)}: {e}", exc_info=True)

def _aux_fn(input_data_for_worker):
    actual_function_to_call = input_data_for_worker[0]
    args_for_actual_function = input_data_for_worker[1:]
    return actual_function_to_call(*args_for_actual_function)


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
        mask = np.zeros(floor_y.shape[0], dtype=np.bool_)
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
        if os.path.isfile(map_save_path):
            continue

        with open(os.path.join(houses_dim_root, env + ".json"), "r") as fp:
            houses_dim = json.load(fp)
        f = h5py.File(obj_f, "r")

        # Generate floor-wise maps
        per_floor_dims = {}
        for key, val in houses_dim.items():
            match = re.search(f"{env}_(\d+)", key)
            if match:
                per_floor_dims[int(match.group(1))] = val

        all_vertices = np.array(f["vertices"])
        all_obj_ids = np.array(f["obj_ids"])
        all_sem_ids = np.array(f["sem_ids"])
        all_colors = np.array(f["colors"])

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

        # Pre-assign objects to different floors
        per_floor_obj_ids = {floor_id: [] for floor_id in per_floor_dims.keys()}
        obj_ids_set = set(all_obj_ids.tolist())
        ## -1 corresponds to wall and floor
        if -1 in obj_ids_set:
            obj_ids_set.remove(-1)
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
                # Print message for debugging purposes
                print(
                    f"NOTE: Object id {obj_id} from scene {env} does not belong to any floor!"
                )
                continue
            per_floor_obj_ids[best_floor_id].append(obj_id)

        # Build maps per floor
        per_floor_maps = {}
        for floor_id, floor_dims in per_floor_dims.items():

            curr_floor_y = floor_dims["ylo"]
            if floor_id + 1 in per_floor_dims:
                next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y = math.inf

            # Get navigable and wall vertices based on height thresholds
            is_on_floor = (all_vertices[:, 1] >= curr_floor_y) & (
                all_vertices[:, 1] <= next_floor_y - 0.5
            )
            is_floor = (all_sem_ids == OBJECT_CATEGORY_MAP["floor"]) & is_on_floor
            is_wall = (all_sem_ids == OBJECT_CATEGORY_MAP["wall"]) & is_on_floor

            # Get object vertices based on height thresholds for individual object instances
            is_object = np.zeros_like(is_on_floor)
            for obj_id in per_floor_obj_ids[floor_id]:
                is_object = is_object | (all_obj_ids == obj_id)

            mask = is_floor | is_wall | is_object

            vertices = np.copy(all_vertices[mask])
            obj_ids = np.copy(all_obj_ids[mask])
            sem_ids = np.copy(all_sem_ids[mask])

            # -- some maps have 0 obj of interest
            if len(vertices) == 0:
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

            # Set the min_y for the floor. This will be used during episode generation to find
            # a random navigable start location.
            floor_mask = sem_ids == OBJECT_CATEGORY_MAP["floor"]
            min_y = vertices[floor_mask, 1].min()
            info[env][floor_id] = {"y_min": float(min_y.item())}

            # Reduce heights of floor and navigable space to ensure objects are taller.
            wall_mask = sem_ids == OBJECT_CATEGORY_MAP["wall"]
            vertices[wall_mask, 1] -= 0.5
            vertices[floor_mask, 1] -= 0.5

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

            # assert outside_map_indices.sum() == 0
            y_values = y_values[~outside_map_indices]
            vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
            vertex_to_map_x = vertex_to_map_x[~outside_map_indices]

            obj_ids = obj_ids[~outside_map_indices]
            sem_ids = sem_ids[~outside_map_indices]

            # -- get the z values for projection
            # -- shift to positive values
            y_values = y_values - min_y
            y_values += 1.0

            # -- projection
            feat_index = (
                world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x
            ).long()
            flat_highest_z = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_highest_z, argmax_flat_spatial_map = scatter_max(
                y_values,
                feat_index,
                dim=0,
                out=flat_highest_z,
            )
            # NOTE: This is needed only for torch_scatter>=2.3
            argmax_flat_spatial_map[argmax_flat_spatial_map == y_values.shape[0]] = -1

            m = argmax_flat_spatial_map >= 0
            flat_map_instance = (
                torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
            )

            flat_map_instance[m.view(-1)] = obj_ids[argmax_flat_spatial_map[m]]

            flat_map_semantic = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_map_semantic[m.view(-1)] = sem_ids[argmax_flat_spatial_map[m]]

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

            per_floor_maps[floor_id] = {
                "mask": mask,
                "map_z": map_z,
                "map_instance": map_instance,
                "map_semantic": map_semantic,
                "map_semantic_rgb": map_semantic_rgb,
            }

            rgb_save_path = os.path.join(save_dir, f"{env}_{floor_id}.png")
            cv2.imwrite(rgb_save_path, map_semantic_rgb)

        with h5py.File(map_save_path, "w") as f:
            f.create_dataset(f"wall_sem_id", data=OBJECT_CATEGORY_MAP["wall"])
            f.create_dataset(f"floor_sem_id", data=OBJECT_CATEGORY_MAP["floor"])
            f.create_dataset(
                f"out-of-bounds_sem_id", data=OBJECT_CATEGORY_MAP["out-of-bounds"]
            )
            for floor_id, floor_map in per_floor_maps.items():
                mask = floor_map["mask"]
                map_z = floor_map["map_z"]
                map_instance = floor_map["map_instance"]
                map_semantic = floor_map["map_semantic"]
                map_semantic_rgb = floor_map["map_semantic_rgb"]

                f.create_dataset(f"{floor_id}/mask", data=mask, dtype=bool)
                f.create_dataset(
                    f"{floor_id}/map_heights", data=map_z, dtype=np.float32
                )
                f.create_dataset(
                    f"{floor_id}/map_instance", data=map_instance, dtype=np.int32
                )
                f.create_dataset(
                    f"{floor_id}/map_semantic", data=map_semantic, dtype=np.int32
                )
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=map_semantic_rgb)

    json.dump(info, open(os.path.join(save_dir, "semmap_GT_info.json"), "w"))


if __name__ == "__main__":
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    os.makedirs(PC_SAVE_ROOT, exist_ok=True)
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)

    scene_paths_all_splits = []
    semantic_annotation_paths_all_splits = []

    if ACTIVE_DATASET == "hm3d":
        logger.info(f"Starting HM3D scene discovery in SCENES_ROOT: {SCENES_ROOT}")
        dataset_splits_to_process = SPLIT_SCENES.get(ACTIVE_DATASET, {})
        
        if not dataset_splits_to_process or \
           (not dataset_splits_to_process.get("train", []) and not dataset_splits_to_process.get("val", [])):
            logger.error(f"CRITICAL: No train/val scenes defined or lists are empty in poni/constants.py for HM3D under SPLIT_SCENES['hm3d']. Exiting.")
            logger.info(f"Content of SPLIT_SCENES['hm3d']: {dataset_splits_to_process}")
            try:
                actual_dirs = [d for d in os.listdir(SCENES_ROOT) if os.path.isdir(os.path.join(SCENES_ROOT, d))]
                logger.info(f"Actual directories found in {SCENES_ROOT}: {actual_dirs[:10]} (showing first 10)")
            except FileNotFoundError:
                logger.error(f"SCENES_ROOT directory {SCENES_ROOT} itself not found.")
            exit(1)

        for split_type in dataset_splits_to_process.keys(): 
            scenes_in_this_split = dataset_splits_to_process.get(split_type, [])
            if not scenes_in_this_split:
                logger.info(f"No scenes defined for HM3D {split_type} split in poni/constants.py.")
                continue
            
            logger.info(f"Looking for {len(scenes_in_this_split)} scenes in HM3D {split_type} split as defined in constants...")
            logger.info(f"Scene IDs from constants for {split_type}: {scenes_in_this_split[:5]} (first 5)")
            
            for scene_id_folder_name in scenes_in_this_split: 
                scene_folder_path = os.path.join(SCENES_ROOT, scene_id_folder_name)
                logger.debug(f"Checking scene folder: {scene_folder_path}")

                if not os.path.isdir(scene_folder_path):
                    logger.warning(f"Scene folder not found: {scene_folder_path} (using scene ID '{scene_id_folder_name}' from constants)")
                    continue

                short_scene_id_match = re.match(r'^\d+-(.*)$', scene_id_folder_name)
                if not short_scene_id_match:
                    logger.warning(f"Could not parse short_scene_id from folder name: {scene_id_folder_name}. Using full name for files.")
                    short_scene_id_for_filename = scene_id_folder_name
                else:
                    short_scene_id_for_filename = short_scene_id_match.group(1)
                
                logger.debug(f"Using short_scene_id '{short_scene_id_for_filename}' for files in folder '{scene_id_folder_name}'")

                glb_file_found = None
                potential_glb_filenames = [
                    f"{short_scene_id_for_filename}.semantic.glb", 
                    f"{short_scene_id_for_filename}.basis.glb",    
                    f"{short_scene_id_for_filename}.glb"
                ]
                for glb_name_candidate in potential_glb_filenames:
                    candidate_path = os.path.join(scene_folder_path, glb_name_candidate)
                    if os.path.isfile(candidate_path):
                        glb_file_found = candidate_path
                        logger.debug(f"Found GLB for {scene_id_folder_name}: {glb_file_found}")
                        break
                
                semantic_txt_file = os.path.join(scene_folder_path, f"{short_scene_id_for_filename}.semantic.txt")

                if glb_file_found and os.path.isfile(semantic_txt_file):
                    scene_paths_all_splits.append(glb_file_found)
                    semantic_annotation_paths_all_splits.append(semantic_txt_file)
                else:
                    if not glb_file_found: logger.warning(f"HM3D GLB (any type using short_id '{short_scene_id_for_filename}') not found in {scene_folder_path}")
                    if not os.path.isfile(semantic_txt_file): logger.warning(f"HM3D semantic.txt (using short_id '{short_scene_id_for_filename}') not found: {semantic_txt_file}")
        
        unique_glb_paths = sorted(list(set(scene_paths_all_splits)))
        temp_semantic_map = {}
        for p_sem_txt in semantic_annotation_paths_all_splits:
            short_id_sem = os.path.basename(p_sem_txt).replace(".semantic.txt","")
            temp_semantic_map[short_id_sem] = p_sem_txt
        
        final_semantic_paths = []
        final_scene_paths = []

        for glb_p in unique_glb_paths:
            glb_basename = os.path.basename(glb_p)
            short_id_glb_match = re.match(r'^([a-zA-Z0-9]+)', glb_basename) 
            if short_id_glb_match:
                short_id_glb = short_id_glb_match.group(1)
                if short_id_glb in temp_semantic_map:
                    final_semantic_paths.append(temp_semantic_map[short_id_glb])
                    final_scene_paths.append(glb_p)
                else:
                    logger.warning(f"Alignment issue: Could not find matching semantic.txt for GLB: {glb_p} (extracted short_id: {short_id_glb}) using semantic map keys: {list(temp_semantic_map.keys())[:5]}")
            else:
                 logger.warning(f"Could not extract short_id from GLB filename: {glb_basename} to align with semantic.txt")

        scene_paths_all_splits = final_scene_paths
        semantic_annotation_paths_all_splits = final_semantic_paths

    elif ACTIVE_DATASET == "gibson":
        # (original Gibson path finding logic, adapted for SPLIT_SCENES)
        scene_paths_all_splits = sorted(glob.glob(os.path.join(SCENES_ROOT, "*.glb")))
        valid_scenes_for_split_gibson = []
        for split_type_gibson in SPLIT_SCENES.get(ACTIVE_DATASET, {}).keys():
            valid_scenes_for_split_gibson.extend(SPLIT_SCENES[ACTIVE_DATASET].get(split_type_gibson, []))
        if valid_scenes_for_split_gibson: 
            scene_paths_all_splits = list(
                filter(lambda x: os.path.basename(x).split(".")[0] in valid_scenes_for_split_gibson, scene_paths_all_splits)
            )
        semantic_annotation_paths_all_splits = [p.replace(".glb", "_semantic.ply") for p in scene_paths_all_splits]
        final_scene_paths = []
        final_semantic_paths = []
        for glb_p, sem_p in zip(scene_paths_all_splits, semantic_annotation_paths_all_splits):
            if os.path.isfile(sem_p):
                final_scene_paths.append(glb_p)
                final_semantic_paths.append(sem_p)
            else:
                logger.warning(f"Gibson semantic PLY not found: {sem_p} for GLB {glb_p}")
        scene_paths_all_splits = final_scene_paths
        semantic_annotation_paths_all_splits = final_semantic_paths


    elif ACTIVE_DATASET == "mp3d":
        # (original MP3D path finding logic, adapted for SPLIT_SCENES)
        scene_paths_all_splits = sorted(glob.glob(os.path.join(SCENES_ROOT, "**/*.glb"), recursive=True))
        scene_paths_all_splits = list(filter(lambda x: os.path.basename(x).split(".")[0] != "prefetch_test_scene", scene_paths_all_splits))
        valid_scenes_for_split_mp3d = []
        for split_type_mp3d in SPLIT_SCENES.get(ACTIVE_DATASET, {}).keys():
            valid_scenes_for_split_mp3d.extend(SPLIT_SCENES[ACTIVE_DATASET].get(split_type_mp3d, []))
        if valid_scenes_for_split_mp3d: 
             scene_paths_all_splits = list(
                filter(lambda x: os.path.basename(x).split(".")[0] in valid_scenes_for_split_mp3d, scene_paths_all_splits)
            )
        semantic_annotation_paths_all_splits = [p.replace(".glb", ".scn") for p in scene_paths_all_splits]
        final_scene_paths_mp3d = []
        final_semantic_paths_mp3d = []
        for glb_p, sem_p in zip(scene_paths_all_splits, semantic_annotation_paths_all_splits):
            if os.path.isfile(sem_p):
                final_scene_paths_mp3d.append(glb_p)
                final_semantic_paths_mp3d.append(sem_p)
            else:
                logger.warning(f"MP3D .scn not found: {sem_p} for GLB {glb_p}")
        scene_paths_all_splits = final_scene_paths_mp3d
        semantic_annotation_paths_all_splits = final_semantic_paths_mp3d


    logger.info(f"Found {len(scene_paths_all_splits)} scenes for {ACTIVE_DATASET} to process after filtering by splits and checking for semantic files.")
    if not scene_paths_all_splits:
        logger.error(f"CRITICAL: No scenes found to process for {ACTIVE_DATASET}. Check SCENES_ROOT ({SCENES_ROOT}), SPLIT_SCENES in poni/constants.py, and ensure semantic files exist and scene IDs match folder names. Exiting.")
        exit(1)

    # Stage 1: Extract scene_boundaries
    logger.info("===========> Stage 1: Extracting scene boundaries <===========")
    boundary_processing_list = [] 
    for scene_path in scene_paths_all_splits:
        scene_folder_name_for_boundary = os.path.basename(os.path.dirname(scene_path))
        save_path = os.path.join(SB_SAVE_ROOT, f"{scene_folder_name_for_boundary}.json")
        if not os.path.isfile(save_path):
            boundary_processing_list.append( (get_scene_boundaries, (scene_path, save_path)) ) # Pass as a tuple for _aux_fn
        else:
            logger.info(f"Boundary file for {scene_folder_name_for_boundary} already exists at {save_path}, skipping.")


    if boundary_processing_list:
        logger.info(f"Found {len(boundary_processing_list)} scenes for boundary extraction.")
        # Try 'spawn' context if 'forkserver' is problematic or unavailable, 'fork' is least safe with CUDA.
        start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        logger.info(f"Using multiprocessing context for boundaries: {start_method}")
        try:
            ctx = mp.get_context(start_method)
            with ctx.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
                list(tqdm.tqdm(pool.imap_unordered(_aux_fn, boundary_processing_list), total=len(boundary_processing_list), desc="Scene Boundaries"))
        except Exception as e_pool:
            logger.error(f"Multiprocessing pool for Scene Boundaries failed: {e_pool}", exc_info=True)
            logger.info("Attempting serial execution for Scene Boundaries as a fallback...")
            for item in tqdm.tqdm(boundary_processing_list, desc="Scene Boundaries (Serial Fallback)"):
                try:
                    _aux_fn(item)
                except Exception as e_item:
                    logger.error(f"Error processing item {item[1]} serially: {e_item}", exc_info=True)

    else:
        logger.info("All scene boundaries seem to be precomputed or no new scenes found for boundary extraction.")

    # Stage 2: Generate point-clouds for each scene
    logger.info("===========> Stage 2: Extracting point-clouds <===========")
    pc_inputs = []
    for i, scene_path_pc in enumerate(scene_paths_all_splits):
        scene_folder_name_for_pc = os.path.basename(os.path.dirname(scene_path_pc))
        pc_save_path_val = os.path.join(PC_SAVE_ROOT, f"{scene_folder_name_for_pc}.h5")
        houses_dim_path_val = os.path.join(SB_SAVE_ROOT, f"{scene_folder_name_for_pc}.json")
        
        if i >= len(semantic_annotation_paths_all_splits):
            logger.error(f"Mismatch between scene_paths and semantic_annotation_paths at index {i}. Skipping PC for {scene_folder_name_for_pc}")
            continue
        semantic_file_for_scene = semantic_annotation_paths_all_splits[i] 

        if not os.path.isfile(houses_dim_path_val):
            logger.warning(f"Skipping PC for {scene_folder_name_for_pc}, boundary file missing: {houses_dim_path_val}")
            continue
        if not os.path.isfile(pc_save_path_val):
            pc_inputs.append(
                ( 
                    extract_scene_point_clouds, 
                    scene_path_pc,               
                    semantic_file_for_scene,    
                    houses_dim_path_val,        
                    pc_save_path_val,           
                    SAMPLING_RESOLUTION         
                )
            )
        else:
            logger.info(f"Point cloud for {scene_folder_name_for_pc} already exists at {pc_save_path_val}, skipping.")

    if pc_inputs:
        logger.info(f"Found {len(pc_inputs)} scenes for point cloud extraction.")
        effective_num_workers_pc = min(NUM_WORKERS, len(pc_inputs))
        if effective_num_workers_pc > 0:
            start_method_pc = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            logger.info(f"Using multiprocessing context for point clouds: {start_method_pc}")
            try:
                ctx_pc = mp.get_context(start_method_pc)
                with ctx_pc.Pool(effective_num_workers_pc, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
                    list(tqdm.tqdm(pool.imap_unordered(_aux_fn, pc_inputs), total=len(pc_inputs), desc="Point Clouds"))
            except Exception as e_pool_pc:
                logger.error(f"Multiprocessing pool for Point Clouds failed: {e_pool_pc}", exc_info=True)
                logger.info("Attempting serial execution for Point Clouds as a fallback...")
                for item_pc in tqdm.tqdm(pc_inputs, desc="Point Clouds (Serial Fallback)"):
                    try:
                        _aux_fn(item_pc)
                    except Exception as e_item_pc:
                        logger.error(f"Error processing item {item_pc[1]} serially: {e_item_pc}", exc_info=True)
        else:
            logger.info("No scenes to process for point cloud extraction after filtering.")
    else:
        logger.info("All point clouds seem to be precomputed or no new scenes found for point cloud extraction.")

    # Stage 3: Extract semantic maps from point clouds
    logger.info("===========> Stage 3: Extracting semantic maps <===========")
    convert_point_cloud_to_semantic_map(
        PC_SAVE_ROOT, SB_SAVE_ROOT, SEM_SAVE_ROOT,
        resolution=0.05
    )
    logger.info("===========> Semantic map creation process finished. <===========")

