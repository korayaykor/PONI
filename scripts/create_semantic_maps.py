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
    current_wall_thresh=None # current_wall_thresh eklendi
):
    if current_wall_thresh is None: # current_wall_thresh için varsayılan değer
        current_wall_thresh = WALL_THRESH
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
    # BEGIN FIX
    sim = hab_utils.robust_load_sim(glb_path)
    navmesh_triangles_flat = np.array(sim.pathfinder.build_navmesh_vertices())

    if navmesh_triangles_flat.size > 0 and navmesh_triangles_flat.size % 9 == 0:
        navmesh_triangles_reshaped = navmesh_triangles_flat.reshape(-1, 3, 3)
        navmesh_vertices = hab_utils.dense_sampling_trimesh(
            navmesh_triangles_reshaped, sampling_density
        )
    elif navmesh_triangles_flat.size == 0:
        logger.warning(f"No navmesh vertices found for {os.path.basename(glb_path)} in extract_wall_point_clouds.")
        navmesh_vertices = np.array([])
    else:
        logger.error(f"Navmesh vertex data size {navmesh_triangles_flat.size} is not divisible by 9 for {os.path.basename(glb_path)}. Cannot form triangles.")
        navmesh_vertices = np.array([])
    # END FIX
    sim.close()

    per_floor_xz_map = {}
    nav_points_per_floor = {}
    for floor_id, floor_dims in per_floor_dims.items():
        # BEGIN FIX for potential IndexError on navmesh_vertices
        if navmesh_vertices.ndim == 2 and navmesh_vertices.shape[0] > 0 and navmesh_vertices.shape[1] == 3:
            floor_nav_mask = (navmesh_vertices[:, 1] >= floor_dims["ylo"]) & \
                             (navmesh_vertices[:, 1] < floor_dims["yhi"])
            floor_navmesh_vertices_on_this_floor = navmesh_vertices[floor_nav_mask]
        else:
            logger.warning(f"Sampled navmesh_vertices is not valid for floor {floor_id} in {env}. Shape: {navmesh_vertices.shape if isinstance(navmesh_vertices, np.ndarray) else type(navmesh_vertices)}")
            floor_navmesh_vertices_on_this_floor = np.array([])
        # END FIX

        nav_points_per_floor[floor_id] = floor_navmesh_vertices_on_this_floor # Store the possibly empty array

        # Ensure subsequent code correctly handles an empty floor_navmesh_vertices_on_this_floor
        if floor_navmesh_vertices_on_this_floor.shape[0] > 0:
            floor_x = np.rint(floor_navmesh_vertices_on_this_floor[:, 0] / grid_size).astype(np.int32)
            floor_z = np.rint(floor_navmesh_vertices_on_this_floor[:, 2] / grid_size).astype(np.int32)
            floor_y_coords = floor_navmesh_vertices_on_this_floor[:, 1] # Rename to avoid conflict with loop var
            floor_xz_sets = set(zip(floor_x, floor_z))
            current_floor_xz_map = {} # Use a temporary dict for the current floor
            for x_coord, z_coord in floor_xz_sets: # Iterate using x_coord, z_coord
                mask = (floor_x == x_coord) & (floor_z == z_coord)
                current_floor_xz_map[(x_coord, z_coord)] = np.median(floor_y_coords[mask]) # Use floor_y_coords
            per_floor_xz_map[floor_id] = current_floor_xz_map
        else:
            per_floor_xz_map[floor_id] = {}


    # Get all mesh triangles in the scene
    scene = trimesh.load(glb_path)
    # It seems `scene.triangles` might not always be what's expected, ensure it's Nx3x3
    wall_pc_candidate_triangles = scene.triangles
    if wall_pc_candidate_triangles.ndim == 2 and wall_pc_candidate_triangles.shape[1] == 3: # If it's a list of vertices
        if wall_pc_candidate_triangles.shape[0] % 3 == 0:
             wall_pc_candidate_triangles = wall_pc_candidate_triangles.reshape(-1,3,3)
        else:
            logger.warning(f"Scene mesh for {glb_path} has {wall_pc_candidate_triangles.shape[0]} vertices, not divisible by 3 to form triangles. Skipping wall PC from this source.")
            wall_pc_candidate_triangles= np.array([])


    if wall_pc_candidate_triangles.ndim == 3 and wall_pc_candidate_triangles.shape[1:] == (3,3):
        wall_pc = hab_utils.dense_sampling_trimesh(wall_pc_candidate_triangles, sampling_density)
    else:
        logger.warning(f"Scene mesh for {glb_path} does not provide triangles in expected Nx3x3 format. Shape: {wall_pc_candidate_triangles.shape}. Attempting to sample vertices directly if it's a point cloud.")
        if isinstance(scene, trimesh.PointCloud) and scene.vertices.shape[0] > 0:
            wall_pc = scene.vertices
        elif isinstance(scene, trimesh.Trimesh) and scene.vertices.shape[0] > 0 and scene.faces.shape[0] == 0: # Vertices but no faces
             wall_pc = scene.vertices
        else:
            wall_pc = np.array([])


    # Convert coordinate systems
    if wall_pc.shape[0] > 0:
        wall_pc = np.stack([wall_pc[:, 0], wall_pc[:, 2], -wall_pc[:, 1]], axis=1)
    else:
        logger.warning(f"No wall points to process for {glb_path} after trimesh load and sampling.")

    ############################################################################
    # Assign wall points to floors
    ############################################################################
    per_floor_point_clouds = defaultdict(list)
    if wall_pc.shape[0] > 0: # Proceed only if there are wall points
        for floor_id, floor_dims in per_floor_dims.items():
            # Identify points belonging to this floor
            curr_floor_y_level = floor_dims["ylo"] # Use consistent naming
            if floor_id + 1 in per_floor_dims:
                next_floor_y_level = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y_level = math.inf # Use math.inf

            # Ensure wall_pc has points before masking
            if wall_pc.shape[0] == 0:
                per_floor_point_clouds[floor_id] = np.array([])
                continue

            floor_mask = (curr_floor_y_level <= wall_pc[:, 1]) & \
                         (wall_pc[:, 1] <= next_floor_y_level - 0.5)
            
            # Ensure floor_mask can be applied to wall_pc
            if floor_mask.shape[0] != wall_pc.shape[0]:
                logger.error(f"Shape mismatch: floor_mask ({floor_mask.shape}) vs wall_pc ({wall_pc.shape}) for floor {floor_id} in {env}. Skipping.")
                per_floor_point_clouds[floor_id] = np.array([])
                continue

            current_floor_pc = wall_pc[floor_mask, :] # Use current_floor_pc
            
            # Ensure current_floor_pc has points before further processing
            if current_floor_pc.shape[0] == 0:
                per_floor_point_clouds[floor_id] = np.array([])
                continue

            floor_xz_map = per_floor_xz_map.get(floor_id, {}) # Get the map for current floor_id

            # Decide whether each point is a wall point or not
            # Use current_floor_pc for these calculations
            floor_x_disc = np.around(current_floor_pc[:, 0] / grid_size).astype(np.int32)
            floor_z_disc = np.around(current_floor_pc[:, 2] / grid_size).astype(np.int32)
            floor_y_values = current_floor_pc[:, 1] # Use a different variable name
            
            height_mask = np.zeros(floor_y_values.shape[0], dtype=np.bool_) # Use height_mask
            for i, (x_disc, z_disc, y_val) in enumerate( # Use y_val
                zip(floor_x_disc, floor_z_disc, floor_y_values) # Use floor_y_values
            ):
                # floor_y_level_from_map = per_floor_dims[floor_id]["ylo"] # This was just ylo, not from xz_map
                floor_y_level_from_map = floor_xz_map.get((x_disc, z_disc), per_floor_dims[floor_id]["ylo"])

                # Add point if within height thresholds
                # Use current_wall_thresh which was passed or defaulted from WALL_THRESH
                if current_wall_thresh[0] <= y_val - floor_y_level_from_map < current_wall_thresh[1]:
                    height_mask[i] = True # Use height_mask
            per_floor_point_clouds[floor_id] = current_floor_pc[height_mask] # Use height_mask
    else: # wall_pc was empty
        for floor_id in per_floor_dims.keys():
            per_floor_point_clouds[floor_id] = np.array([])


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
    if not obj_files:
        logger.warning(f"No .h5 files found in point cloud directory: {pc_dir}")
        # Initialize info here if it's used later, even if no files are processed
        info = {} 
        json.dump(info, open(os.path.join(save_dir, "semmap_GT_info.json"), "w"))
        return


    info = {}

    for obj_f in tqdm.tqdm(obj_files):

        env = obj_f.split("/")[-1].split(".")[0]
        map_save_path = os.path.join(save_dir, env + ".h5")
        # if os.path.isfile(map_save_path): # If you need to reprocess, comment this out
        #     logger.info(f"Semantic map for {env} already exists at {map_save_path}, skipping.")
        #     # Load existing info if needed for the final JSON dump, or ensure keys are handled
        #     # This part might need adjustment if you want to merge with existing info.json
        #     try:
        #         with open(os.path.join(save_dir, "semmap_GT_info.json"), "r") as f_info_existing:
        #             existing_info_all = json.load(f_info_existing)
        #             if env in existing_info_all:
        #                  info[env] = existing_info_all[env] # Preserve existing info for skipped files
        #     except FileNotFoundError:
        #         pass # No existing info file, it's fine
        #     except json.JSONDecodeError:
        #         logger.warning(f"Could not parse existing semmap_GT_info.json, will overwrite.")
        #     continue


        houses_dim_json_path = os.path.join(houses_dim_root, env + ".json")
        if not os.path.exists(houses_dim_json_path):
            logger.warning(f"Dimension file {houses_dim_json_path} not found for scene {env}. Skipping semantic map creation for this scene.")
            info[env] = {"error": f"Dimension JSON file not found at {houses_dim_json_path}"}
            continue

        try:
            with open(houses_dim_json_path, "r") as fp:
                houses_dim = json.load(fp)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {houses_dim_json_path} for scene {env}. Skipping.")
            info[env] = {"error": f"Could not decode dimension JSON file {houses_dim_json_path}"}
            continue


        # Check if the main environment key exists in houses_dim
        if env not in houses_dim:
            logger.error(f"Key '{env}' (overall scene bounds) not found in dimension file {houses_dim_json_path}. Scene: {env}. This scene might have had issues during boundary extraction (Stage 1). Skipping semantic map creation for this scene.")
            info[env] = {"error": f"Key '{env}' for overall scene bounds not found in its dimension JSON file."}
            continue # Skip to the next scene

        # --  set discret dimensions
        try:
            # This is where the original KeyError occurred
            center = np.array(houses_dim[env]["center"])
            sizes = np.array(houses_dim[env]["sizes"])
        except KeyError as e:
            logger.error(f"Missing 'center' or 'sizes' for key '{env}' in {houses_dim_json_path}. Content for key '{env}': {houses_dim.get(env)}. Error: {e}. Skipping scene {env}.")
            info[env] = {"error": f"Missing 'center' or 'sizes' for key '{env}' in its dimension JSON file."}
            continue
        except TypeError as e: # Handles if houses_dim[env] is not a dict (e.g. None or other type)
            logger.error(f"Data for key '{env}' in {houses_dim_json_path} is not a dictionary or is malformed. Content: {houses_dim.get(env)}. Error: {e}. Skipping scene {env}.")
            info[env] = {"error": f"Data for key '{env}' in dimension JSON file is malformed."}
            continue


        f = h5py.File(obj_f, "r")

        # Generate floor-wise maps
        per_floor_dims = {}
        for key, val in houses_dim.items():
            # Match keys like "scene_id_0", "scene_id_1", etc. for floors
            # env here is the base scene_id (e.g., "00006-HkseAnWCgqk")
            match = re.search(f"^{re.escape(env)}_(\d+)$", key) # Ensure it matches keys like "env_0", "env_1"
            if match:
                floor_num = int(match.group(1))
                # Ensure 'val' is a dictionary and contains 'ylo' before using it
                if isinstance(val, dict) and "ylo" in val:
                    per_floor_dims[floor_num] = val
                else:
                    logger.warning(f"Floor key '{key}' in {houses_dim_json_path} does not have expected dictionary structure or 'ylo'. Skipping this floor entry.")


        all_vertices = np.array(f["vertices"])
        all_obj_ids = np.array(f["obj_ids"])
        all_sem_ids = np.array(f["sem_ids"])
        # all_colors = np.array(f["colors"]) # Not used in this function currently

        f.close()

        # -- change coordinates to match map
        # --  set discret dimensions

        world_dim = sizes.copy()
        # world_dim[1] = 0 # Y dimension (height) of the overall box is in sizes[1]

        central_pos = center.copy()
        # central_pos[1] = 0 # Y coordinate of the center is center[1]

        map_world_shift = central_pos - world_dim / 2.0 # Element-wise subtraction

        world_dim_discret = [
            int(np.round(world_dim[0] / resolution)), # X size
            0, # Placeholder for Y, as map is 2D (X,Z) projection
            int(np.round(world_dim[2] / resolution)), # Z size
        ]

        current_scene_info = { # Store info for the current scene under its 'env' key
            "dim": world_dim_discret,
            "central_pos": [float(c) for c in central_pos],
            "map_world_shift": [float(s) for s in map_world_shift],
            "resolution": resolution,
        }


        # Pre-assign objects to different floors
        per_floor_obj_ids = {floor_id: [] for floor_id in per_floor_dims.keys()}
        obj_ids_set = set(all_obj_ids.tolist())
        ## -1 corresponds to wall and floor
        if -1 in obj_ids_set:
            obj_ids_set.remove(-1)
        
        for obj_id_val in obj_ids_set: # Renamed to avoid conflict
            is_obj_id = all_obj_ids == obj_id_val
            if not np.any(is_obj_id): continue # Should not happen if obj_id_val is from obj_ids_set

            obj_vertices = all_vertices[is_obj_id, :]
            if obj_vertices.shape[0] == 0: continue

            min_y = obj_vertices[:, 1].min()
            best_floor_id = None
            best_diff = math.inf
            for floor_id_loop, floor_dims_loop in per_floor_dims.items(): # Renamed loop vars
                # Ensure floor_dims_loop is a dict and has 'ylo'
                if not (isinstance(floor_dims_loop, dict) and "ylo" in floor_dims_loop):
                    logger.warning(f"Floor {floor_id_loop} in {env} has malformed dimensions: {floor_dims_loop}. Skipping object assignment for this floor.")
                    continue

                diff = abs(min_y - floor_dims_loop["ylo"])
                if (diff < best_diff) and min_y - floor_dims_loop["ylo"] > -0.5: # Object slightly below floor is ok
                    best_diff = diff
                    best_floor_id = floor_id_loop
            
            if best_floor_id is None:
                logger.warning(
                    f"NOTE: Object id {obj_id_val} from scene {env} (min_y: {min_y:.2f}) does not belong to any defined floor. Floors: {per_floor_dims}"
                )
                continue
            per_floor_obj_ids[best_floor_id].append(obj_id_val)

        # Build maps per floor
        per_floor_maps = {}
        for floor_id, floor_dims in per_floor_dims.items():
            # Ensure floor_dims is a dict and has 'ylo' (already checked for per_floor_obj_ids assignment, but good practice)
            if not (isinstance(floor_dims, dict) and "ylo" in floor_dims):
                logger.warning(f"Skipping floor {floor_id} for map generation in {env} due to malformed floor_dims: {floor_dims}")
                current_scene_info[floor_id] = {"error": "Malformed floor dimensions"}
                continue

            curr_floor_y = floor_dims["ylo"]
            if floor_id + 1 in per_floor_dims and isinstance(per_floor_dims[floor_id+1], dict) and "ylo" in per_floor_dims[floor_id+1] :
                next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y = math.inf

            # Get navigable and wall vertices based on height thresholds
            is_on_floor_height_wise = (all_vertices[:, 1] >= curr_floor_y) & \
                                     (all_vertices[:, 1] <= next_floor_y - 0.5)
            
            is_floor_semantic = (all_sem_ids == CURRENT_OBJECT_CATEGORY_MAP["floor"])
            is_wall_semantic = (all_sem_ids == CURRENT_OBJECT_CATEGORY_MAP["wall"])

            is_floor = is_floor_semantic & is_on_floor_height_wise
            is_wall = is_wall_semantic & is_on_floor_height_wise


            # Get object vertices based on height thresholds for individual object instances
            is_object = np.zeros_like(all_obj_ids, dtype=bool) # Initialize as boolean
            for obj_id_val_floor in per_floor_obj_ids.get(floor_id, []): # Use .get for safety
                is_object = is_object | (all_obj_ids == obj_id_val_floor)
            
            is_object = is_object & is_on_floor_height_wise # Objects also need to be within floor height range

            mask = is_floor | is_wall | is_object

            vertices_this_floor = np.copy(all_vertices[mask])
            obj_ids_this_floor = np.copy(all_obj_ids[mask])
            sem_ids_this_floor = np.copy(all_sem_ids[mask])


            # -- some maps have 0 obj of interest
            if len(vertices_this_floor) == 0:
                logger.warning(f"No vertices found for floor {floor_id} in scene {env} after filtering. Map will be empty.")
                current_scene_info[str(floor_id)] = {"y_min": float(curr_floor_y)} # Use curr_floor_y as fallback
                dims = (world_dim_discret[2], world_dim_discret[0]) # Z, X
                # ... (rest of the empty map creation logic)
                map_semantic_empty = np.zeros(dims, dtype=np.int32)
                per_floor_maps[floor_id] = {
                    "mask": np.zeros(dims, dtype=bool),
                    "map_z": np.zeros(dims, dtype=np.float32),
                    "map_instance": np.zeros(dims, dtype=np.int32),
                    "map_semantic": map_semantic_empty,
                    "map_semantic_rgb": visualize_sem_map(map_semantic_empty),
                }
                continue

            vertices_this_floor_shifted = vertices_this_floor - map_world_shift

            # Set the min_y for the floor. This will be used during episode generation to find
            # a random navigable start location.
            floor_mask_for_min_y = sem_ids_this_floor == CURRENT_OBJECT_CATEGORY_MAP["floor"]
            if np.any(floor_mask_for_min_y):
                 min_y_this_floor = vertices_this_floor_shifted[floor_mask_for_min_y, 1].min()
            else: # No floor points, use the nominal ylo for this floor
                 min_y_this_floor = curr_floor_y - map_world_shift[1] # Adjust by shift
            current_scene_info[str(floor_id)] = {"y_min": float(min_y_this_floor)}


            # Reduce heights of floor and navigable space to ensure objects are taller.
            wall_mask_on_floor = sem_ids_this_floor == CURRENT_OBJECT_CATEGORY_MAP["wall"]
            vertices_this_floor_shifted[wall_mask_on_floor, 1] -= 0.5 
            vertices_this_floor_shifted[floor_mask_for_min_y, 1] -= 0.5 


            # -- discretize point cloud
            vertices_torch = torch.FloatTensor(vertices_this_floor_shifted)
            obj_ids_torch = torch.FloatTensor(obj_ids_this_floor) # Keep as float for scatter_max if IDs can be non-int for some reason
            sem_ids_torch = torch.FloatTensor(sem_ids_this_floor)

            y_values_torch = vertices_torch[:, 1]

            vertex_to_map_x = (vertices_torch[:, 0] / resolution).round()
            vertex_to_map_z = (vertices_torch[:, 2] / resolution).round() # Z dimension is mapped to map rows

            # Boundary checks for map coordinates
            outside_map_indices = (
                (vertex_to_map_x >= world_dim_discret[0]) | # X-size of map
                (vertex_to_map_z >= world_dim_discret[2]) | # Z-size of map (which is map height)
                (vertex_to_map_x < 0) |
                (vertex_to_map_z < 0)
            )
            
            if torch.any(outside_map_indices):
                logger.debug(f"Scene {env}, floor {floor_id}: {outside_map_indices.sum().item()} points are outside map discret bounds. Clamping them.")
                # Clamp points to be within map boundaries to avoid scatter_max errors
                # This is a simple way to handle it; ideally, points exactly on far boundary might need care
                vertex_to_map_x = torch.clamp(vertex_to_map_x, 0, world_dim_discret[0] - 1)
                vertex_to_map_z = torch.clamp(vertex_to_map_z, 0, world_dim_discret[2] - 1)


            y_values_torch = y_values_torch[~outside_map_indices]
            vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
            vertex_to_map_x = vertex_to_map_x[~outside_map_indices]
            obj_ids_torch = obj_ids_torch[~outside_map_indices]
            sem_ids_torch = sem_ids_torch[~outside_map_indices]
            
            if y_values_torch.numel() == 0: # No points left after filtering
                logger.warning(f"No valid points left for floor {floor_id} in scene {env} after boundary check. Map will be empty.")
                current_scene_info[str(floor_id)]["warning"] = "No valid points after boundary check"
                dims = (world_dim_discret[2], world_dim_discret[0])
                map_semantic_empty = np.zeros(dims, dtype=np.int32)
                per_floor_maps[floor_id] = {
                    "mask": np.zeros(dims, dtype=bool),
                    "map_z": np.zeros(dims, dtype=np.float32),
                    "map_instance": np.zeros(dims, dtype=np.int32),
                    "map_semantic": map_semantic_empty,
                    "map_semantic_rgb": visualize_sem_map(map_semantic_empty),
                }
                continue


            # -- get the z values for projection
            # -- shift to positive values (relative to this floor's min_y, which is now 0 after y_values_torch -= min_y_torch)
            min_y_torch_val = y_values_torch.min() # min height of points on this floor, already shifted relative to map_world_shift
            y_values_torch_proj = y_values_torch - min_y_torch_val # Make smallest height on this slice effectively 0 for projection ranking
            y_values_torch_proj += 1.0 # Ensure all are >0 for scatter_max

            # -- projection
            # map is (map_height_Z, map_width_X)
            # world_dim_discret[2] is map height (from Z dimension)
            # world_dim_discret[0] is map width (from X dimension)
            feat_index = (
                world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x # row-major index from (z_map_coord, x_map_coord)
            ).long()

            flat_map_size = int(world_dim_discret[0] * world_dim_discret[2])
            
            flat_highest_z = torch.zeros(flat_map_size, device=vertices_torch.device)
            flat_highest_z, argmax_flat_spatial_map = scatter_max(
                y_values_torch_proj, # Use relative heights for projection ranking
                feat_index,
                dim=0,
                out=flat_highest_z,
            )
            
            argmax_flat_spatial_map[argmax_flat_spatial_map == y_values_torch_proj.shape[0]] = -1 # Handle out-of-bounds from scatter_max

            m = argmax_flat_spatial_map >= 0 # Mask for valid indices from scatter_max
            
            flat_map_instance = torch.full((flat_map_size,), -1, dtype=torch.float32, device=vertices_torch.device) # Default to -1
            if m.any():
                flat_map_instance[m.view(-1)] = obj_ids_torch[argmax_flat_spatial_map[m]]

            flat_map_semantic = torch.zeros(flat_map_size, dtype=torch.float32, device=vertices_torch.device) # Default to 0 (e.g. floor or out-of-bounds if not set)
            if m.any():
                flat_map_semantic[m.view(-1)] = sem_ids_torch[argmax_flat_spatial_map[m]]


            # -- format data
            map_shape_z_x = (world_dim_discret[2], world_dim_discret[0])

            mask_np = m.reshape(map_shape_z_x).cpu().numpy().astype(bool)
            map_z_np = flat_highest_z.reshape(map_shape_z_x).cpu().numpy().astype(np.float32)
            map_z_np[mask_np] += min_y_torch_val.cpu().numpy() # Add back the min_y offset to get original shifted heights
            map_z_np[~mask_np] = 0 # Or some other indicator for non-projected cells

            map_instance_np = flat_map_instance.reshape(map_shape_z_x).cpu().numpy().astype(np.int32)
            map_semantic_np = flat_map_semantic.reshape(map_shape_z_x).cpu().numpy().astype(np.int32)
            map_semantic_rgb_np = visualize_sem_map(map_semantic_np)

            per_floor_maps[floor_id] = {
                "mask": mask_np,
                "map_z": map_z_np,
                "map_instance": map_instance_np,
                "map_semantic": map_semantic_np,
                "map_semantic_rgb": map_semantic_rgb_np,
            }

            rgb_save_path = os.path.join(save_dir, f"{env}_{floor_id}.png")
            cv2.imwrite(rgb_save_path, map_semantic_rgb_np) # Use BGR for cv2
        
                                info[env] = current_scene_info # Save the collected info for this scene (env)

        with h5py.File(map_save_path, "w") as f_h5: # Use f_h5
            # Ensure "floor" and "wall" and "out-of-bounds" exist in CURRENT_OBJECT_CATEGORY_MAP
            f_h5.create_dataset(f"wall_sem_id", data=CURRENT_OBJECT_CATEGORY_MAP.get("wall", 1)) # Default to 1 if not found
            f_h5.create_dataset(f"floor_sem_id", data=CURRENT_OBJECT_CATEGORY_MAP.get("floor", 0)) # Default to 0
            f_h5.create_dataset(f"out-of-bounds_sem_id", data=CURRENT_OBJECT_CATEGORY_MAP.get("out-of-bounds", -1)) # Or another suitable default

            for floor_id_save, floor_map_save in per_floor_maps.items(): # Use new loop vars
                grp = f_h5.create_group(str(floor_id_save)) # Group by floor_id string
                grp.create_dataset("mask", data=floor_map_save["mask"], dtype=bool)
                grp.create_dataset("map_heights", data=floor_map_save["map_z"], dtype=np.float32)
                grp.create_dataset("map_instance", data=floor_map_save["map_instance"], dtype=np.int32)
                grp.create_dataset("map_semantic", data=floor_map_save["map_semantic"], dtype=np.int32)
                grp.create_dataset("map_semantic_rgb", data=floor_map_save["map_semantic_rgb"])


    json_info_path = os.path.join(save_dir, "semmap_GT_info.json")
    try:
        with open(json_info_path, "w") as fp_json:
             json.dump(info, fp_json, indent=4)
        logger.info(f"Successfully wrote semantic map GT info to {json_info_path}")
    except Exception as e_json:
        logger.error(f"Failed to write semantic map GT info to {json_info_path}: {e_json}", exc_info=True)


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

