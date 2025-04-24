#!/usr/bin/env python3
"""
Improved script to preprocess HM3D dataset files for use with PONI.
This script adapts the HM3D format to be compatible with the existing PONI pipeline.
"""

import os
import sys
import argparse
import json
import traceback
import gc
import numpy as np
import h5py
import time
import shutil
from datetime import datetime
from tqdm import tqdm

# Check if required packages are installed
try:
    import habitat_sim
    import trimesh
except ImportError as e:
    print(f"ERROR: {e}")
    print("Please make sure habitat_sim and trimesh are installed.")
    sys.exit(1)

# Import constants for visualization if available
try:
    from poni.constants import d3_40_colors_rgb
except ImportError:
    # Define fallback colors
    d3_40_colors_rgb = np.array([
        [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
        [44, 160, 44], [152, 223, 138], [206, 219, 156], [140, 109, 49],
        [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
        [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
        [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
    ], dtype=np.uint8)

# Define HM3D category mappings 
HM3D_CATEGORY_MAP = {
    "floor": 1,
    "wall": 2,
    "chair": 3,
    "table": 4,
    "bed": 5,
    "cabinet": 6,
    "sofa": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
}

def setup_parser():
    parser = argparse.ArgumentParser(description="Preprocess HM3D dataset for PONI")
    parser.add_argument(
        "--hm3d_path", 
        required=True, 
        help="Path to HM3D dataset root directory"
    )
    parser.add_argument(
        "--output_path", 
        default="data/scene_datasets/hm3d",
        help="Path to output processed HM3D files"
    )
    parser.add_argument(
        "--semantic_path", 
        default="data/semantic_maps/hm3d",
        help="Path to output semantic maps"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--limit", 
        type=int,
        default=0,
        help="Limit processing to N scenes (0=all)"
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="Process one scene at a time to avoid GPU memory issues"
    )
    return parser

def find_glb_files(hm3d_path):
    """Find all .glb files in the HM3D dataset."""
    glb_files = []
    for root, dirs, files in os.walk(hm3d_path):
        for file in files:
            if file.endswith(".basis.glb"):  # Specifically use .basis.glb files
                glb_files.append(os.path.join(root, file))
    
    print(f"Found {len(glb_files)} .glb files in {hm3d_path}")
    return glb_files

def make_configuration(scene_path, scene_dataset_config=None, radius=0.18, height=0.88):
    """Create a Habitat-Sim configuration for scene loading with renderer enabled."""
    
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    backend_cfg.create_renderer = True  # CRITICAL: Must be true for navmesh generation
    
    if scene_dataset_config is not None:
        backend_cfg.scene_dataset_config_file = scene_dataset_config

    # agent configuration
    # Add minimal sensor to satisfy renderer requirements
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = "depth"
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.resolution = [120, 160]
    depth_sensor_cfg.position = [0.0, height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = height
    agent_cfg.radius = radius
    agent_cfg.sensor_specifications = [depth_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    # Convert numpy float32 to regular floats before JSON serialization
def convert_to_json_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# When writing to JSON, add the default parameter
with open(output_file, 'w') as f:
    json.dump(boundaries, f, indent=2, default=convert_to_json_serializable)

   def generate_scene_boundaries(scene_path, output_dir, debug=False):
    # [existing code here...]
    
    try:
        # [existing code...]
        
        # Right before the code that saves to JSON, add:
        # Add this function to recursively convert numpy types to Python native types
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {convert_numpy_to_python(k): convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(i) for i in obj]
            return obj

        # Then use it before JSON serialization:
        boundaries = convert_numpy_to_python(boundaries)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(boundaries, f, indent=2)
        
        # [rest of existing code]
    
    except Exception as e:
        # [exception handling]

def safe_create_simulator(scene_path, debug=False):
    """Create a simulator instance with proper error handling."""
    try:
        cfg = make_configuration(scene_path)
        
        # Create the simulator
        sim = habitat_sim.Simulator(cfg)
        
        # Check if we need to compute the navmesh
        if not sim.pathfinder.is_loaded:
            if debug:
                print(f"Computing navmesh for {scene_path}")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        
        return sim
    except Exception as e:
        print(f"Error initializing simulator for {scene_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None

def safe_close_simulator(sim, debug=False):
    """Safely close the simulator to avoid GL context errors."""
    if sim is None:
        return
    
    try:
        # Explicitly remove references that might hold GL context
        sim.close()
        del sim
    except Exception as e:
        print(f"Error closing simulator: {str(e)}")
        if debug:
            traceback.print_exc()

def get_floor_heights(sim, sampling_resolution=0.10):
    """Get heights of different floors in a scene using navmesh points."""
    try:
        # Get all vertices from the navmesh
        navmesh_vertices = np.array(sim.pathfinder.get_topdown_view().shape)
        
        # Get y-coordinate heights and cluster them
        y_coords = np.array([v[1] for v in sim.pathfinder.build_navmesh_vertices()])
        
        # Simple approach: round to nearest 10cm and find clusters
        y_rounded = np.round(y_coords * 10) / 10
        unique_heights, counts = np.unique(y_rounded, return_counts=True)
        
        # Only keep heights with significant point counts
        significant_heights = []
        min_points = len(y_coords) * 0.05  # At least 5% of all points
        
        for height, count in zip(unique_heights, counts):
            if count > min_points:
                significant_heights.append(height)
        
        significant_heights = sorted(significant_heights)
        
        # Create floor data structures
        floor_extents = []
        for i, height in enumerate(significant_heights):
            # Define floor boundaries
            if i < len(significant_heights) - 1:
                next_height = significant_heights[i + 1]
                floor_bounds = (height - 0.5, (height + next_height) / 2 + 0.2)
            else:
                floor_bounds = (height - 0.5, height + 2.0)
            
            floor_extents.append({
                "min": float(floor_bounds[0]),
                "max": float(floor_bounds[1]),
                "mean": float(height)
            })
            
        return floor_extents
    
    except Exception as e:
        print(f"Error analyzing floor heights: {str(e)}")
        traceback.print_exc()
        return [{"min": 0.0, "max": 3.0, "mean": 0.88}]  # Default single floor

def generate_scene_boundaries(scene_path, output_dir, debug=False):
    """Extract scene boundaries and save them to a file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        scene_name = os.path.basename(scene_path).split('.')[0]
        
        # Check if output already exists to avoid reprocessing
        output_file = os.path.join(output_dir, f"{scene_name}.json")
        if os.path.exists(output_file):
            if debug:
                print(f"Skipping {scene_name}, output already exists at {output_file}")
            return output_file, True
        
        # Create a simulator instance for this scene
        sim = safe_create_simulator(scene_path, debug)
        if sim is None:
            print(f"Failed to process {scene_name}, simulator initialization failed")
            return None, False
        
        # Get scene bounds from pathfinder
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
        
        # Add buffer to bounds
        buffer = np.array([3.0, 0.0, 3.0])  # Buffer in XYZ space (meters)
        lower_bound_with_buffer = lower_bound - buffer
        upper_bound_with_buffer = upper_bound + buffer
        
        # Create main scene boundary info
        boundaries = {
            scene_name: {
                "xlo": float(lower_bound_with_buffer[0]),
                "ylo": float(lower_bound_with_buffer[1]),
                "zlo": float(lower_bound_with_buffer[2]),
                "xhi": float(upper_bound_with_buffer[0]),
                "yhi": float(upper_bound_with_buffer[1]),
                "zhi": float(upper_bound_with_buffer[2]),
                "center": [(lower_bound[i] + upper_bound[i]) / 2.0 for i in range(3)],
                "sizes": [upper_bound[i] - lower_bound[i] for i in range(3)],
                "map_world_shift": [lower_bound[0], lower_bound[1], lower_bound[2]],
                "resolution": 0.05  # 5cm default resolution
            }
        }
        
        # Get floor height information
        floor_extents = get_floor_heights(sim, sampling_resolution=0.1)
        
        # Add floor-specific boundaries
        for i, floor_data in enumerate(floor_extents):
            floor_name = f"{scene_name}_{i}"
            floor_lower = lower_bound.copy()
            floor_upper = upper_bound.copy()
            
            # Set y bounds for this floor
            floor_lower[1] = floor_data["min"] 
            floor_upper[1] = floor_data["max"]
            
            boundaries[floor_name] = {
                "xlo": float(floor_lower[0] - buffer[0]),
                "ylo": float(floor_lower[1]),
                "zlo": float(floor_lower[2] - buffer[2]),
                "xhi": float(floor_upper[0] + buffer[0]),
                "yhi": float(floor_upper[1]),
                "zhi": float(floor_upper[2] + buffer[2]),
                "center": [(floor_lower[i] + floor_upper[i]) / 2.0 for i in range(3)],
                "sizes": [floor_upper[i] - floor_lower[i] for i in range(3)],
                "y_min": floor_data["mean"]
            }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(boundaries, f, indent=2)
        
        # Clean up simulator
        safe_close_simulator(sim, debug)
        
        return output_file, True
    
    except Exception as e:
        print(f"Error processing {scene_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None, False

def extract_scene_point_clouds(scene_path, boundaries_path, pc_save_path, sampling_density=1600.0, debug=False):
    """Extract point cloud data from a scene."""
    try:
        os.makedirs(os.path.dirname(pc_save_path), exist_ok=True)
        scene_name = os.path.basename(scene_path).split('.')[0]
        
        # Check if output already exists
        if os.path.exists(pc_save_path):
            if debug:
                print(f"Skipping point cloud extraction for {scene_name}, output already exists")
            return pc_save_path, True
        
        # Load boundaries
        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)
        
        # Create a simulator instance for this scene
        sim = safe_create_simulator(scene_path, debug)
        if sim is None:
            print(f"Failed to process point cloud for {scene_name}, simulator initialization failed")
            return None, False
        
        # Extract vertices from navmesh
        vertices = []
        sem_ids = []
        obj_ids = []
        
        # Get navmesh vertices for floor
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        if debug:
            print(f"Found {len(navmesh_vertices)} navmesh vertices")
        
        # Add floor points
        floor_id = HM3D_CATEGORY_MAP["floor"]
        for vertex in navmesh_vertices:
            vertices.append(vertex)
            sem_ids.append(floor_id)
            obj_ids.append(-1)  # No specific object
            
        # Detect walls by sampling points near vertical surfaces
        wall_id = HM3D_CATEGORY_MAP["wall"]
        # Get scene bounds
        scene_info = boundaries[scene_name]
        scene_bounds = np.array([
            [scene_info["xlo"], scene_info["ylo"], scene_info["zlo"]],
            [scene_info["xhi"], scene_info["yhi"], scene_info["zhi"]]
        ])
        
        # Sample random points near walls (simplistic approach)
        num_wall_samples = 2000
        for _ in range(num_wall_samples):
            # Sample point along the perimeter
            wall_side = np.random.randint(0, 4)
            if wall_side == 0:  # front
                x = np.random.uniform(scene_bounds[0][0], scene_bounds[1][0])
                y = np.random.uniform(scene_bounds[0][1], scene_bounds[1][1])
                z = scene_bounds[0][2]
            elif wall_side == 1:  # back
                x = np.random.uniform(scene_bounds[0][0], scene_bounds[1][0])
                y = np.random.uniform(scene_bounds[0][1], scene_bounds[1][1])
                z = scene_bounds[1][2]
            elif wall_side == 2:  # left
                x = scene_bounds[0][0]
                y = np.random.uniform(scene_bounds[0][1], scene_bounds[1][1])
                z = np.random.uniform(scene_bounds[0][2], scene_bounds[1][2])
            else:  # right
                x = scene_bounds[1][0]
                y = np.random.uniform(scene_bounds[0][1], scene_bounds[1][1])
                z = np.random.uniform(scene_bounds[0][2], scene_bounds[1][2])
                
            vertices.append([x, y, z])
            sem_ids.append(wall_id)
            obj_ids.append(-1)
        
        # Convert to numpy arrays
        vertices = np.array(vertices)
        sem_ids = np.array(sem_ids)
        obj_ids = np.array(obj_ids)
        
        if len(vertices) == 0:
            print(f"Warning: No vertices found for {scene_name}")
            # Create dummy data
            vertices = np.zeros((10, 3))
            sem_ids = np.ones(10) * floor_id
            obj_ids = np.ones(10) * -1
            
        # Save to HDF5
        with h5py.File(pc_save_path, 'w') as fp:
            fp.create_dataset("vertices", data=vertices)
            fp.create_dataset("obj_ids", data=obj_ids)
            fp.create_dataset("sem_ids", data=sem_ids)
        
        # Clean up simulator
        safe_close_simulator(sim, debug)
        
        return pc_save_path, True
    
    except Exception as e:
        print(f"Error extracting point cloud from {scene_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None, False

def generate_basic_semantic_maps(pc_path, boundaries_path, output_path, resolution=0.05, debug=False):
    """Generate simple semantic maps from point clouds."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scene_name = os.path.basename(pc_path).split('.')[0]
        
        # Check if output already exists
        if os.path.exists(output_path):
            if debug:
                print(f"Skipping semantic map generation for {scene_name}, output already exists")
            return output_path, True
        
        # Load point cloud data
        with h5py.File(pc_path, 'r') as f:
            vertices = np.array(f["vertices"])
            sem_ids = np.array(f["sem_ids"])
            obj_ids = np.array(f["obj_ids"])
        
        # Load boundaries
        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)
        
        # Process each floor
        with h5py.File(output_path, 'w') as f:
            # Store semantic ID constants
            f.create_dataset(f"wall_sem_id", data=HM3D_CATEGORY_MAP["wall"])
            f.create_dataset(f"floor_sem_id", data=HM3D_CATEGORY_MAP["floor"])
            f.create_dataset(f"out-of-bounds_sem_id", data=0)
            
            for floor_key, floor_bounds in boundaries.items():
                if not floor_key.startswith(scene_name + "_"):
                    continue
                
                # Extract floor number
                floor_id = floor_key.split('_')[-1]
                
                # Filter points to this floor
                floor_min = floor_bounds.get("ylo", floor_bounds.get("y_min", 0) - 0.5)
                floor_max = floor_bounds.get("yhi", floor_bounds.get("y_min", 0) + 2.5)
                
                floor_mask = (vertices[:, 1] >= floor_min) & (vertices[:, 1] <= floor_max)
                floor_vertices = vertices[floor_mask]
                floor_sem_ids = sem_ids[floor_mask]
                floor_obj_ids = obj_ids[floor_mask]
                
                if len(floor_vertices) == 0:
                    print(f"No points found for floor {floor_key}")
                    # Create empty floor maps
                    map_size = 240  # 24 meters at 10cm resolution
                    mask = np.zeros((map_size, map_size), dtype=bool)
                    map_heights = np.zeros((map_size, map_size), dtype=np.float32)
                    map_instance = np.zeros((map_size, map_size), dtype=np.int32)
                    map_semantic = np.zeros((map_size, map_size), dtype=np.int32)
                    map_semantic_rgb = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                    
                    # Create empty floor group
                    floor_group = f.create_group(floor_id)
                    floor_group.create_dataset("mask", data=mask)
                    floor_group.create_dataset("map_heights", data=map_heights)
                    floor_group.create_dataset("map_instance", data=map_instance)
                    floor_group.create_dataset("map_semantic", data=map_semantic)
                    floor_group.create_dataset("map_semantic_rgb", data=map_semantic_rgb)
                    continue
                
                # Get scene coordinates
                world_center = np.array(floor_bounds["center"])
                world_size = np.array(floor_bounds["sizes"])
                
                # Create 2D top-down map
                # Map dimensions based on bounds
                map_size_x = int(world_size[0] / resolution) + 20  # Add padding
                map_size_z = int(world_size[2] / resolution) + 20
                
                # Initialize maps
                mask = np.zeros((map_size_z, map_size_x), dtype=bool)
                map_heights = np.zeros((map_size_z, map_size_x), dtype=np.float32)
                map_instance = np.zeros((map_size_z, map_size_x), dtype=np.int32)
                map_semantic = np.zeros((map_size_z, map_size_x), dtype=np.int32)
                
                # Project points to 2D grid
                for i, (vertex, sem_id, obj_id) in enumerate(zip(floor_vertices, floor_sem_ids, floor_obj_ids)):
                    # Get grid coordinates (top-down view)
                    grid_x = int((vertex[0] - (world_center[0] - world_size[0]/2)) / resolution)
                    grid_z = int((vertex[2] - (world_center[2] - world_size[2]/2)) / resolution)
                    
                    # Skip if outside grid
                    if grid_x < 0 or grid_x >= map_size_x or grid_z < 0 or grid_z >= map_size_z:
                        continue
                    
                    # Update maps
                    mask[grid_z, grid_x] = True
                    map_heights[grid_z, grid_x] = vertex[1]
                    map_instance[grid_z, grid_x] = obj_id
                    map_semantic[grid_z, grid_x] = sem_id
                
                # Create RGB visualization
                map_semantic_rgb = np.zeros((map_size_z, map_size_x, 3), dtype=np.uint8)
                
                # Color the map
                # Floor (light gray)
                floor_mask = map_semantic == HM3D_CATEGORY_MAP["floor"]
                map_semantic_rgb[floor_mask] = [230, 230, 230]
                
                # Wall (dark gray)
                wall_mask = map_semantic == HM3D_CATEGORY_MAP["wall"]
                map_semantic_rgb[wall_mask] = [77, 77, 77]
                
                # Create floor group in HDF5 file
                floor_group = f.create_group(floor_id)
                floor_group.create_dataset("mask", data=mask)
                floor_group.create_dataset("map_heights", data=map_heights)
                floor_group.create_dataset("map_instance", data=map_instance)
                floor_group.create_dataset("map_semantic", data=map_semantic)
                floor_group.create_dataset("map_semantic_rgb", data=map_semantic_rgb)
        
        # Create metadata for semantic maps
        metadata_file = os.path.join(os.path.dirname(output_path), "semmap_GT_info.json")
        
        # Load existing metadata if available
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Update metadata with scene information
        scene_info = boundaries[scene_name]
        metadata[scene_name] = {
            "dim": [
                int(scene_info["sizes"][0] / resolution),
                0,
                int(scene_info["sizes"][2] / resolution)
            ],
            "central_pos": scene_info["center"],
            "map_world_shift": scene_info.get("map_world_shift", [0, 0, 0]),
            "resolution": resolution
        }
        
        # Add floor-specific metadata
        for floor_key, floor_bounds in boundaries.items():
            if not floor_key.startswith(scene_name + "_"):
                continue
            
            floor_id = floor_key.split("_")[-1]
            metadata[scene_name][floor_id] = {
                "y_min": float(floor_bounds.get("y_min", floor_bounds.get("ylo", 0)))
            }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path, True
    
    except Exception as e:
        print(f"Error generating semantic maps for {pc_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None, False

def process_single_scene(scene_path, args):
    """Process a single scene with proper cleanup between scenes."""
    try:
        scene_name = os.path.basename(scene_path).split('.')[0]
        boundaries_dir = os.path.join(args.semantic_path, "scene_boundaries")
        pc_dir = os.path.join(args.semantic_path, "point_clouds")
        sem_map_dir = os.path.join(args.semantic_path, "semantic_maps")
        
        # Create output directories
        os.makedirs(boundaries_dir, exist_ok=True)
        os.makedirs(pc_dir, exist_ok=True)
        os.makedirs(sem_map_dir, exist_ok=True)
        
        # Output file paths
        boundaries_path = os.path.join(boundaries_dir, f"{scene_name}.json")
        pc_path = os.path.join(pc_dir, f"{scene_name}.h5")
        sem_map_path = os.path.join(sem_map_dir, f"{scene_name}.h5")
        
        # Step 1: Generate scene boundaries
        print(f"Processing scene boundaries for {scene_name}")
        boundaries_file, success = generate_scene_boundaries(
            scene_path, boundaries_dir, args.debug
        )
        
        if not success:
            print(f"Failed to process scene boundaries for {scene_name}")
            return False
        
        # Step 2: Extract point cloud
        print(f"Extracting point cloud for {scene_name}")
        pc_file, success = extract_scene_point_clouds(
            scene_path, boundaries_path, pc_path, debug=args.debug
        )
        
        if not success:
            print(f"Failed to extract point cloud for {scene_name}")
            return False
        
        # Step 3: Generate semantic maps
        print(f"Generating semantic maps for {scene_name}")
        sem_map_file, success = generate_basic_semantic_maps(
            pc_path, boundaries_path, sem_map_path, debug=args.debug
        )
        
        if not success:
            print(f"Failed to generate semantic maps for {scene_name}")
            return False
        
        print(f"Successfully processed {scene_name}")
        return True
    
    except Exception as e:
        print(f"Error processing scene {scene_path}: {str(e)}")
        if args.debug:
            traceback.print_exc()
        return False

def log_progress(successful, failed, num_total, start_time):
    """Log processing progress."""
    elapsed = time.time() - start_time
    success_rate = successful / max(1, (successful + failed))
    
    print(f"\n===== Processing Progress =====")
    print(f"Successful: {successful}/{num_total} ({successful/num_total*100:.1f}%)")
    print(f"Failed: {failed}/{num_total} ({failed/num_total*100:.1f}%)")
    print(f"Success rate: {success_rate*100:.1f}%")
    
    if successful > 0:
        time_per_scene = elapsed / successful
        remaining = (num_total - successful - failed) * time_per_scene
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Estimated time remaining: {remaining/60:.1f} minutes")
        print(f"Estimated completion: {datetime.now().strftime('%H:%M:%S')}")
    
    print("===============================\n")

def create_hm3d_split_metadata(scenes, output_dir):
    """Create dataset split metadata for HM3D scenes."""
    # Create train/val split (80/20)
    np.random.shuffle(scenes)
    split_idx = int(len(scenes) * 0.8)
    
    splits = {
        "train": scenes[:split_idx],
        "val": scenes[split_idx:]
    }
    
    # Save splits to JSON
    with open(os.path.join(output_dir, "hm3d_splits.json"), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created dataset splits: {len(splits['train'])} train scenes, {len(splits['val'])} val scenes")

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "scene_boundaries"), exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "point_clouds"), exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "semantic_maps"), exist_ok=True)
    
    # Find GLB files
    glb_files = find_glb_files(args.hm3d_path)
    
    # Process only a subset if requested
    if args.limit > 0:
        glb_files = glb_files[:args.limit]
    
    # Process scenes
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Create a backup of PYTHONPATH before modifying
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    
    if args.single_process:
        print("\nProcessing one scene at a time to avoid memory issues...")
        for i, scene_file in enumerate(glb_files):
            print(f"\nProcessing scene {i+1}/{len(glb_files)}: {os.path.basename(scene_file)}")
            
            # Reset PYTHONPATH for each scene to avoid memory leaks
            if 'PYTHONPATH' in os.environ:
                os.environ['PYTHONPATH'] = original_pythonpath
            
            success = process_single_scene(scene_file, args)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Log progress every 5 scenes
            if (i+1) % 5 == 0 or i == len(glb_files) - 1:
                log_progress(successful, failed, len(glb_files), start_time)
            
            # Force clean up any remaining GL contexts
            # This is crucial to avoid the GL::Context crashes
            if 'habitat_sim' in sys.modules:
                del sys.modules['habitat_sim']
            gc.collect()
    else:
        # Process all scenes in parallel (less safe for GPU memory)
        # This approach is faster but more likely to encounter GL context errors
        print("\nProcessing all scenes (parallel approach)...")
        
        # Step 1: Generate scene boundaries
        print("\nGenerating scene boundaries:")
        scene_boundaries = {}
        
        for i, scene_file in enumerate(tqdm(glb_files, desc="Processing scene boundaries")):
            scene_name = os.path.basename(scene_file).split('.')[0]
            boundaries_path = os.path.join(args.semantic_path, "scene_boundaries", f"{scene_name}.json")
            
            # Skip if already processed
            if os.path.exists(boundaries_path) and not args.debug:
                successful += 1
                scene_boundaries[scene_name] = {
                    "scene_file": scene_file, 
                    "boundaries_path": boundaries_path
                }
                continue
            
            try:
                boundaries_file, success = generate_scene_boundaries(
                    scene_file, 
                    os.path.join(args.semantic_path, "scene_boundaries"),
                    args.debug
                )
                
                if success:
                    successful += 1
                    scene_boundaries[scene_name] = {
                        "scene_file": scene_file, 
                        "boundaries_path": boundaries_file
                    }
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing scene boundaries for {scene_name}: {e}")
                if args.debug:
                    traceback.print_exc()
                failed += 1
            
            # Force cleanup
            gc.collect()
            
        # Step 2: Extract point clouds
        print("\nExtracting point clouds:")
        successful = 0
        failed = 0
        
        for scene_name, scene_info in tqdm(scene_boundaries.items(), desc="Extracting point clouds"):
            if "boundaries_path" not in scene_info:
                failed += 1
                continue
                
            pc_path = os.path.join(args.semantic_path, "point_clouds", f"{scene_name}.h5")
            
            # Skip if already processed
            if os.path.exists(pc_path) and not args.debug:
                successful += 1
                scene_boundaries[scene_name]["pc_path"] = pc_path
                continue
                
            try:
                pc_file, success = extract_scene_point_clouds(
                    scene_info["scene_file"],
                    scene_info["boundaries_path"],
                    os.path.join(args.semantic_path, "point_clouds", f"{scene_name}.h5"),
                    debug=args.debug
                )
                
                if success:
                    successful += 1
                    scene_boundaries[scene_name]["pc_path"] = pc_file
                else:
                    failed += 1
            except Exception as e:
                print(f"Error extracting point cloud for {scene_name}: {e}")
                if args.debug:
                    traceback.print_exc()
                failed += 1
            
            # Force cleanup
            gc.collect()
        
        # Step 3: Generate semantic maps
        print("\nGenerating semantic maps:")
        successful = 0
        failed = 0
        
        for scene_name, scene_info in tqdm(scene_boundaries.items(), desc="Generating semantic maps"):
            if "pc_path" not in scene_info or "boundaries_path" not in scene_info:
                failed += 1
                continue
                
            sem_map_path = os.path.join(args.semantic_path, "semantic_maps", f"{scene_name}.h5")
            
            # Skip if already processed
            if os.path.exists(sem_map_path) and not args.debug:
                successful += 1
                continue
                
            try:
                sem_map_file, success = generate_basic_semantic_maps(
                    scene_info["pc_path"],
                    scene_info["boundaries_path"],
                    os.path.join(args.semantic_path, "semantic_maps", f"{scene_name}.h5"),
                    resolution=0.05,
                    debug=args.debug
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error generating semantic map for {scene_name}: {e}")
                if args.debug:
                    traceback.print_exc()
                failed += 1
    
    # Create dataset split information
    print("\nCreating HM3D split metadata...")
    processed_scenes = [os.path.basename(f).split('.')[0] for f in glb_files 
                      if os.path.exists(os.path.join(args.semantic_path, "semantic_maps", f"{os.path.basename(f).split('.')[0]}.h5"))]
    
    if processed_scenes:
        create_hm3d_split_metadata(processed_scenes, args.semantic_path)
    
    print("\nPreprocessing complete!")
    total_time = (time.time() - start_time) / 60.0
    print(f"Total processing time: {total_time:.1f} minutes")
    
    # Finalize: Create an updated constants.py file to include HM3D dataset
    try:
        print("\nUpdating PONI constants to include HM3D dataset...")
        # Check if constants_hm3d.py exists
        constants_hm3d_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "poni", "constants_hm3d.py")
        
        if os.path.exists(constants_hm3d_path):
            # Run the update_constants.py script
            update_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_constants.py")
            
            if os.path.exists(update_script_path):
                print(f"Running constants update script: {update_script_path}")
                import subprocess
                subprocess.run([sys.executable, update_script_path], check=True)
                print("Constants updated successfully!")
            else:
                print(f"Constants update script not found at {update_script_path}")
        else:
            print(f"HM3D constants file not found at {constants_hm3d_path}")
    except Exception as e:
        print(f"Error updating constants: {e}")
        if args.debug:
            traceback.print_exc()

if __name__ == "__main__":
    main()