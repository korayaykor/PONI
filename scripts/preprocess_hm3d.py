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
import numpy as np
import h5py
import time
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

# Define HM3D category mappings (simplified for now)
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

def initialize_simulator_for_scene(scene_path, debug=False):
    """Initialize the habitat simulator for a given scene with proper configuration."""
    try:
        # Create simulator config with rendering and physics explicitly set
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False
        backend_cfg.create_renderer = True  # Ensure renderer is created
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.88
        agent_cfg.radius = 0.18
        agent_cfg.sensor_specifications = []  # No sensors needed
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        # Initialize simulator
        sim = habitat_sim.Simulator(cfg)
        
        # Check if navmesh needs to be computed
        if not sim.pathfinder.is_loaded:
            print(f"Warning: Navmesh is not loaded for {scene_path}. Computing a new navmesh...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        
        return sim
    except Exception as e:
        if debug:
            print(f"Error initializing simulator for {scene_path}: {e}")
            traceback.print_exc()
        return None

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
        
        # Initialize simulator with proper configuration for this scene
        sim = initialize_simulator_for_scene(scene_path, debug)
        if sim is None:
            print(f"Failed to process {scene_name}, simulator initialization failed")
            return None, False
        
        # Get scene bounds
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
        
        # Convert to the format expected by PONI
        buffer = np.array([3.0, 0.0, 3.0])  # Buffer in XYZ space
        lower_bound_with_buffer = lower_bound - buffer
        upper_bound_with_buffer = upper_bound + buffer
        
        # Create boundaries dictionary
        boundaries = {
            scene_name: {
                "xlo": float(lower_bound_with_buffer[0]),
                "ylo": float(lower_bound_with_buffer[1]),
                "zlo": float(lower_bound_with_buffer[2]),
                "xhi": float(upper_bound_with_buffer[0]),
                "yhi": float(upper_bound_with_buffer[1]),
                "zhi": float(upper_bound_with_buffer[2]),
                "center": [(lower_bound[i] + upper_bound[i]) / 2.0 for i in range(3)],
                "sizes": [upper_bound[i] - lower_bound[i] for i in range(3)]
            }
        }
        
        # Check for floors using navmesh vertices
        try:
            navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
            y_coords = navmesh_vertices[:, 1]
            
            # Simple floor detection using height clustering
            y_coords_rounded = np.round(y_coords * 10) / 10  # Round to nearest 0.1m
            unique_heights, counts = np.unique(y_coords_rounded, return_counts=True)
            
            # Filter out heights with few points (likely noise)
            significant_heights = []
            for height, count in zip(unique_heights, counts):
                if count > len(y_coords) * 0.05:  # At least 5% of all points
                    significant_heights.append(height)
            
            significant_heights = sorted(significant_heights)
            
            # Add floor boundaries
            for i, height in enumerate(significant_heights):
                floor_name = f"{scene_name}_{i}"
                floor_lower = lower_bound.copy()
                floor_upper = upper_bound.copy()
                
                # Set y bounds for this floor
                if i < len(significant_heights) - 1:
                    next_height = significant_heights[i + 1]
                    floor_bounds = (height - 0.5, (height + next_height) / 2 + 0.2)
                else:
                    floor_bounds = (height - 0.5, height + 2.0)
                
                floor_lower[1] = floor_bounds[0]
                floor_upper[1] = floor_bounds[1]
                
                boundaries[floor_name] = {
                    "xlo": float(floor_lower[0] - buffer[0]),
                    "ylo": float(floor_lower[1]),
                    "zlo": float(floor_lower[2] - buffer[2]),
                    "xhi": float(floor_upper[0] + buffer[0]),
                    "yhi": float(floor_upper[1]),
                    "zhi": float(floor_upper[2] + buffer[2]),
                    "center": [(floor_lower[i] + floor_upper[i]) / 2.0 for i in range(3)],
                    "sizes": [floor_upper[i] - floor_lower[i] for i in range(3)]
                }
        except Exception as e:
            if debug:
                print(f"Warning: Could not analyze floors for {scene_name}: {e}")
                traceback.print_exc()
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(boundaries, f, indent=2)
        
        # Clean up
        sim.close()
        
        return output_file, True
    except Exception as e:
        print(f"Error processing {scene_path}: {e}")
        if debug:
            traceback.print_exc()
        return None, False

def extract_point_cloud(scene_path, boundaries_path, output_path, debug=False):
    """Extract point cloud data from a scene."""
    try:
        os.makedirs(output_path, exist_ok=True)
        scene_name = os.path.basename(scene_path).split('.')[0]
        output_file = os.path.join(output_path, f"{scene_name}.h5")
        
        # Check if output already exists
        if os.path.exists(output_file):
            if debug:
                print(f"Skipping point cloud extraction for {scene_name}, output already exists")
            return output_file, True
        
        # Load boundaries
        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)
        
        # Load the scene mesh
        try:
            scene = trimesh.load(scene_path)
        except Exception as e:
            print(f"Error loading scene mesh: {e}")
            return None, False
        
        # Sample points from the scene mesh
        points = []
        vertices = np.array(scene.vertices)
        faces = np.array(scene.faces)
        
        # Sample points from each face
        points_per_face = 10  # Number of points to sample per face
        for face in tqdm(faces, desc=f"Sampling points for {scene_name}", disable=not debug):
            v1, v2, v3 = vertices[face]
            # Generate barycentric coordinates
            u = np.random.rand(points_per_face, 1)
            v = np.random.rand(points_per_face, 1)
            mask = u + v > 1
            u[mask] = 1 - u[mask]
            v[mask] = 1 - v[mask]
            w = 1 - u - v
            
            # Generate points
            sampled_points = u * v1 + v * v2 + w * v3
            points.extend(sampled_points)
        
        points = np.array(points)
        
        # Classify points as floor, wall or other
        sem_ids = np.zeros(len(points), dtype=np.int32)
        obj_ids = np.zeros(len(points), dtype=np.int32)
        
        # Simple classification by height and orientation
        for i, point in enumerate(points):
            floor_threshold = 0.3  # meters above the floor
            scene_info = boundaries[scene_name]
            
            # Point is near the floor
            if abs(point[1] - scene_info["ylo"]) < floor_threshold:
                sem_ids[i] = HM3D_CATEGORY_MAP["floor"]
            # Point is near a vertical surface (wall)
            elif point[1] > scene_info["ylo"] + floor_threshold and point[1] < scene_info["yhi"] - floor_threshold:
                sem_ids[i] = HM3D_CATEGORY_MAP["wall"]
            # Otherwise just count it as an "unknown" object
            else:
                sem_ids[i] = 0  # Out of bounds
        
        # Save to HDF5
        with h5py.File(output_file, 'w') as f:
            f.create_dataset("vertices", data=points)
            f.create_dataset("sem_ids", data=sem_ids)
            f.create_dataset("obj_ids", data=obj_ids)
        
        return output_file, True
    except Exception as e:
        print(f"Error extracting point cloud from {scene_path}: {e}")
        if debug:
            traceback.print_exc()
        return None, False

def generate_semantic_maps(pc_path, boundaries_path, output_path, resolution=0.05, debug=False):
    """Generate semantic maps from point clouds."""
    try:
        os.makedirs(output_path, exist_ok=True)
        scene_name = os.path.basename(pc_path).split('.')[0]
        output_file = os.path.join(output_path, f"{scene_name}.h5")
        
        # Check if output already exists
        if os.path.exists(output_file):
            if debug:
                print(f"Skipping semantic map generation for {scene_name}, output already exists")
            return output_file, True
        
        # Load point cloud data
        with h5py.File(pc_path, 'r') as f:
            vertices = np.array(f["vertices"])
            sem_ids = np.array(f["sem_ids"])
            obj_ids = np.array(f["obj_ids"])
        
        # Load boundaries
        with open(boundaries_path, 'r') as f:
            boundaries = json.load(f)
        
        # Process each floor
        for floor_key, floor_bounds in boundaries.items():
            if not floor_key.startswith(scene_name + "_"):
                continue
            
            # Filter points to this floor
            floor_mask = (vertices[:, 1] >= floor_bounds["ylo"]) & (vertices[:, 1] <= floor_bounds["yhi"])
            floor_vertices = vertices[floor_mask]
            floor_sem_ids = sem_ids[floor_mask]
            floor_obj_ids = obj_ids[floor_mask]
            
            if len(floor_vertices) == 0:
                print(f"No points found for floor {floor_key}")
                continue
            
            # Project to 2D grid
            world_dim = np.array([floor_bounds["sizes"][0], 0, floor_bounds["sizes"][2]])
            world_dim += 2  # Add padding
            
            central_pos = np.array(floor_bounds["center"])
            central_pos[1] = 0  # Set y to 0
            
            map_world_shift = central_pos - world_dim / 2
            
            world_dim_discret = [
                int(np.round(world_dim[0] / resolution)),
                0,
                int(np.round(world_dim[2] / resolution))
            ]
            
            # Convert vertices to map coordinates
            map_vertices = floor_vertices.copy()
            map_vertices -= map_world_shift
            
            # Discretize to grid
            grid_x = np.round(map_vertices[:, 0] / resolution).astype(np.int32)
            grid_z = np.round(map_vertices[:, 2] / resolution).astype(np.int32)
            
            # Filter out points outside the grid
            valid_mask = (grid_x >= 0) & (grid_x < world_dim_discret[0]) & \
                         (grid_z >= 0) & (grid_z < world_dim_discret[2])
            
            grid_x = grid_x[valid_mask]
            grid_z = grid_z[valid_mask]
            floor_sem_ids = floor_sem_ids[valid_mask]
            floor_obj_ids = floor_obj_ids[valid_mask]
            heights = map_vertices[valid_mask, 1]
            
            # Create semantic map
            sem_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.int32)
            obj_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.int32)
            height_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.float32)
            
            # Fill maps using a voxel grid approach
            grid_indices = np.vstack([grid_z, grid_x]).T  # [N, 2]
            
            # For each point, find corresponding grid cell
            for i in range(len(grid_indices)):
                z, x = grid_indices[i]
                if z < world_dim_discret[2] and x < world_dim_discret[0]:
                    # Only update if point is higher than current height or cell is empty
                    if height_map[z, x] < heights[i] or height_map[z, x] == 0:
                        height_map[z, x] = heights[i]
                        sem_map[z, x] = floor_sem_ids[i]
                        obj_map[z, x] = floor_obj_ids[i]
            
            # Get floor ID
            floor_id = floor_key.split('_')[-1]
            
            # Create or update HDF5 file
            with h5py.File(output_file, 'a') as f:
                # Create floor group if it doesn't exist
                if floor_id not in f:
                    f.create_group(floor_id)
                
                # Create datasets
                mask = (sem_map > 0).astype(bool)
                f.create_dataset(f"{floor_id}/mask", data=mask)
                f.create_dataset(f"{floor_id}/map_heights", data=height_map)
                f.create_dataset(f"{floor_id}/map_instance", data=obj_map)
                f.create_dataset(f"{floor_id}/map_semantic", data=sem_map)
                
                # Create RGB visualization
                sem_rgb = np.zeros((world_dim_discret[2], world_dim_discret[0], 3), dtype=np.uint8)
                
                # Floor is light gray
                sem_rgb[sem_map == HM3D_CATEGORY_MAP["floor"]] = [230, 230, 230]
                # Wall is dark gray
                sem_rgb[sem_map == HM3D_CATEGORY_MAP["wall"]] = [77, 77, 77]
                # Other objects get colors based on their semantic ID
                for id in range(3, 20):  # Assuming up to 20 semantic categories
                    if np.any(sem_map == id):
                        # Use a color from a predefined palette
                        color_idx = (id - 3) % len(d3_40_colors_rgb)
                        sem_rgb[sem_map == id] = d3_40_colors_rgb[color_idx]
                
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=sem_rgb)
        
        # Save metadata
        metadata_file = os.path.join(output_path, "semmap_GT_info.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        for floor_key, floor_bounds in boundaries.items():
            if not floor_key.startswith(scene_name):
                continue
            
            world_dim = np.array([floor_bounds["sizes"][0], 0, floor_bounds["sizes"][2]])
            world_dim += 2  # Add padding
            
            central_pos = np.array(floor_bounds["center"])
            map_world_shift = central_pos - world_dim / 2
            
            world_dim_discret = [
                int(np.round(world_dim[0] / resolution)),
                0,
                int(np.round(world_dim[2] / resolution))
            ]
            
            if scene_name not in metadata:
                metadata[scene_name] = {
                    "dim": world_dim_discret,
                    "central_pos": central_pos.tolist(),
                    "map_world_shift": map_world_shift.tolist(),
                    "resolution": resolution
                }
            
            if floor_key.startswith(scene_name + "_"):
                floor_id = floor_key.split('_')[-1]
                metadata[scene_name][floor_id] = {
                    "y_min": float(floor_bounds["ylo"])
                }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_file, True
    except Exception as e:
        print(f"Error generating semantic maps for {pc_path}: {e}")
        if debug:
            traceback.print_exc()
        return None, False

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
    
    # Process each scene
    results = {
        "boundaries": {"success": 0, "failed": 0},
        "point_clouds": {"success": 0, "failed": 0},
        "semantic_maps": {"success": 0, "failed": 0}
    }
    
    scene_file_name_map = {}  # To keep track of processed file paths
    
    # Stage 1: Generate scene boundaries
    print("\nGenerating scene boundaries:")
    for scene_file in tqdm(glb_files, desc="Processing scene boundaries"):
        scene_name = os.path.basename(scene_file).split('.')[0]
        boundaries_path = os.path.join(args.semantic_path, "scene_boundaries", f"{scene_name}.json")
        
        # Skip if already processed
        if os.path.exists(boundaries_path) and not args.debug:
            results["boundaries"]["success"] += 1
            scene_file_name_map[scene_name] = {"scene_file": scene_file, "boundaries_path": boundaries_path}
            continue
        
        try:
            boundaries_file, success = generate_scene_boundaries(
                scene_file, 
                os.path.join(args.semantic_path, "scene_boundaries"),
                args.debug
            )
            
            if success:
                results["boundaries"]["success"] += 1
                scene_file_name_map[scene_name] = {"scene_file": scene_file, "boundaries_path": boundaries_file}
            else:
                results["boundaries"]["failed"] += 1
        except Exception as e:
            print(f"Error processing scene boundaries for {scene_name}: {e}")
            if args.debug:
                traceback.print_exc()
            results["boundaries"]["failed"] += 1
    
    # Stage 2: Extract point clouds
    print("\nExtracting point clouds:")
    for scene_name, scene_info in tqdm(scene_file_name_map.items(), desc="Extracting point clouds"):
        if "boundaries_path" not in scene_info:
            results["point_clouds"]["failed"] += 1
            continue
            
        pc_path = os.path.join(args.semantic_path, "point_clouds", f"{scene_name}.h5")
        
        # Skip if already processed
        if os.path.exists(pc_path) and not args.debug:
            results["point_clouds"]["success"] += 1
            scene_file_name_map[scene_name]["pc_path"] = pc_path
            continue
            
        try:
            pc_file, success = extract_point_cloud(
                scene_info["scene_file"],
                scene_info["boundaries_path"],
                os.path.join(args.semantic_path, "point_clouds"),
                args.debug
            )
            
            if success:
                results["point_clouds"]["success"] += 1
                scene_file_name_map[scene_name]["pc_path"] = pc_file
            else:
                results["point_clouds"]["failed"] += 1
        except Exception as e:
            print(f"Error extracting point cloud for {scene_name}: {e}")
            if args.debug:
                traceback.print_exc()
            results["point_clouds"]["failed"] += 1
    
    # Stage 3: Generate semantic maps
    print("\nGenerating semantic maps:")
    for scene_name, scene_info in tqdm(scene_file_name_map.items(), desc="Generating semantic maps"):
        if "pc_path" not in scene_info or "boundaries_path" not in scene_info:
            results["semantic_maps"]["failed"] += 1
            continue
            
        sem_map_path = os.path.join(args.semantic_path, "semantic_maps", f"{scene_name}.h5")
        
        # Skip if already processed
        if os.path.exists(sem_map_path) and not args.debug:
            results["semantic_maps"]["success"] += 1
            continue
            
        try:
            sem_map_file, success = generate_semantic_maps(
                scene_info["pc_path"],
                scene_info["boundaries_path"],
                os.path.join(args.semantic_path, "semantic_maps"),
                resolution=0.05,
                debug=args.debug
            )
            
            if success:
                results["semantic_maps"]["success"] += 1
            else:
                results["semantic_maps"]["failed"] += 1
        except Exception as e:
            print(f"Error generating semantic map for {scene_name}: {e}")
            if args.debug:
                traceback.print_exc()
            results["semantic_maps"]["failed"] += 1
    
    # Print summary
    print("\nPreprocessing complete!")
    print(f"Scene boundaries: {results['boundaries']['success']} succeeded, {results['boundaries']['failed']} failed")
    print(f"Point clouds: {results['point_clouds']['success']} succeeded, {results['point_clouds']['failed']} failed")
    print(f"Semantic maps: {results['semantic_maps']['success']} succeeded, {results['semantic_maps']['failed']} failed")
    
    # Save metadata for HM3D scenes
    print("\nCreating HM3D split metadata...")
    
    # Create dataset split information based on successfully processed scenes
    hm3d_splits = {
        "train": [],
        "val": []
    }
    
    # Assign scenes to splits (80% train, 20% val)
    processed_scenes = list(scene_file_name_map.keys())
    np.random.shuffle(processed_scenes)
    
    train_count = int(0.8 * len(processed_scenes))
    hm3d_splits["train"] = processed_scenes[:train_count]
    hm3d_splits["val"] = processed_scenes[train_count:]
    
    # Save split information
    with open(os.path.join(args.semantic_path, "hm3d_splits.json"), 'w') as f:
        json.dump(hm3d_splits, f, indent=2)
    
    print(f"Created dataset splits: {len(hm3d_splits['train'])} train scenes, {len(hm3d_splits['val'])} val scenes")
    print(f"Output directories:\n- {args.output_path}\n- {args.semantic_path}")

if __name__ == "__main__":
    main()