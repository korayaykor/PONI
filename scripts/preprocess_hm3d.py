#!/usr/bin/env python3
"""
Script to preprocess HM3D dataset files for use with PONI.
This script adapts the HM3D format to be compatible with the existing PONI pipeline.
"""

import os
import argparse
import json
import numpy as np
import h5py
import habitat_sim
import trimesh
from tqdm import tqdm

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
    return parser

def load_scene_metadata(hm3d_path):
    """Load scene metadata from HM3D dataset."""
    metadata_file = os.path.join(hm3d_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Warning: Metadata file not found at {metadata_file}")
        return {}
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def generate_scene_boundaries(scene_path, output_dir):
    """Extract scene boundaries and save them to a file."""
    try:
        # Create simulator config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.88  # Same height as in PONI
        agent_cfg.radius = 0.18  # Same radius as in PONI
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        # Get scene bounds
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
        
        # Convert to the format expected by PONI
        scene_name = os.path.basename(scene_path).split('.')[0]
        buffer = np.array([3.0, 0.0, 3.0])  # Same buffer as in PONI
        lower_bound_with_buffer = lower_bound - buffer
        upper_bound_with_buffer = upper_bound + buffer
        
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
        
        # Get floor heights
        floor_heights = []
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        y_coords = navmesh_vertices[:, 1]
        
        # Simple clustering of y coordinates to find floor heights
        y_coords = np.round(y_coords * 10) / 10  # Round to nearest 0.1m
        unique_heights, counts = np.unique(y_coords, return_counts=True)
        
        # Filter out minor heights (noise)
        significant_heights = unique_heights[counts > len(y_coords) * 0.05]
        
        # Sort heights
        significant_heights.sort()
        
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
        
        # Save boundaries
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{scene_name}.json")
        with open(output_file, 'w') as f:
            json.dump(boundaries, f, indent=2)
        
        sim.close()
        return boundaries
        
    except Exception as e:
        print(f"Error processing {scene_path}: {e}")
        return None

def extract_point_cloud(scene_path, semantic_path, output_path, boundaries):
    """Extract point cloud from a scene and save it."""
    try:
        scene_name = os.path.basename(scene_path).split('.')[0]
        
        # Load the scene mesh
        scene = trimesh.load(scene_path)
        
        # Extract vertices and faces
        vertices = np.array(scene.vertices)
        faces = np.array(scene.faces)
        
        # Create point cloud
        points = []
        for face in faces:
            # Sample points from triangle
            v1, v2, v3 = vertices[face]
            # Generate barycentric coordinates
            u = np.random.rand(100, 1)
            v = np.random.rand(100, 1)
            mask = u + v > 1
            u[mask] = 1 - u[mask]
            v[mask] = 1 - v[mask]
            w = 1 - u - v
            
            # Generate points
            sampled_points = u * v1 + v * v2 + w * v3
            points.extend(sampled_points)
        
        points = np.array(points)
        
        # TODO: Get semantic labels for points
        # This would require access to semantic annotations for HM3D
        # For now, we'll create dummy semantic labels
        sem_ids = np.zeros(len(points), dtype=np.int32)
        obj_ids = np.zeros(len(points), dtype=np.int32)
        
        # Identify floor and wall points based on orientation and height
        # This is a simplistic approach, actual semantic segmentation would be better
        for i, point in enumerate(points):
            if point[1] < boundaries[scene_name]["ylo"] + 0.1:
                # Floor points
                sem_ids[i] = 1  # Assuming 1 is floor
            elif np.abs(point[1] - boundaries[scene_name]["center"][1]) > boundaries[scene_name]["sizes"][1] * 0.3:
                # Wall points (simplistic approach)
                sem_ids[i] = 2  # Assuming 2 is wall
        
        # Save to HDF5
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{scene_name}.h5")
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset("vertices", data=points)
            f.create_dataset("sem_ids", data=sem_ids)
            f.create_dataset("obj_ids", data=obj_ids)
        
        return True
        
    except Exception as e:
        print(f"Error extracting point cloud from {scene_path}: {e}")
        return False

def generate_semantic_maps(pc_path, boundaries_path, output_path, resolution=0.05):
    """Generate semantic maps from point clouds."""
    try:
        # Load point cloud
        with h5py.File(pc_path, 'r') as f:
            vertices = np.array(f["vertices"])
            sem_ids = np.array(f["sem_ids"])
            obj_ids = np.array(f["obj_ids"])
        
        # Load boundaries
        scene_name = os.path.basename(pc_path).split('.')[0]
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
            map_vertices = map_vertices[valid_mask]
            
            # Create semantic map
            sem_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.int32)
            obj_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.int32)
            height_map = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.float32)
            
            # Fill maps
            for i in range(len(grid_x)):
                x, z = grid_x[i], grid_z[i]
                if height_map[z, x] < map_vertices[i, 1]:
                    height_map[z, x] = map_vertices[i, 1]
                    sem_map[z, x] = floor_sem_ids[i]
                    obj_map[z, x] = floor_obj_ids[i]
            
            # Save to HDF5
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{scene_name}.h5")
            
            floor_id = floor_key.split('_')[-1]
            
            with h5py.File(output_file, 'a') as f:
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
                sem_rgb[sem_map == 1] = [230, 230, 230]
                # Wall is dark gray
                sem_rgb[sem_map == 2] = [77, 77, 77]
                # Other objects get colors based on their semantic ID
                for id in range(3, 20):  # Assuming up to 20 semantic categories
                    if np.any(sem_map == id):
                        # Use a color from a predefined palette
                        color_idx = (id - 3) % len(d3_40_colors_rgb)
                        sem_rgb[sem_map == id] = d3_40_colors_rgb[color_idx]
                
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=sem_rgb)
            
            # Also save metadata
            if not os.path.exists(os.path.join(output_path, "semmap_GT_info.json")):
                metadata = {}
            else:
                with open(os.path.join(output_path, "semmap_GT_info.json"), 'r') as f:
                    metadata = json.load(f)
            
            if scene_name not in metadata:
                metadata[scene_name] = {
                    "dim": world_dim_discret,
                    "central_pos": central_pos.tolist(),
                    "map_world_shift": map_world_shift.tolist(),
                    "resolution": resolution
                }
            
            metadata[scene_name][floor_id] = {
                "y_min": float(floor_bounds["ylo"])
            }
            
            with open(os.path.join(output_path, "semmap_GT_info.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error generating semantic maps for {pc_path}: {e}")
        return False

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "scene_boundaries"), exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "point_clouds"), exist_ok=True)
    os.makedirs(os.path.join(args.semantic_path, "semantic_maps"), exist_ok=True)
    
    # Load metadata
    metadata = load_scene_metadata(args.hm3d_path)
    
    # Define d3_40_colors_rgb for visualization
    # This should match the one in poni/constants.py
    global d3_40_colors_rgb
    d3_40_colors_rgb = np.array([
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [206, 219, 156],
        [140, 109, 49],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
    ], dtype=np.uint8)
    
    # Find all GLB files in the HM3D path
    scene_files = []
    for root, dirs, files in os.walk(args.hm3d_path):
        for file in files:
            if file.endswith(".glb"):
                scene_files.append(os.path.join(root, file))
    
    print(f"Found {len(scene_files)} scene files.")
    
    # Process each scene
    for scene_file in tqdm(scene_files, desc="Processing scenes"):
        scene_name = os.path.basename(scene_file).split('.')[0]
        
        # Generate scene boundaries
        boundaries = generate_scene_boundaries(
            scene_file, 
            os.path.join(args.semantic_path, "scene_boundaries")
        )
        
        if boundaries:
            # Extract point cloud
            extract_point_cloud(
                scene_file,
                os.path.join(args.semantic_path, "scene_boundaries", f"{scene_name}.json"),
                os.path.join(args.semantic_path, "point_clouds"),
                boundaries
            )
            
            # Generate semantic maps
            generate_semantic_maps(
                os.path.join(args.semantic_path, "point_clouds", f"{scene_name}.h5"),
                os.path.join(args.semantic_path, "scene_boundaries", f"{scene_name}.json"),
                os.path.join(args.semantic_path, "semantic_maps")
            )
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()