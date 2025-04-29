"""
Script to create semantic maps for HM3D scenes using the extracted semantic annotations.
This version handles the specific format found in hm3d-train-semantic-annots.tar 
and hm3d-train-semantic-config.tar files.
"""
import os
import numpy as np
import cv2
import habitat_sim
import json
from habitat_sim.utils.common import d3_40_colors_rgb
import argparse
from tqdm import tqdm
import glob
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hm3d")
    parser.add_argument("--output_dir", type=str, default="semantic_maps")
    parser.add_argument("--data_path", type=str, default=os.environ.get("ACTIVE_DATASET_DIR", "data/scene_datasets"))
    parser.add_argument("--semantic_path", type=str, default="/data/hm3d_semantic", 
                        help="Path to the HM3D semantic data directory containing both annotations and configs")
    parser.add_argument("--setup_semantics", action="store_true", 
                        help="Link semantic files to scene directories")
    parser.add_argument("--split", type=str, default="train", 
                        help="Dataset split (train, val, etc.)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up paths to the semantic annotation files
    args.semantic_path = os.path.abspath(args.semantic_path)
    
    # For backward compatibility, set both paths to the same directory
    args.semantic_annot_path = args.semantic_path
    args.semantic_config_path = args.semantic_path
    
    # Find all GLB scene files for HM3D
    scene_files = glob.glob(f"{args.data_path}/{args.dataset}/{args.split}/**/*.glb", recursive=True)
    
    if not scene_files:
        print(f"No .glb files found in {args.data_path}/{args.dataset}/{args.split}. Please check the data path.")
        return
    
    print(f"Found {len(scene_files)} scene files.")
    
    # If requested, set up semantic files by linking them to scene directories
    if args.setup_semantics:
        setup_semantic_files(args, scene_files)
    
    # Process each scene file
    for scene_file in tqdm(scene_files):
        scene_dir = os.path.dirname(scene_file)
        scene_id = os.path.basename(scene_dir)
        
        # Prepare paths for semantic data - trying both possible directory structures
        # First structure: Split is at the top level
        semantic_config_file = f"{args.semantic_path}/{args.split}/{scene_id}/{scene_id}.semantic_config.json"
        semantic_annot_file = f"{args.semantic_path}/{args.split}/{scene_id}/info_semantic.json"
        
        # Alternative structure: Config and annots in separate directories
        if not os.path.exists(semantic_config_file):
            alt_config_file = f"{args.semantic_path}/configs/{args.split}/{scene_id}/{scene_id}.semantic_config.json"
            if os.path.exists(alt_config_file):
                semantic_config_file = alt_config_file
                
        if not os.path.exists(semantic_annot_file):
            alt_annot_file = f"{args.semantic_path}/annots/{args.split}/{scene_id}/info_semantic.json"
            if os.path.exists(alt_annot_file):
                semantic_annot_file = alt_annot_file
        
        # Check if semantic files exist
        has_semantic_config = os.path.exists(semantic_config_file)
        has_semantic_annot = os.path.exists(semantic_annot_file)
        
        if not has_semantic_config or not has_semantic_annot:
            print(f"Missing semantic files for {scene_id}")
            if not has_semantic_config:
                print(f"  - Missing config: {semantic_config_file}")
            if not has_semantic_annot:
                print(f"  - Missing annotation: {semantic_annot_file}")
            continue
        
        # Load the semantic data
        with open(semantic_annot_file, 'r') as f:
            semantic_data = json.load(f)
        
        with open(semantic_config_file, 'r') as f:
            semantic_config = json.load(f)
        
        # Make sure the semantic files are linked in the scene directory
        scene_info_semantic = os.path.join(scene_dir, "info_semantic.json")
        if not os.path.exists(scene_info_semantic):
            try:
                # Copy the semantic annotation file to the scene directory
                shutil.copy(semantic_annot_file, scene_info_semantic)
                print(f"Copied semantic annotation to {scene_info_semantic}")
            except Exception as e:
                print(f"Failed to copy semantic annotation: {e}")
        
        # Configure the simulator
        sim_settings = {
            "width": 512, 
            "height": 512,
            "scene": scene_file,
            "scene_dataset": args.dataset,
            "sensor_height": 1.5,
            "color_sensor": True,
            "semantic_sensor": True,
            "seed": 1,
            "enable_physics": False,
        }
        
        # Create a Simulator configuration
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene_file
        cfg.enable_physics = False
        
        # Setup the sensor specifications
        sensor_specs = []
        
        # Add semantic sensor
        semantic_sensor_spec = habitat_sim.SensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [512, 512]
        sensor_specs.append(semantic_sensor_spec)
        
        # Create an agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        
        # Configure simulator
        hab_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
        
        # Initialize the simulator
        try:
            sim = habitat_sim.Simulator(hab_cfg)
            
            # Check if semantic scene is loaded
            if sim.semantic_scene is None:
                print(f"Semantic scene not loaded for {scene_id}, falling back to geometric approach")
                semantic_map = create_semantic_map_from_data(sim, semantic_data, semantic_config)
            else:
                semantic_map = create_semantic_map(sim, semantic_data, semantic_config)
            
            # Save the semantic map
            output_path = os.path.join(args.output_dir, f"{scene_id}_semantic_map.png")
            cv2.imwrite(output_path, semantic_map)
            
            print(f"Saved semantic map for {scene_id} to {output_path}")
            
            # Close the simulator
            sim.close()
            
        except Exception as e:
            print(f"Error processing scene {scene_file}: {e}")
            continue

def setup_semantic_files(args, scene_files):
    """Link or copy semantic files to the scene directories for Habitat to find them."""
    print("Setting up semantic files...")
    
    for scene_file in tqdm(scene_files):
        scene_dir = os.path.dirname(scene_file)
        scene_id = os.path.basename(scene_dir)
        
        # Prepare paths for semantic data - trying both possible directory structures
        # First structure: Split is at the top level
        semantic_config_file = f"{args.semantic_path}/{args.split}/{scene_id}/{scene_id}.semantic_config.json"
        semantic_annot_file = f"{args.semantic_path}/{args.split}/{scene_id}/info_semantic.json"
        
        # Alternative structure: Config and annots in separate directories
        if not os.path.exists(semantic_config_file):
            alt_config_file = f"{args.semantic_path}/configs/{args.split}/{scene_id}/{scene_id}.semantic_config.json"
            if os.path.exists(alt_config_file):
                semantic_config_file = alt_config_file
                
        if not os.path.exists(semantic_annot_file):
            alt_annot_file = f"{args.semantic_path}/annots/{args.split}/{scene_id}/info_semantic.json"
            if os.path.exists(alt_annot_file):
                semantic_annot_file = alt_annot_file
        
        # Check if semantic files exist
        if not os.path.exists(semantic_config_file) or not os.path.exists(semantic_annot_file):
            print(f"Missing semantic files for {scene_id}, skipping")
            continue
        
        # Target files in scene directory
        target_info_semantic = os.path.join(scene_dir, "info_semantic.json")
        target_scene_config = os.path.join(scene_dir, f"{scene_id}.semantic_config.json")
        
        # Copy the semantic annotation file
        if not os.path.exists(target_info_semantic):
            try:
                shutil.copy(semantic_annot_file, target_info_semantic)
                print(f"Copied {semantic_annot_file} -> {target_info_semantic}")
            except Exception as e:
                print(f"Failed to copy semantic annotation: {e}")
        
        # Copy the semantic config file
        if not os.path.exists(target_scene_config):
            try:
                shutil.copy(semantic_config_file, target_scene_config)
                print(f"Copied {semantic_config_file} -> {target_scene_config}")
            except Exception as e:
                print(f"Failed to copy semantic config: {e}")

def get_scene_bounds(sim):
    """Get the bounds of the scene for map creation."""
    # Try to get pathfinder bounds first
    if sim.pathfinder.is_loaded:
        bounds = sim.pathfinder.get_bounds()
        if bounds is not None:
            return [bounds[0][0], bounds[1][0], bounds[0][2], bounds[1][2]]
    
    # Fallback: estimate bounds by ray casting
    min_x, max_x = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    # Cast rays in all directions from the center
    center = np.array([0, 1.0, 0])  # Assuming y-up coordinate system
    directions = [
        [1, 0, 0], [-1, 0, 0],  # +/- X
        [0, 0, 1], [0, 0, -1],  # +/- Z
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]  # Diagonals
    ]
    
    for direction in directions:
        ray_dir = np.array(direction)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)  # Normalize
        
        hit_info = sim.cast_ray(center, ray_dir, 100.0)  # Max distance of 100m
        if hit_info.has_hit:
            hit_pos = center + ray_dir * hit_info.ray_distance
            min_x = min(min_x, hit_pos[0])
            max_x = max(max_x, hit_pos[0])
            min_z = min(min_z, hit_pos[2])
            max_z = max(max_z, hit_pos[2])
    
    # If we couldn't determine the bounds, use a default size
    if min_x == float('inf') or min_z == float('inf') or max_x == float('-inf') or max_z == float('-inf'):
        min_x, min_z = -10, -10
        max_x, max_z = 10, 10
    
    # Add some padding
    padding = 2.0
    min_x -= padding
    min_z -= padding
    max_x += padding
    max_z += padding
    
    return [min_x, max_x, min_z, max_z]

def create_semantic_map(sim, semantic_data, semantic_config):
    """Create a semantic map using both Habitat-Sim's semantic scene and the external semantic data."""
    semantic_scene = sim.semantic_scene
    
    # Get scene dimensions
    scene_bb = get_scene_bounds(sim)
    min_x, min_z = scene_bb[0], scene_bb[2]
    max_x, max_z = scene_bb[1], scene_bb[3]
    
    # Determine the resolution of the semantic map
    map_resolution = 0.05  # meters per pixel
    width = int((max_x - min_x) / map_resolution)
    height = int((max_z - min_z) / map_resolution)
    
    # Create an empty semantic map
    semantic_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process semantic objects from the simulator's semantic scene
    if semantic_scene and hasattr(semantic_scene, "objects"):
        print(f"Processing {len(semantic_scene.objects)} objects from semantic scene")
        category_count = {}
        
        for obj in semantic_scene.objects:
            if not hasattr(obj, "category") or obj.category is None or obj.category.name() == "none":
                continue
            
            # Track categories for debugging
            cat_name = obj.category.name()
            if cat_name not in category_count:
                category_count[cat_name] = 0
            category_count[cat_name] += 1
            
            # Get object color based on category index
            if hasattr(obj.category, "index"):
                color_idx = obj.category.index() % 40
            else:
                # Fallback to a hash of the category name
                color_idx = hash(obj.category.name()) % 40
            
            color = d3_40_colors_rgb[color_idx]
            
            # Draw object on map if it has an AABB
            if hasattr(obj, "aabb"):
                aabb = obj.aabb
                center_x, center_z = aabb.center[0], aabb.center[2]
                half_size_x, half_size_z = aabb.sizes[0]/2, aabb.sizes[2]/2
                
                # Convert to pixel coordinates
                x1 = int((center_x - half_size_x - min_x) / map_resolution)
                z1 = int((center_z - half_size_z - min_z) / map_resolution)
                x2 = int((center_x + half_size_x - min_x) / map_resolution)
                z2 = int((center_z + half_size_z - min_z) / map_resolution)
                
                # Clip to map boundaries
                x1 = max(0, min(width-1, x1))
                z1 = max(0, min(height-1, z1))
                x2 = max(0, min(width-1, x2))
                z2 = max(0, min(height-1, z2))
                
                # Draw the object on the semantic map
                cv2.rectangle(semantic_map, (x1, z1), (x2, z2), color.tolist(), -1)
        
        # Print category statistics
        print(f"Found {len(category_count)} unique categories:")
        for cat, count in category_count.items():
            print(f"  - {cat}: {count} instances")
    
    # If no objects were drawn, or we have very few objects, supplement with the external data
    if np.sum(semantic_map) == 0 or (semantic_scene and len(semantic_scene.objects) < 10):
        print("Supplementing with external semantic data")
        semantic_map = create_semantic_map_from_data(sim, semantic_data, semantic_config)
    
    return semantic_map

def create_semantic_map_from_data(sim, semantic_data, semantic_config):
    """Create a semantic map using external semantic data files."""
    # Get scene dimensions
    scene_bb = get_scene_bounds(sim)
    min_x, min_z = scene_bb[0], scene_bb[2]
    max_x, max_z = scene_bb[1], scene_bb[3]
    
    # Determine the resolution of the semantic map
    map_resolution = 0.05  # meters per pixel
    width = int((max_x - min_x) / map_resolution)
    height = int((max_z - min_z) / map_resolution)
    
    # Create an empty semantic map
    semantic_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Load category mapping from semantic config
    category_map = {}
    category_colors = {}
    
    if "categoryMap" in semantic_config:
        for cat_entry in semantic_config["categoryMap"]:
            if "name" in cat_entry and "mpcat40index" in cat_entry:
                cat_name = cat_entry["name"]
                cat_idx = cat_entry["mpcat40index"]
                category_map[cat_name] = cat_idx
                category_colors[cat_name] = d3_40_colors_rgb[cat_idx % 40].tolist()
    
    # Process semantic data
    if "objects" in semantic_data:
        print(f"Processing {len(semantic_data['objects'])} objects from semantic data")
        
        for obj in semantic_data["objects"]:
            if "category" not in obj or "aabb" not in obj:
                continue
            
            # Get category
            category = obj["category"]
            
            # Determine color based on category
            if isinstance(category, str) and category in category_colors:
                color = category_colors[category]
            elif isinstance(category, str):
                # Fallback to a hash of the category name
                color_idx = hash(category) % 40
                color = d3_40_colors_rgb[color_idx].tolist()
            else:
                # Default color for unknown categories
                color = [128, 128, 128]  # Gray
            
            # Get bounding box
            aabb = obj["aabb"]
            
            # Different formats handle AABBs differently
            min_coords = None
            max_coords = None
            
            if isinstance(aabb, dict):
                if "center" in aabb and "sizes" in aabb:
                    center = aabb["center"]
                    sizes = aabb["sizes"]
                    half_sizes = [s/2 for s in sizes]
                    min_coords = [center[i] - half_sizes[i] for i in range(3)]
                    max_coords = [center[i] + half_sizes[i] for i in range(3)]
                elif "min" in aabb and "max" in aabb:
                    min_coords = aabb["min"]
                    max_coords = aabb["max"]
            elif isinstance(aabb, list):
                if len(aabb) == 6:  # [minX, minY, minZ, maxX, maxY, maxZ]
                    min_coords = aabb[:3]
                    max_coords = aabb[3:]
            
            if min_coords is None or max_coords is None:
                continue
            
            # Project to 2D (top-down view using X and Z coordinates)
            x1 = int((min_coords[0] - min_x) / map_resolution)
            z1 = int((min_coords[2] - min_z) / map_resolution)
            x2 = int((max_coords[0] - min_x) / map_resolution)
            z2 = int((max_coords[2] - min_z) / map_resolution)
            
            # Clip to map boundaries
            x1 = max(0, min(width-1, x1))
            z1 = max(0, min(height-1, z1))
            x2 = max(0, min(width-1, x2))
            z2 = max(0, min(height-1, z2))
            
            # Draw the object on the semantic map
            cv2.rectangle(semantic_map, (x1, z1), (x2, z2), color, -1)
            
            # Optionally add a white border for visibility
            cv2.rectangle(semantic_map, (x1, z1), (x2, z2), [255, 255, 255], 1)
    
    # If still no objects drawn, fall back to a geometry-based approach
    if np.sum(semantic_map) == 0:
        print("No objects found in semantic data, falling back to geometry detection")
        
        # Use pathfinder to detect floors
        if sim.pathfinder.is_loaded:
            print("Using pathfinder to detect floors")
            step_size = max(int(width / 100), 1)
            for x in range(0, width, step_size):
                for z in range(0, height, step_size):
                    world_x = min_x + x * map_resolution
                    world_z = min_z + z * map_resolution
                    
                    point = np.array([world_x, 0, world_z])
                    nav_point = sim.pathfinder.snap_point(point)
                    
                    if sim.pathfinder.is_navigable(nav_point):
                        # Floor - green
                        cv2.circle(semantic_map, (x, z), step_size//2, [0, 255, 0], -1)
        
        # Use ray casting to detect walls
        print("Using ray casting to detect walls")
        step_size = max(int(width / 50), 1)
        for x in range(0, width, step_size * 2):
            for height_level in [0.5, 1.5]:
                # Cast rays in Z direction
                for direction in [1, -1]:  # Forward and backward
                    ray_origin = np.array([min_x + x * map_resolution, height_level, 
                                         min_z if direction > 0 else max_z])
                    ray_direction = np.array([0, 0, direction])
                    
                    hit_info = sim.cast_ray(ray_origin, ray_direction)
                    if hit_info.has_hit:
                        hit_pos = ray_origin + ray_direction * hit_info.ray_distance
                        hit_x = int((hit_pos[0] - min_x) / map_resolution)
                        hit_z = int((hit_pos[2] - min_z) / map_resolution)
                        
                        if 0 <= hit_x < width and 0 <= hit_z < height:
                            # Wall - blue
                            cv2.circle(semantic_map, (hit_x, hit_z), step_size//2, [255, 0, 0], -1)
                            
        for z in range(0, height, step_size * 2):
            for height_level in [0.5, 1.5]:
                # Cast rays in X direction
                for direction in [1, -1]:  # Right and left
                    ray_origin = np.array([min_x if direction > 0 else max_x, 
                                         height_level, min_z + z * map_resolution])
                    ray_direction = np.array([direction, 0, 0])
                    
                    hit_info = sim.cast_ray(ray_origin, ray_direction)
                    if hit_info.has_hit:
                        hit_pos = ray_origin + ray_direction * hit_info.ray_distance
                        hit_x = int((hit_pos[0] - min_x) / map_resolution)
                        hit_z = int((hit_pos[2] - min_z) / map_resolution)
                        
                        if 0 <= hit_x < width and 0 <= hit_z < height:
                            # Wall - blue
                            cv2.circle(semantic_map, (hit_x, hit_z), step_size//2, [255, 0, 0], -1)
    
    return semantic_map

# Simple function to run the script with default parameters
def run_with_defaults():
    """Run the script with default parameters pointing to /data/hm3d_semantic"""
    import sys
    sys.argv = [
        'create_semantic_maps.py',
        '--data_path', os.environ.get("ACTIVE_DATASET_DIR", "data/scene_datasets"),
        '--semantic_path', '/data/hm3d_semantic',
        '--setup_semantics',
        '--split', 'train'
    ]
    main()

if __name__ == "__main__":
    main()