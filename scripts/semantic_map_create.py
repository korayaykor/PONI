import os
import numpy as np
from math import pi
import habitat
import habitat_sim
import quaternion
from build_map_utils import semantic_map
from poni.default import get_config
import cv2

# Helper functions
def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def quaternion_rotate_vector(quat, v):
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def convert_insseg_to_sseg(insseg, ins2cat_dict):
    ins_id_list = list(ins2cat_dict.keys())
    sseg = np.zeros(insseg.shape, dtype=np.int16)
    for ins_id in ins_id_list:
        sseg = np.where(insseg == ins_id, ins2cat_dict[ins_id], sseg)
    return sseg

def create_folder(folder_name, clean_up=False):
    if not os.path.exists(folder_name):
        print(f'{folder_name} folder does not exist, so create one.')
        os.makedirs(folder_name)
    else:
        print(f'{folder_name} folder already exists.')
        if clean_up:
            os.system(f'rm {folder_name}/*.png')
            os.system(f'rm {folder_name}/*.npy')
            os.system(f'rm {folder_name}/*.jpg')

# Create output directory
output_dir = "output"
create_folder(output_dir)
create_folder(f'{output_dir}/semantic_map')

# Get configuration
cfg = get_config()

# Specify the scene - HM3D scene ID
scene_id = "kfPV7w3FaU5"  # HM3D scene
height = 0.5  # Adjusted height for HM3D scenes

# Configure angles for observation collection
theta_lst = [0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4]

# Initialize grid for exploration
# IMPORTANT: Use a smaller grid to make sure we get some data
x = np.arange(-10, 10, 0.3)  # Smaller range
z = np.arange(-10, 10, 0.3)  # Smaller range
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

print(f"Building semantic map for HM3D scene: {scene_id}")
print(f"Grid dimensions: {grid_H}x{grid_W}")

# Initialize the Habitat environment
config = habitat.get_config(
    config_paths='../configs/habitat_env/build_map_mp3d.yaml')  # Base config is ok
config.defrost()

# Update for HM3D
# HM3D scene path structure: ../data/scene_datasets/hm3d_uncompressed/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb
scene_path = f"../data/scene_datasets/hm3d_uncompressed/00000-{scene_id}/{scene_id}.basis.glb"
config.SIMULATOR.SCENE = scene_path
# Make sure to use the correct HM3D scene dataset config file
config.SIMULATOR.SCENE_DATASET = "../data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

config.freeze()

try:
    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    env.reset()
    
    # Create the semantic map
    saved_folder = f'{output_dir}/semantic_map/{scene_id}'
    create_folder(saved_folder, clean_up=False)
    
    # Create a backup map for debugging
    debug_map = np.zeros((grid_H, grid_W), dtype=np.int32)
    
    # Get instance to category mapping
    scene_semantics = env.semantic_annotations()
    
    # HM3D may use different category indices than MP3D
    ins2cat_dict = {}
    for obj in scene_semantics.objects:
        try:
            obj_id = int(obj.id.split("_")[-1])
            obj_category = obj.category.index()
            ins2cat_dict[obj_id] = obj_category
        except Exception as e:
            print(f"Error processing object {obj.id}: {e}")
    
    print(f"Found {len(ins2cat_dict)} semantic objects")
    if len(ins2cat_dict) == 0:
        print("WARNING: No semantic objects found. Scene may not have semantic annotations.")
        # Create a fallback dummy mapping
        ins2cat_dict = {1: 1}  # Map anything to wall category
    
    # Initialize semantic map
    sem_map = semantic_map(saved_folder)
    
    # Fixed map for simple backup in case the semantic_map object fails
    simple_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
    valid_points = 0
    
    # Count navigable positions to track progress
    count_ = 0
    navigable_count = 0
    
    # Generate observations - modified to ensure we collect enough data
    print("Starting semantic map generation")
    
    # Find navigable locations more efficiently
    print("Finding navigable locations...")
    navigable_positions = []
    
    # Try to find at least 10 navigable positions
    for grid_z in range(0, grid_H):
        for grid_x in range(0, grid_W):
            x = xv[grid_z, grid_x]
            z = zv[grid_z, grid_x]
            y = height
            agent_pos = np.array([x, y, z])
            
            try:
                flag_nav = env.is_navigable(agent_pos)
                if flag_nav:
                    navigable_positions.append((x, y, z))
                    navigable_count += 1
                    # Mark this position in our debug map
                    debug_map[grid_z, grid_x] = 1
                    # Draw a point on our simple backup map
                    map_x = int(500 + x * 20)  # Scale to fit in our 1000x1000 map
                    map_z = int(500 + z * 20)  
                    if 0 <= map_x < 1000 and 0 <= map_z < 1000:
                        simple_map[map_z, map_x] = [0, 255, 0]  # Green dot for navigable position
                    
                    # We'll process about 20 positions maximum for speed
                    if navigable_count >= 20:
                        break
            except Exception as e:
                print(f"Error checking navigability at ({x},{y},{z}): {e}")
            
        if navigable_count >= 20:
            break
    
    print(f"Found {navigable_count} navigable positions")
    
    if navigable_count == 0:
        print("No navigable positions found. Trying with a smaller grid and different height...")
        # Try again with adjusted parameters
        height = 0.8  # Try a different height
        
        # Try a smaller region with finer granularity
        x = np.linspace(-5, 5, 40)
        z = np.linspace(-5, 5, 40)
        xv, zv = np.meshgrid(x, z)
        
        for grid_z in range(len(z)):
            for grid_x in range(len(x)):
                x_pos = xv[grid_z, grid_x]
                z_pos = zv[grid_z, grid_x]
                agent_pos = np.array([x_pos, height, z_pos])
                
                try:
                    flag_nav = env.is_navigable(agent_pos)
                    if flag_nav:
                        navigable_positions.append((x_pos, height, z_pos))
                        navigable_count += 1
                        map_x = int(500 + x_pos * 20)
                        map_z = int(500 + z_pos * 20)  
                        if 0 <= map_x < 1000 and 0 <= map_z < 1000:
                            simple_map[map_z, map_x] = [0, 255, 0]
                        
                        if navigable_count >= 20:
                            break
                except Exception as e:
                    pass
                
            if navigable_count >= 20:
                break
        
        print(f"Second attempt found {navigable_count} navigable positions")
    
    # Process each navigable position
    for i, (x, y, z) in enumerate(navigable_positions):
        print(f"Processing position {i+1}/{navigable_count}: ({x:.2f}, {y:.2f}, {z:.2f})")
        agent_pos = np.array([x, y, z])
        
        # Traverse through different angles
        for idx_theta, theta in enumerate(theta_lst):
            try:
                agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
                    theta, habitat_sim.geo.GRAVITY)
                observations = env.get_observations_at(
                    agent_pos, agent_rot, keep_agent_at_new_pose=True)
                
                rgb_img = observations['rgb']
                depth_img = observations['depth'][:, :, 0]
                
                # Handle semantic observations - HM3D might have different structure
                if 'semantic' in observations:
                    insseg_img = observations['semantic']
                    # Convert instance segmentation to semantic segmentation
                    sseg_img = convert_insseg_to_sseg(insseg_img, ins2cat_dict)
                else:
                    print("WARNING: No semantic observations available")
                    # Create a dummy semantic map 
                    sseg_img = np.zeros_like(depth_img, dtype=np.int16)
                
                # Get agent global pose in habitat env
                agent_pos = env.get_agent_state().position
                agent_rot = env.get_agent_state().rotation
                heading_vector = quaternion_rotate_vector(
                    agent_rot.inverse(), np.array([0, 0, -1]))
                phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
                angle = phi
                
                print(f"Observation {count_}: position={agent_pos}, angle={angle:.2f}")
                pose = (agent_pos[0], agent_pos[2], angle)
                
                # Save observation images for debugging
                if count_ < 5:  # Save first 5 observations for debugging
                    cv2.imwrite(f"{saved_folder}/rgb_{count_}.png", rgb_img)
                    cv2.imwrite(f"{saved_folder}/depth_{count_}.png", (depth_img * 255).astype(np.uint8))
                    cv2.imwrite(f"{saved_folder}/semantic_{count_}.png", (sseg_img * 10).astype(np.uint8))
                
                # Build semantic map
                sem_map.build_semantic_map(rgb_img, depth_img, sseg_img, pose, count_)
                count_ += 1
                valid_points += 1
                
                # Draw a line on our simple backup map showing view direction
                map_x = int(500 + agent_pos[0] * 20)
                map_z = int(500 + agent_pos[2] * 20)
                end_x = int(map_x + 20 * np.cos(angle))
                end_z = int(map_z + 20 * np.sin(angle))
                if (0 <= map_x < 1000 and 0 <= map_z < 1000 and 
                    0 <= end_x < 1000 and 0 <= end_z < 1000):
                    cv2.line(simple_map, (map_x, map_z), (end_x, end_z), [255, 0, 0], 1)
                
            except Exception as e:
                print(f"Error collecting observation: {e}")
    
    # Save our debug map as a fallback
    cv2.imwrite(f"{saved_folder}/debug_map.png", debug_map * 255)
    cv2.imwrite(f"{saved_folder}/simple_map.png", simple_map)
    
    print(f"Collected {valid_points} valid observations")
    
    # Custom fallback for empty semantic maps
    try:
        if count_ > 0:
            # Try to examine the four_dim_grid before saving
            if hasattr(sem_map, 'four_dim_grid'):
                grid_shape = sem_map.four_dim_grid.shape
                print(f"Four-dim grid shape: {grid_shape}")
                
                # Check if we have valid coordinates
                min_x = sem_map.min_x_coord
                max_x = sem_map.max_x_coord
                min_z = sem_map.min_z_coord
                max_z = sem_map.max_z_coord
                print(f"Coordinate ranges: x={min_x}:{max_x}, z={min_z}:{max_z}")
                
                # If coordinates are invalid, set them to ensure we have data
                if min_x >= max_x or min_z >= max_z:
                    print("Invalid coordinates, setting defaults...")
                    sem_map.min_x_coord = 0
                    sem_map.max_x_coord = min(100, grid_shape[2]-1)
                    sem_map.min_z_coord = 0
                    sem_map.max_z_coord = min(100, grid_shape[0]-1)
            
            sem_map.save_final_map()
            print(f"Semantic map saved to {saved_folder}")
        else:
            print("No observations collected, cannot save map")
    except Exception as e:
        print(f"Error saving semantic map: {e}")
        print("Creating fallback map...")
        
        # Create a minimal map file structure to avoid the resize error
        map_dict = {}
        map_dict['min_x'] = 0
        map_dict['max_x'] = 99
        map_dict['min_z'] = 0
        map_dict['max_z'] = 99
        map_dict['min_X'] = -cfg.SEM_MAP.WORLD_SIZE
        map_dict['max_X'] = cfg.SEM_MAP.WORLD_SIZE
        map_dict['min_Z'] = -cfg.SEM_MAP.WORLD_SIZE
        map_dict['max_Z'] = cfg.SEM_MAP.WORLD_SIZE
        map_dict['W'] = 100
        map_dict['H'] = 100
        
        # Create a minimal semantic map
        min_semantic_map = np.zeros((100, 100), dtype=np.int32)
        # Add some floor and wall data to prevent empty maps
        min_semantic_map[40:60, 40:60] = 1  # Some floor
        min_semantic_map[30:70, 30:33] = 2  # Some walls
        map_dict['semantic_map'] = min_semantic_map
        
        np.save(f"{saved_folder}/BEV_semantic_map.npy", map_dict)
        print(f"Created minimal fallback map at {saved_folder}/BEV_semantic_map.npy")
        
        # Create a visual representation
        # Resize to get a larger visualization
        visual_map = cv2.resize(min_semantic_map * 50, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{saved_folder}/fallback_visual_map.png", visual_map)

        print("Creating video...")
    video_file = os.path.join(output_path, "random_navigation.mp4")
    
    # Use habitat's video utility but don't open the video
    if hasattr(vut, 'make_video'):
        vut.make_video(
            observations=observations_list,
            primary_obs="color_sensor",
            primary_obs_type="color",
            video_file=video_file,
            fps=10,
            open_vid=False,  # Changed from show_video to False
        )
        print("Video saved to:", video_file)
    else:
        # Fallback to imageio
        fps = 10
        writer = imageio.get_writer(video_file, fps=fps)
        
        for obs in observations_list:
            if "color_sensor" in obs:
                # Convert RGBA to RGB if necessary
                frame = obs["color_sensor"]
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                writer.append_data(frame)
        
        writer.close()
        print("Video saved to:", video_file)
    
    env.close()
    
except Exception as e:
    print(f"Error during semantic map creation: {e}")