import glob
import json
import math
import multiprocessing as mp
import os
import random
import re


from collections import defaultdict
import imageio
import cv2
import h5py
import numpy as np
import torch
import tqdm
import trimesh
import matplotlib.font_manager
from PIL import Image, ImageDraw, ImageFont
from torch_scatter import scatter_max

Image.MAX_IMAGE_PIXELS = 1000000000
import poni.hab_utils as hab_utils
import matplotlib.pyplot as plt
from plyfile import PlyData

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# Output directory for videos
output_directory = "output"
output_path = "output/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
else:
    show_video = True  # Changed to True for testing
    do_make_video = True  # Changed to True for testing
    display = True  # Changed to True for testing

# Import the maps module alone for topdown mapping
if display:
    try:
        from habitat.utils.visualizations import maps
    except ImportError:
        print("Warning: Could not import habitat maps module")

# Second section with more complex configuration
#test_scene = "../data/scene_datasets/mp3d_uncompressed/17DRP5sb8fy/17DRP5sb8fy.glb"
test_scene = "../data/scene_datasets/hm3d_uncompressed/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb"

rgb_sensor = True
depth_sensor = True
semantic_sensor = True

sim_settings = {
    "width": 256,
    "height": 256,
    "scene": test_scene,
    "default_agent": 0,
    "sensor_height": 1.5,
    "color_sensor": rgb_sensor,
    "depth_sensor": depth_sensor,
    "semantic_sensor": semantic_sensor,
    "seed": 1,
    "enable_physics": False,
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
try:
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

def print_scene_recur(scene, limit_output=10):
    print("House has %d levels, %d regions and %d objects" % 
        (len(scene.levels), len(scene.regions), len(scene.objects)))
    print("House center:%s dims:%s" % (scene.aabb.center, scene.aabb.sizes))

    count = 0
    for level in scene.levels:
        print("Level id:{}, center:{}, dims:{}".format(
            level.id, level.aabb.center, level.aabb.sizes
        ))
        for region in level.regions:
            print("Region id:{}, category:{}, center:{}, dims:{}".format(
                region.id, region.category.name(), region.aabb.center, region.aabb.sizes
            ))
            for obj in region.objects:
                print("Object id:{}, category:{}, center:{}, dims:{}".format(
                    obj.id, obj.category.name(), obj.aabb.center, obj.aabb.sizes
                ))
                count += 1
                if count >= limit_output:
                    return None

# Print semantic annotation information
scene = sim.semantic_scene
print_scene_recur(scene)

# Set up randomness
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# Collect observations for video
observations_list = []
total_frames = 0
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
max_frames = 200  # Increased for better video

while total_frames < max_frames:
    action = random.choice(action_names)
    print("action", action)
    observations = sim.step(action)
    
    # Store observations for video
    observations_list.append(observations)
    
    # Display if enabled
    if display:
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        display_sample(rgb, semantic, depth)
    
    total_frames += 1

# Create video if enabled
# Replace this section of code (around line 310-325):
if do_make_video and observations_list:
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

        # convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)


# @markdown ###Configure Example Parameters:
# @markdown Configure the map resolution:
meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown ---
# @markdown Customize the map slice height (global y coordinate):
custom_height = False  # @param {type:"boolean"}
height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)

print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not custom_height:
    # get bounding box minumum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]


if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # Get the topdown map directly from the Habitat-sim API with PathFinder.get_topdown_view
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

    if display:
        # Process the map using the Habitat-Lab maps module
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        print("Displaying the raw map from get_topdown_view:")
        display_map(sim_topdown_map)
        print("Displaying the map from the Habitat-Lab maps module:")
        display_map(hablab_topdown_map)

        # Save the raw map from get_topdown_view
        raw_map_filename = os.path.join(output_path, "raw_topdown_map.png")
        # Convert boolean array to grayscale image (True -> 255, False -> 0)
        raw_map_image = (sim_topdown_map * 255).astype(np.uint8)
        imageio.imsave(raw_map_filename, raw_map_image)
        print(f"Raw topdown map saved to: {raw_map_filename}")

        # Save the Habitat-Lab processed map
        hablab_map_filename = os.path.join(output_path, "hablab_topdown_map.png")
        imageio.imsave(hablab_map_filename, hablab_topdown_map)
        print(f"Habitat-Lab topdown map saved to: {hablab_map_filename}")

def save_map_with_keypoints(topdown_map, key_points, filename):
    """Save a topdown map with key points overlaid"""
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()  # Close the figure to free memory
    print(f"Map with key points saved to: {filename}")

# This should be placed after your first topdown map section
# NavMesh querying section
vis_points = []  # Initialize this before any conditionals

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # NavMesh area and bounding box can be queried
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    # A random point on the NavMesh
    pathfinder_seed = 1
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()
    print("Random navigable point : " + str(nav_point))
    print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

    # Island radius
    print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

    # Closest boundary point
    max_search_radius = 2.0
    print(
        "Distance to obstacle: "
        + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
    )
    hit_record = sim.pathfinder.closest_obstacle_surface_point(
        nav_point, max_search_radius
    )
    print("Closest obstacle HitRecord:")
    print(" point: " + str(hit_record.hit_pos))
    print(" normal: " + str(hit_record.hit_normal))
    print(" distance: " + str(hit_record.hit_dist))

    vis_points = [nav_point]

    # HitRecord will have infinite distance if no valid point was found
    if math.isinf(hit_record.hit_dist):
        print("No obstacle found within search radius.")
    else:
        # Points near the boundary or above the NavMesh can be snapped onto it
        perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
        print("Perturbed point : " + str(perturbed_point))
        print(
            "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
        )
        snapped_point = sim.pathfinder.snap_point(perturbed_point)
        print("Snapped point : " + str(snapped_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
        vis_points.append(snapped_point)

    # Visualization
    meters_per_pixel = 0.1

    if display and vis_points:  # Make sure vis_points is not empty
        xy_vis_points = convert_points_to_topdown(
            sim.pathfinder, vis_points, meters_per_pixel
        )
        # use the y coordinate of the sampled nav_point for the map height slice
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        
        print("\nDisplay the map with key_point overlay:")
        display_map(top_down_map, key_points=xy_vis_points)
        
        # Save the map with key points
        navmesh_query_map_filename = os.path.join(output_path, "navmesh_query_map_with_keypoints.png")
        save_map_with_keypoints(top_down_map, xy_vis_points, navmesh_query_map_filename)