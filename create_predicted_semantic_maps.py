#!/usr/bin/env python3
"""
Script to demonstrate how to use PONI's neural Semantic_Mapping class 
to generate predicted semantic maps from RGB-D observations and save them to files.

This script shows the complete pipeline from loading RGB-D observations to 
generating neural network-based semantic map predictions and saving them.

Usage:
    python create_predicted_semantic_maps.py --num_steps 10 --save_dir ./predicted_maps
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import h5py
from argparse import Namespace
from PIL import Image

# Import PONI's semantic mapping module
from semexp.model import Semantic_Mapping
from hlab.utils.semantic_mapping import Semantic_Mapping as SM_Wrapper


def create_mock_args(device="cuda:0", num_processes=1):
    """Create mock arguments for Semantic_Mapping initialization"""
    args = Namespace()
    
    # Device and processing
    args.device = device
    args.num_processes = num_processes
    
    # Frame/image dimensions
    args.frame_height = 480  # Standard RGB-D frame height
    args.frame_width = 640   # Standard RGB-D frame width
    args.hfov = 79.0        # Horizontal field of view in degrees
    
    # Map parameters
    args.map_resolution = 5        # cm per pixel
    args.map_size_cm = 2400       # Total map size in cm (24m x 24m)
    args.camera_height = 88.0     # Camera height in cm
    args.global_downscaling = 1   # No downscaling
    
    # Semantic mapping parameters
    args.vision_range = 100       # Vision range in map pixels
    args.du_scale = 1            # Depth upsampling scale
    args.cat_pred_threshold = 5.0     # Category prediction threshold
    args.exp_pred_threshold = 1.0     # Explored area threshold
    args.map_pred_threshold = 1.0     # Map prediction threshold
    args.num_sem_categories = 21      # Number of semantic categories
    
    return args


def create_mock_rgb_depth_observation(height=480, width=640, num_sem_categories=21):
    """
    Create a mock RGB-D observation with semantic predictions.
    
    Args:
        height: Image height
        width: Image width
        num_sem_categories: Number of semantic categories
        
    Returns:
        obs: Tensor of shape (1, 4+num_sem_categories, height, width)
             Channels: [RGB (3), Depth (1), Semantic Categories (num_sem_categories)]
    """
    batch_size = 1
    
    # Create mock RGB image (normalized to [0,1])
    rgb = torch.rand(batch_size, 3, height, width)
    
    # Create mock depth image (in meters, typical range 0.1-10m)
    depth = torch.rand(batch_size, 1, height, width) * 5.0 + 0.5
    
    # Create mock semantic segmentation (probability distributions)
    # Each pixel has probabilities for each semantic category
    semantic = torch.rand(batch_size, num_sem_categories, height, width)
    # Normalize to make it look like probability distributions
    semantic = torch.softmax(semantic, dim=1)
    
    # Combine all channels: RGB + Depth + Semantic
    obs = torch.cat([rgb, depth, semantic], dim=1)
    
    return obs


def create_mock_pose():
    """
    Create a mock pose observation.
    
    Returns:
        pose: Tensor of shape (1, 3) with [dx, dy, dtheta]
              dx, dy: translation in meters
              dtheta: rotation in radians
    """
    # Small random movement (agent moving forward with slight rotation)
    dx = np.random.normal(0.1, 0.05)      # Forward movement ~10cm
    dy = np.random.normal(0.0, 0.02)      # Side movement ~0cm
    dtheta = np.random.normal(0.0, 0.1)   # Rotation ~0 radians
    
    pose = torch.tensor([[dx, dy, dtheta]], dtype=torch.float32)
    return pose


def visualize_semantic_map(semantic_map, save_path=None):
    """
    Visualize a semantic map with different colors for each category.
    
    Args:
        semantic_map: Tensor of shape (num_categories, H, W)
        save_path: Path to save the visualization
        
    Returns:
        vis_rgb: RGB visualization as numpy array
    """
    num_categories, H, W = semantic_map.shape
    
    # Create a colored visualization
    # Take argmax to get the dominant category at each pixel
    category_map = torch.argmax(semantic_map, dim=0).cpu().numpy()
    
    # Create a color palette (you can customize this)
    colors = [
        [0, 0, 0],        # 0: Empty/Unknown - Black
        [128, 128, 128],  # 1: Floor - Gray
        [64, 64, 64],     # 2: Wall - Dark Gray
        [255, 0, 0],      # 3: Chair - Red
        [0, 255, 0],      # 4: Table - Green
        [0, 0, 255],      # 5: Bed - Blue
        [255, 255, 0],    # 6: Sofa - Yellow
        [255, 0, 255],    # 7: TV - Magenta
        [0, 255, 255],    # 8: Toilet - Cyan
        [128, 0, 0],      # 9: Counter - Dark Red
        [0, 128, 0],      # 10: Sink - Dark Green
        [0, 0, 128],      # 11: Bathtub - Dark Blue
        [128, 128, 0],    # 12: Refrigerator - Olive
        [128, 0, 128],    # 13: Book - Purple
        [0, 128, 128],    # 14: Clock - Teal
        [192, 192, 192],  # 15: Vase - Light Gray
        [255, 128, 0],    # 16: Scissors - Orange
        [255, 0, 128],    # 17: Teddy Bear - Pink
        [128, 255, 0],    # 18: Hair Dryer - Lime
        [0, 255, 128],    # 19: Toothbrush - Spring Green
        [128, 0, 255],    # 20: Microwave - Violet
    ]
    
    # Extend colors if needed
    while len(colors) < num_categories:
        colors.append([np.random.randint(0, 256) for _ in range(3)])
    
    # Create RGB image
    vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(min(num_categories, len(colors))):
        mask = (category_map == i)
        vis_rgb[mask] = colors[i]
    
    if save_path:
        # Save as PNG
        img = Image.fromarray(vis_rgb)
        img.save(save_path)
        print(f"Semantic map visualization saved to: {save_path}")
    
    return vis_rgb


def save_semantic_map(semantic_map, save_path, metadata=None):
    """
    Save semantic map to HDF5 file with metadata.
    
    Args:
        semantic_map: Tensor of shape (num_categories, H, W)
        save_path: Path to save the HDF5 file
        metadata: Dictionary with additional metadata
    """
    semantic_map_np = semantic_map.cpu().numpy()
    
    with h5py.File(save_path, 'w') as f:
        # Save the semantic map
        f.create_dataset('semantic_map', data=semantic_map_np, dtype=np.float32)
        
        # Save metadata
        if metadata:
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    meta_group.create_dataset(key, data=value)
    
    print(f"Semantic map saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate predicted semantic maps using PONI's neural network")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of simulation steps")
    parser.add_argument("--save_dir", type=str, default="./predicted_maps", help="Directory to save maps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of semantic maps")
    parser.add_argument("--save_raw", action="store_true", default=True, help="Save raw semantic maps as HDF5")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize the semantic mapping module
    print("Initializing Semantic Mapping module...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create arguments for the semantic mapping module
    sem_args = create_mock_args(device=device)
    
    # Initialize the semantic mapping neural network
    sem_map_module = Semantic_Mapping(sem_args).to(device)
    sem_map_module.eval()  # Set to evaluation mode
    
    print(f"Semantic Mapping module initialized on {device}")
    print(f"Map size: {sem_args.map_size_cm}cm x {sem_args.map_size_cm}cm")
    print(f"Map resolution: {sem_args.map_resolution}cm per pixel")
    print(f"Vision range: {sem_args.vision_range} pixels")
    print(f"Number of semantic categories: {sem_args.num_sem_categories}")
    
    # Initialize map and pose states
    map_size = sem_args.map_size_cm // sem_args.map_resolution
    num_categories = 4 + sem_args.num_sem_categories  # Obstacle, explored, agent, past + semantic
    
    # Initialize global map (accumulated over time)
    global_map = torch.zeros(1, num_categories, map_size, map_size).to(device)
    
    # Initialize pose (x, y, theta in meters and radians)
    global_pose = torch.zeros(1, 3).to(device)
    
    print(f"\nStarting prediction loop for {args.num_steps} steps...")
    
    with torch.no_grad():  # No gradient computation needed for inference
        for step in range(args.num_steps):
            print(f"\nStep {step + 1}/{args.num_steps}")
            
            # Create mock RGB-D observation
            obs = create_mock_rgb_depth_observation(
                height=sem_args.frame_height,
                width=sem_args.frame_width,
                num_sem_categories=sem_args.num_sem_categories
            ).to(device)
            
            # Create mock pose change
            pose_change = create_mock_pose().to(device)
            
            print(f"  Input observation shape: {obs.shape}")
            print(f"  Pose change: dx={pose_change[0,0]:.3f}m, dy={pose_change[0,1]:.3f}m, dtheta={pose_change[0,2]:.3f}rad")
            
            # Run semantic mapping neural network
            fp_map_pred, global_map, pose_pred, current_poses = sem_map_module(
                obs, pose_change, global_map, global_pose
            )
            
            # Update global pose
            global_pose = current_poses
            
            print(f"  Predicted map shape: {global_map.shape}")
            print(f"  Current pose: x={current_poses[0,0]:.3f}m, y={current_poses[0,1]:.3f}m, theta={current_poses[0,2]:.3f}rad")
            
            # Extract semantic map (channels 4 onwards are semantic categories)
            semantic_map = global_map[0, 4:, :, :]  # Shape: (num_sem_categories, H, W)
            
            # Save raw semantic map
            if args.save_raw:
                metadata = {
                    'step': step,
                    'pose_x': float(current_poses[0, 0]),
                    'pose_y': float(current_poses[0, 1]),
                    'pose_theta': float(current_poses[0, 2]),
                    'map_size_cm': sem_args.map_size_cm,
                    'map_resolution': sem_args.map_resolution,
                    'num_categories': sem_args.num_sem_categories,
                    'vision_range': sem_args.vision_range,
                }
                
                save_path = os.path.join(args.save_dir, f"semantic_map_step_{step:03d}.h5")
                save_semantic_map(semantic_map, save_path, metadata)
            
            # Save visualization
            if args.visualize:
                vis_path = os.path.join(args.save_dir, f"semantic_map_vis_step_{step:03d}.png")
                visualize_semantic_map(semantic_map, vis_path)
            
            # Print some statistics
            obstacle_map = global_map[0, 0, :, :]  # Obstacle channel
            explored_map = global_map[0, 1, :, :]  # Explored channel
            
            total_pixels = obstacle_map.numel()
            explored_pixels = (explored_map > 0.5).sum().item()
            obstacle_pixels = (obstacle_map > 0.5).sum().item()
            
            print(f"  Explored area: {explored_pixels}/{total_pixels} pixels ({100*explored_pixels/total_pixels:.1f}%)")
            print(f"  Obstacle area: {obstacle_pixels}/{total_pixels} pixels ({100*obstacle_pixels/total_pixels:.1f}%)")
            
            # Check semantic predictions
            semantic_pixels = (semantic_map.max(dim=0)[0] > 0.1).sum().item()
            print(f"  Semantic predictions: {semantic_pixels}/{total_pixels} pixels ({100*semantic_pixels/total_pixels:.1f}%)")
    
    # Save final combined map
    print(f"\nSaving final combined map...")
    final_map_path = os.path.join(args.save_dir, "final_semantic_map.h5")
    
    final_metadata = {
        'total_steps': args.num_steps,
        'final_pose_x': float(current_poses[0, 0]),
        'final_pose_y': float(current_poses[0, 1]),
        'final_pose_theta': float(current_poses[0, 2]),
        'map_size_cm': sem_args.map_size_cm,
        'map_resolution': sem_args.map_resolution,
        'num_categories': sem_args.num_sem_categories,
    }
    
    # Save the full map (all channels)
    with h5py.File(final_map_path, 'w') as f:
        f.create_dataset('full_map', data=global_map[0].cpu().numpy(), dtype=np.float32)
        f.create_dataset('semantic_map', data=semantic_map.cpu().numpy(), dtype=np.float32)
        f.create_dataset('obstacle_map', data=global_map[0, 0].cpu().numpy(), dtype=np.float32)
        f.create_dataset('explored_map', data=global_map[0, 1].cpu().numpy(), dtype=np.float32)
        f.create_dataset('agent_map', data=global_map[0, 2].cpu().numpy(), dtype=np.float32)
        f.create_dataset('past_locations_map', data=global_map[0, 3].cpu().numpy(), dtype=np.float32)
        
        meta_group = f.create_group('metadata')
        for key, value in final_metadata.items():
            meta_group.attrs[key] = value
    
    print(f"Final map saved to: {final_map_path}")
    
    # Save final visualization
    if args.visualize:
        final_vis_path = os.path.join(args.save_dir, "final_semantic_map_vis.png")
        visualize_semantic_map(semantic_map, final_vis_path)
    
    # Save summary
    summary = {
        'experiment': {
            'num_steps': args.num_steps,
            'device': str(device),
            'save_dir': args.save_dir,
        },
        'final_pose': {
            'x': float(current_poses[0, 0]),
            'y': float(current_poses[0, 1]),
            'theta': float(current_poses[0, 2]),
        },
        'map_stats': {
            'total_pixels': int(total_pixels),
            'explored_pixels': int(explored_pixels),
            'obstacle_pixels': int(obstacle_pixels),
            'semantic_pixels': int(semantic_pixels),
            'explored_percentage': float(100 * explored_pixels / total_pixels),
            'obstacle_percentage': float(100 * obstacle_pixels / total_pixels),
            'semantic_percentage': float(100 * semantic_pixels / total_pixels),
        },
        'parameters': final_metadata,
    }
    
    summary_path = os.path.join(args.save_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {summary_path}")
    print(f"\nExperiment completed! Check {args.save_dir} for results.")


if __name__ == "__main__":
    main()
