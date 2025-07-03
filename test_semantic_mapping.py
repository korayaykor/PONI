#!/usr/bin/env python3
"""
Simple test script to verify PONI's Semantic_Mapping neural network module is working correctly.

This script runs a basic test of the semantic mapping functionality without saving files.
Use this to quickly check if the neural network is properly configured and functional.

Usage:
    python test_semantic_mapping.py
"""

import torch
import numpy as np
from argparse import Namespace

# Import PONI's semantic mapping module
try:
    from semexp.model import Semantic_Mapping
    print("✓ Successfully imported Semantic_Mapping from semexp.model")
except ImportError as e:
    print(f"✗ Failed to import Semantic_Mapping: {e}")
    print("Make sure PONI is properly installed and PYTHONPATH is set correctly.")
    exit(1)


def create_test_args(device="cuda:0"):
    """Create minimal arguments for testing"""
    args = Namespace()
    args.device = device
    args.num_processes = 1
    args.frame_height = 480
    args.frame_width = 640
    args.hfov = 79.0
    args.map_resolution = 5
    args.map_size_cm = 1200  # Smaller for testing
    args.camera_height = 88.0
    args.global_downscaling = 1
    args.vision_range = 100
    args.du_scale = 1
    args.cat_pred_threshold = 5.0
    args.exp_pred_threshold = 1.0
    args.map_pred_threshold = 1.0
    args.num_sem_categories = 21
    return args


def test_semantic_mapping():
    """Run basic test of semantic mapping functionality"""
    print("\n" + "="*60)
    print("PONI Neural Semantic Mapping Test")
    print("="*60)
    
    # Check device availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create test arguments
    args = create_test_args(device=device)
    print(f"\nInitializing Semantic_Mapping with parameters:")
    print(f"  Map size: {args.map_size_cm}cm x {args.map_size_cm}cm")
    print(f"  Resolution: {args.map_resolution}cm/pixel")
    print(f"  Frame size: {args.frame_width}x{args.frame_height}")
    print(f"  Semantic categories: {args.num_sem_categories}")
    
    try:
        # Initialize semantic mapping module
        sem_map_module = Semantic_Mapping(args).to(device)
        sem_map_module.eval()
        print("✓ Semantic_Mapping module initialized successfully")
        
        # Print model parameters
        total_params = sum(p.numel() for p in sem_map_module.parameters())
        print(f"  Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Failed to initialize Semantic_Mapping: {e}")
        return False
    
    # Create test data
    print("\nCreating test input data...")
    batch_size = 1
    num_channels = 4 + args.num_sem_categories  # RGB + Depth + Semantic
    
    try:
        # Mock RGB-D observation with semantic predictions
        obs = torch.rand(batch_size, num_channels, args.frame_height, args.frame_width).to(device)
        # Make depth values realistic (0.5-5.0 meters)
        obs[:, 3, :, :] = obs[:, 3, :, :] * 4.5 + 0.5
        
        # Mock pose change (small movement)
        pose_change = torch.tensor([[0.1, 0.0, 0.05]]).to(device)  # dx, dy, dtheta
        
        # Initialize map and pose
        map_size = args.map_size_cm // args.map_resolution
        full_channels = 4 + args.num_sem_categories
        global_map = torch.zeros(batch_size, full_channels, map_size, map_size).to(device)
        global_pose = torch.zeros(batch_size, 3).to(device)
        
        print("✓ Test data created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Pose change: {pose_change[0].tolist()}")
        print(f"  Map shape: {global_map.shape}")
        
    except Exception as e:
        print(f"✗ Failed to create test data: {e}")
        return False
    
    # Run semantic mapping forward pass
    print("\nRunning semantic mapping forward pass...")
    
    try:
        with torch.no_grad():
            fp_map_pred, updated_map, pose_pred, current_poses = sem_map_module(
                obs, pose_change, global_map, global_pose
            )
        
        print("✓ Forward pass completed successfully")
        print(f"  Output map shape: {updated_map.shape}")
        print(f"  Local prediction shape: {fp_map_pred.shape}")
        print(f"  Updated pose: {current_poses[0].tolist()}")
        
        # Check output validity
        if torch.isnan(updated_map).any():
            print("✗ Warning: NaN values detected in output map")
            return False
        
        if torch.isinf(updated_map).any():
            print("✗ Warning: Infinite values detected in output map")
            return False
        
        print("✓ Output values are valid (no NaN or Inf)")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Analyze outputs
    print("\nAnalyzing outputs...")
    
    try:
        # Extract different map channels
        obstacle_map = updated_map[0, 0, :, :]
        explored_map = updated_map[0, 1, :, :]
        semantic_maps = updated_map[0, 4:, :, :]
        
        # Calculate statistics
        total_pixels = obstacle_map.numel()
        explored_pixels = (explored_map > 0.1).sum().item()
        obstacle_pixels = (obstacle_map > 0.1).sum().item()
        semantic_pixels = (semantic_maps.max(dim=0)[0] > 0.1).sum().item()
        
        print(f"  Map statistics:")
        print(f"    Total pixels: {total_pixels}")
        print(f"    Explored pixels: {explored_pixels} ({100*explored_pixels/total_pixels:.1f}%)")
        print(f"    Obstacle pixels: {obstacle_pixels} ({100*obstacle_pixels/total_pixels:.1f}%)")
        print(f"    Semantic pixels: {semantic_pixels} ({100*semantic_pixels/total_pixels:.1f}%)")
        
        # Check if reasonable updates occurred
        if explored_pixels > 0:
            print("✓ Exploration mapping is working")
        else:
            print("? Warning: No explored areas detected")
        
        if semantic_pixels > 0:
            print("✓ Semantic mapping is working")
        else:
            print("? Warning: No semantic predictions detected")
        
    except Exception as e:
        print(f"✗ Failed to analyze outputs: {e}")
        return False
    
    # Test multiple steps
    print("\nTesting multiple sequential steps...")
    
    try:
        with torch.no_grad():
            for step in range(3):
                # Create slightly different observation
                obs_new = torch.rand(batch_size, num_channels, args.frame_height, args.frame_width).to(device)
                obs_new[:, 3, :, :] = obs_new[:, 3, :, :] * 4.5 + 0.5
                
                # Small pose change
                pose_change_new = torch.tensor([[0.05, 0.02, 0.03]]).to(device)
                
                # Update map
                fp_map_pred, updated_map, pose_pred, current_poses = sem_map_module(
                    obs_new, pose_change_new, updated_map, current_poses
                )
                
                print(f"  Step {step+1}: Pose = {current_poses[0].tolist()}")
        
        print("✓ Multi-step processing works correctly")
        
    except Exception as e:
        print(f"✗ Multi-step processing failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! Semantic mapping is working correctly.")
    print("="*60)
    print("\nYou can now run the full scripts:")
    print("  python create_predicted_semantic_maps.py")
    print("  python visualize_predicted_maps.py")
    
    return True


if __name__ == "__main__":
    success = test_semantic_mapping()
    exit(0 if success else 1)
