
import os
import glob
import json
from pathlib import Path
import random

# Seed for reproducibility
random.seed(42)

def find_hm3d_scenes_with_semantic():
    """Find all HM3D scenes that have semantic GLB files"""
    
    # Try different possible paths
    possible_roots = [
        "/app/PONI/data/scene_datasets/hm3d_uncompressed",
        "data/scene_datasets/hm3d_uncompressed",
        "data/scene_datasets/hm3d",
    ]
    
    scenes_with_semantic = []
    
    for root in possible_roots:
        if os.path.exists(root):
            print(f"Checking {root}")
            
            # Find all basis.glb files
            glb_files = list(Path(root).glob("**/*.basis.glb"))
            
            for glb_file in glb_files:
                # Check if corresponding semantic.glb exists
                semantic_glb = str(glb_file).replace(".basis.glb", ".semantic.glb")
                
                if os.path.exists(semantic_glb):
                    # Extract scene ID
                    # For paths like: /path/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb
                    scene_name = glb_file.stem.replace(".basis", "")
                    scenes_with_semantic.append(scene_name)
            
            if scenes_with_semantic:
                break
    
    return sorted(list(set(scenes_with_semantic)))

def create_train_val_split(scenes, train_ratio=0.8):
    """Split scenes into train and val sets"""
    
    # Shuffle scenes
    shuffled_scenes = scenes.copy()
    random.shuffle(shuffled_scenes)
    
    # Calculate split
    num_train = int(len(scenes) * train_ratio)
    
    train_scenes = shuffled_scenes[:num_train]
    val_scenes = shuffled_scenes[num_train:]
    
    return train_scenes, val_scenes

def main():
    # Find all scenes with semantic GLB files
    scenes = find_hm3d_scenes_with_semantic()
    
    print(f"Found {len(scenes)} scenes with semantic GLB files")
    
    if len(scenes) == 0:
        print("No scenes found! Please check the dataset path.")
        return
    
    print("\nFirst 10 scenes:")
    for i, scene in enumerate(scenes[:10]):
        print(f"  {i+1}: {scene}")
    
    # Create train/val split
    train_scenes, val_scenes = create_train_val_split(scenes)
    
    print(f"\nSplit: {len(train_scenes)} train, {len(val_scenes)} val")
    
    # Create the split dictionary
    hm3d_split = {
        "train": train_scenes,
        "val": val_scenes
    }
    
    # Print in the format needed for poni/constants.py
    print("\n=== Split for poni/constants.py ===")
    print('"hm3d": {')
    print('    "train": [')
    for scene in train_scenes:
        print(f'        "{scene}",')
    print('    ],')
    print('    "val": [')
    for scene in val_scenes:
        print(f'        "{scene}",')
    print('    ],')
    print('},')
    
    # Save to JSON file
    output_file = "hm3d_split.json"
    with open(output_file, "w") as f:
        json.dump(hm3d_split, f, indent=4)
    
    print(f"\nSplit saved to {output_file}")
    
    # Also save a Python file that can be directly imported
    python_output = "hm3d_split.py"
    with open(python_output, "w") as f:
        f.write("# HM3D train/val split\n")
        f.write("HM3D_SPLIT = {\n")
        f.write('    "train": [\n')
        for scene in train_scenes:
            f.write(f'        "{scene}",\n')
        f.write('    ],\n')
        f.write('    "val": [\n')
        for scene in val_scenes:
            f.write(f'        "{scene}",\n')
        f.write('    ],\n')
        f.write('}\n')
    
    print(f"Python split saved to {python_output}")

if __name__ == "__main__":
    main()