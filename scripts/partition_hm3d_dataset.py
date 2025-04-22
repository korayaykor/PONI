#!/usr/bin/env python3
"""
Script to partition HM3D dataset into multiple parts for parallel evaluation.
"""

import os
import json
import argparse
import gzip
import shutil
from collections import defaultdict


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Partition HM3D dataset for parallel evaluation"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the partitioned dataset",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=11,
        help="Number of parts to split the dataset into",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to partition",
    )
    return parser


def load_dataset(dataset_path):
    """Load dataset from JSON or GZ file."""
    if dataset_path.endswith(".gz"):
        with gzip.open(dataset_path, "rt") as f:
            return json.load(f)
    else:
        with open(dataset_path, "r") as f:
            return json.load(f)


def save_dataset(data, output_path):
    """Save dataset to JSON or GZ file."""
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wt") as f:
            json.dump(data, f)
    else:
        with open(output_path, "w") as f:
            json.dump(data, f)


def partition_dataset(dataset, num_parts):
    """Partition dataset by scene."""
    # Group episodes by scene
    episodes_by_scene = defaultdict(list)
    for episode in dataset["episodes"]:
        scene_id = episode["scene_id"]
        episodes_by_scene[scene_id].append(episode)
    
    # Sort scenes by number of episodes (descending)
    scenes_sorted = sorted(
        episodes_by_scene.keys(),
        key=lambda x: len(episodes_by_scene[x]),
        reverse=True
    )
    
    # Initialize partitions
    partitions = [[] for _ in range(num_parts)]
    partition_sizes = [0] * num_parts
    
    # Distribute scenes to partitions (greedy bin packing)
    for scene_id in scenes_sorted:
        episodes = episodes_by_scene[scene_id]
        # Find partition with minimum episodes
        min_idx = partition_sizes.index(min(partition_sizes))
        partitions[min_idx].extend(episodes)
        partition_sizes[min_idx] += len(episodes)
    
    return partitions


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.split}_parts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Partition dataset
    partitions = partition_dataset(dataset, args.num_parts)
    
    # Save partitions
    for i, episodes in enumerate(partitions):
        part_dir = os.path.join(output_dir, f"val_part_{i}")
        os.makedirs(part_dir, exist_ok=True)
        
        part_dataset = {
            "episodes": episodes,
        }
        # Copy other fields from original dataset
        for key, value in dataset.items():
            if key != "episodes":
                part_dataset[key] = value
        
        output_path = os.path.join(part_dir, f"val_part_{i}.json.gz")
        save_dataset(part_dataset, output_path)
        
        print(f"Part {i}: {len(episodes)} episodes")
    
    print(f"Dataset partitioned into {args.num_parts} parts in {output_dir}")


if __name__ == "__main__":
    main()