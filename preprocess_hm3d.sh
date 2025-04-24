#!/bin/bash

# Script to preprocess HM3D dataset for PONI
# This script handles error checking and environment setup

set -e  # Exit on error

# Check PONI_ROOT environment variable
if [ -z "$PONI_ROOT" ]; then
    echo "ERROR: PONI_ROOT environment variable is not set"
    echo "Please set it with: export PONI_ROOT=<path to PONI>"
    exit 1
fi

# Check for required Python modules
python3 -c "import habitat_sim, trimesh, h5py" 2>/dev/null || {
    echo "ERROR: Missing required Python modules"
    echo "Please make sure habitat_sim, trimesh, and h5py are installed"
    echo "You can install them with: pip install habitat-sim trimesh h5py"
    exit 1
}

# Path to HM3D dataset (adjust as needed)
HM3D_PATH=${1:-"/app/PONI/data/scene_datasets/hm3d"}
SEMANTIC_PATH=${2:-"$PONI_ROOT/data/semantic_maps/hm3d"}

# Create the improved script
SCRIPT_PATH="$PONI_ROOT/scripts/improved_preprocess_hm3d.py"

# Check if improved script exists already
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Creating improved HM3D preprocessing script at $SCRIPT_PATH"