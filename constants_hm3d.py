import numpy as np
from poni.constants import d3_40_colors_rgb

# Add HM3D to registered datasets
REGISTERED_DATASETS = ["gibson", "mp3d", "hm3d"]

# HM3D scene splits (you'll need to populate these with actual scene IDs)
# These are placeholder scene names that should be replaced with actual HM3D scene IDs
SPLIT_SCENES = {
    "hm3d": {
        "train": [
            # Add training scenes here
            "train_00",
            "train_01",
            "train_02",
            # ... more training scenes
        ],
        "val": [
            # Add validation scenes here
            "val_00",
            "val_01",
            "val_02",
            # ... more validation scenes
        ],
    }
}

# HM3D object categories
# Based on commonly found categories in HM3D, similar to MP3D categories
OBJECT_CATEGORIES = {
    "hm3d": [
        "floor",
        "wall",
        "chair",
        "table",
        "picture",
        "cabinet",
        "cushion",
        "sofa",
        "bed",
        "chest_of_drawers",
        "plant",
        "sink",
        "toilet",
        "stool",
        "towel",
        "tv_monitor",
        "shower",
        "bathtub",
        "counter",
        "fireplace",
        "gym_equipment",
        "seating",
        "clothes",
    ],
}

# Object category mapping
OBJECT_CATEGORY_MAP = {}
INV_OBJECT_CATEGORY_MAP = {}
NUM_OBJECT_CATEGORIES = {}
for dset, categories in OBJECT_CATEGORIES.items():
    OBJECT_CATEGORY_MAP[dset] = {obj: idx for idx, obj in enumerate(categories)}
    INV_OBJECT_CATEGORY_MAP[dset] = {v: k for k, v in OBJECT_CATEGORY_MAP[dset].items()}
    NUM_OBJECT_CATEGORIES[dset] = len(categories)

# HM3D color palette
HM3D_OBJECT_COLORS = []  # Excluding 'out-of-bounds', 'floor', and 'wall'
for color in d3_40_colors_rgb[:len(OBJECT_CATEGORIES["hm3d"]) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    HM3D_OBJECT_COLORS.append(color)