from typing import List, Optional, Union

import yacs.config

# It's good practice to import your constants if you need to derive values from them here,
# e.g., to set a default num_categories based on a chosen dset_name.
# However, for yacs, direct complex logic in defaults is tricky.
# Usually, these would be updated post-config load or via command-line opts.
# For now, we'll make num_categories a general placeholder that should be overridden.
# from poni.constants import NUM_OBJECT_CATEGORIES


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config
CONFIG_FILE_SEPARATOR = ","

_C = CN()

_C.SEED = 123

_C.MODEL = CN()
# !!! ACTION REQUIRED: This should be overridden based on the selected dataset !!!
# For example, if using HM3D, set this to NUM_OBJECT_CATEGORIES['hm3d'] from poni/constants.py
# The current value is for Gibson/MP3D.
# Original values were 17 (Gibson) or 23 (MP3D).
_C.MODEL.num_categories = 23  # Needs to be updated based on the dataset (e.g., for HM3D)
_C.MODEL.nsf = 32
_C.MODEL.unet_bilinear_interp = True
_C.MODEL.object_loss_type = "l2"  # options: bce, l1, l2, xent
_C.MODEL.area_loss_type = "l2"  # options: bce, l1, l2, xent
_C.MODEL.object_activation = "sigmoid"  # options: sigmoid, relu, none
_C.MODEL.area_activation = "sigmoid"  # options: sigmoid, relu, none
_C.MODEL.embedding_size = 64
_C.MODEL.map_size = 480  # This likely refers to the input map size to the model, not the world map size.
_C.MODEL.pretrained_path = ""
################################################################################
# These are set automatically by the code in train.py based on other DATASET flags
_C.MODEL.output_type = "map"  # options: map, dirs, locs, acts
_C.MODEL.ndirs = 8
_C.MODEL.enable_area_head = False
################################################################################

_C.OPTIM = CN()
_C.OPTIM.lr = 1e-3
_C.OPTIM.num_total_updates = 40000
_C.OPTIM.batch_size = 20
_C.OPTIM.num_workers = 8
_C.OPTIM.lr_sched_milestones = [
    20000, # Example: decay LR at 20k and 30k steps
    30000,
]
_C.OPTIM.lr_sched_gamma = 0.1

_C.LOGGING = CN()
_C.LOGGING.log_dir = "./"
_C.LOGGING.tb_dir = "./"
_C.LOGGING.ckpt_dir = "./checkpoints"
_C.LOGGING.log_interval = 10
_C.LOGGING.eval_interval = 1000
_C.LOGGING.ckpt_interval = 1000
_C.LOGGING.verbose = False

_C.DATASET = CN()
# !!! ACTION REQUIRED: Update these paths for HM3D !!!
# Option 1: Set a generic placeholder and always override via command line or specific config files.
# Option 2: Set to a sensible default if 'hm3d' is the primary new dataset.
_C.DATASET.root = "data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square" # Placeholder, override for HM3D
_C.DATASET.dset_name = "gibson"  # Default dataset, override to "hm3d" when using HM3D
_C.DATASET.seed = 123 # This is likely the seed for dataset generation randomness.
_C.DATASET.output_map_size = 24.0  # meters, size of the map patch PONI operates on
_C.DATASET.masking_mode = "spath"  # options: [spath]
_C.DATASET.masking_shape = "square"  # options: [square]
_C.DATASET.visibility_size = 3.0  # m
_C.DATASET.dilate_free_map = True
_C.DATASET.dilate_iters = 1
_C.DATASET.object_pf_cutoff_dist = 10.0
_C.DATASET.potential_function_masking = True # PONI-specific
_C.DATASET.potential_function_frontier_scaling = 1.0
_C.DATASET.potential_function_non_visible_scaling = 0.0
_C.DATASET.potential_function_non_frontier_scaling = 0.0 # Was 0.0 in original, 0.2 in some experiments

# !!! ACTION REQUIRED: Update this path for HM3D FMM distances !!!
# This should point to where your precomputed FMM distances for HM3D are stored.
# The name often includes output_map_size and seed for clarity.
_C.DATASET.fmm_dists_saved_root = "data/semantic_maps/gibson/fmm_dists_24.0_123" # Placeholder, override for HM3D

# Unexplored area prediction
_C.DATASET.enable_unexp_area = False # Set to True for PONI method
_C.DATASET.normalize_area_by_constant = False
_C.DATASET.max_unexp_area = 60.0 # Example: 40.0 for mp3d
# Baselines
_C.DATASET.enable_directions = False
_C.DATASET.prediction_directions = [0, 45, 90, 135, 180, 225, 270, 315]
_C.DATASET.enable_locations = False
# Predict actions
_C.DATASET.enable_actions = False
_C.DATASET.num_actions = 4 # Typically STOP, FORWARD, TURN_LEFT, TURN_RIGHT
_C.DATASET.turn_angle = 30 # Degrees, relevant if enable_actions is True


def get_cfg(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    # Post-processing: Update num_categories if dset_name is changed and constants are available
    # This is a common pattern but requires poni.constants to be importable here.
    # If direct import causes issues (e.g. circular dependency if constants also imports default),
    # this logic might need to be in the main script after loading config.
    try:
        from poni.constants import NUM_OBJECT_CATEGORIES
        if config.DATASET.dset_name in NUM_OBJECT_CATEGORIES:
            config.defrost()
            config.MODEL.num_categories = NUM_OBJECT_CATEGORIES[config.DATASET.dset_name]
            # For PONI, the actual number of output channels for the object decoder
            # might be num_categories - 2 (if floor and wall are handled separately or not predicted by object decoder)
            # Or it might be num_categories if the model predicts all of them.
            # The original train.py logic for "map" output_type:
            # y_hat = object_preds[:, 2:]
            # This implies the object decoder outputs C channels, and the first 2 are ignored if they are floor/wall.
            # So, model.num_categories should indeed be the total number of categories including floor/wall
            # if the decoder is designed to predict all of them and then they are sliced.
            # The current PONI model.py UNetDecoder(model_cfg.num_categories, ...)
            # means it outputs model_cfg.num_categories channels.
            # Let's assume model_cfg.num_categories is the total number of distinct semantic classes
            # the input maps can have, which PONI will learn to process.
            # In `poni/dataset.py`, `convert_maps_to_oh` creates N one-hot channels
            # where N = NUM_OBJECT_CATEGORIES[dset]. The `CAT_OFFSET` (usually 2) is used
            # such that `semmap_oh[i]` corresponds to the (i+CAT_OFFSET)-th category from `OBJECT_CATEGORY_MAP`.
            # This means if `NUM_OBJECT_CATEGORIES['hm3d']` is 17, then `semmap_oh` will have 17 channels.
            # The input to the UNetEncoder in `poni/model.py` is `model_cfg.num_categories`.
            # This means `_C.MODEL.num_categories` should be set to `NUM_OBJECT_CATEGORIES[config.DATASET.dset_name]`.

            # If `enable_unexp_area` is true, this also influences model architecture.
            if config.DATASET.enable_unexp_area:
                 config.MODEL.enable_area_head = True # This is already handled in train.py init


            # Update output type based on dataset flags, as done in train.py
            if config.DATASET.enable_directions:
                assert config.MODEL.object_loss_type == "xent"
                assert config.MODEL.object_activation == "none"
                config.MODEL.output_type = "dirs"
                config.MODEL.ndirs = len(config.DATASET.prediction_directions)
            elif config.DATASET.enable_locations:
                assert config.MODEL.object_loss_type in ["l1", "l2"]
                assert config.MODEL.object_activation == "sigmoid"
                config.MODEL.output_type = "locs"
            elif config.DATASET.enable_actions:
                assert config.MODEL.object_loss_type == "xent"
                assert config.MODEL.object_activation == "none"
                config.MODEL.output_type = "acts"
            else: # Default map prediction
                config.MODEL.output_type = "map"

            config.freeze()
    except ImportError:
        print("Warning: poni.constants not found, MODEL.num_categories might need manual override.")
        pass


    config.freeze()
    return config