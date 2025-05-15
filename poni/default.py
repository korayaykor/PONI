from typing import List, Optional, Union

import yacs.config


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config
CONFIG_FILE_SEPARATOR = ","

_C = CN()

_C.SEED = 123

# ==================================== for sensor =======================
_C.SENSOR = CN()
_C.SENSOR.DEPTH_MIN = 0.0
_C.SENSOR.DEPTH_MAX = 5.0
_C.SENSOR.SENSOR_HEIGHT = 1.25
_C.SENSOR.AGENT_HEIGHT = 1.5
_C.SENSOR.AGENT_RADIUS = 0.1


# ================================ for semantic map ===============================
_C.SEM_MAP = CN()
_C.SEM_MAP.ENLARGE_SIZE = 10
_C.SEM_MAP.IGNORED_MAP_CLASS = [0, ]
# for semantic segmentation, class 17 is ceiling
_C.SEM_MAP.IGNORED_SEM_CLASS = [0, 1]
_C.SEM_MAP.OBJECT_MASK_PIXEL_THRESH = 100
# explored but semantic-unrecognized pixel
_C.SEM_MAP.UNDETECTED_PIXELS_CLASS = 41
_C.SEM_MAP.CELL_SIZE = 0.05
# world model size in each dimension (left, right, top , bottom)
_C.SEM_MAP.WORLD_SIZE = 50.0
#_C.SEM_MAP.GRID_Y_SIZE = 60
_C.SEM_MAP.GRID_CLASS_SIZE = 42
_C.SEM_MAP.HABITAT_FLOOR_IDX = 2
_C.SEM_MAP.POINTS_CNT = 2
# complement the gap between the robot neighborhood and the projected occupancy map
_C.SEM_MAP.GAP_COMPLEMENT = 10

_C.FE = CN()
_C.FE.COLLISION_VAL = 1
_C.FE.FREE_VAL = 2
_C.FE.UNOBSERVED_VAL = 0
_C.FE.OBSTACLE_THRESHOLD = 1


# ================================ for visualization ============================
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = True


# ================================ for slurm ==============================
_C.SLURM = CN()
_C.SLURM.NUM_PROCESS = 1
_C.SLURM.PROC_PER_GPU = 1

# ==================================== for sensor =======================

_C.MODEL = CN()
_C.MODEL.num_categories = 17
_C.MODEL.nsf = 32
_C.MODEL.unet_bilinear_interp = True
_C.MODEL.object_loss_type = "l2"  # options: bce, l1, l2, xent
_C.MODEL.area_loss_type = "l2"  # options: bce, l1, l2, xent
_C.MODEL.object_activation = "sigmoid"  # options: sigmoid, relu, none
_C.MODEL.area_activation = "sigmoid"  # options: sigmoid, relu, none
_C.MODEL.embedding_size = 64
_C.MODEL.map_size = 480
_C.MODEL.pretrained_path = ""
################################################################################
# These are set automatically by the code
_C.MODEL.output_type = "map"  # options: map, dirs, locs
_C.MODEL.ndirs = 8
_C.MODEL.enable_area_head = False
################################################################################

_C.OPTIM = CN()
_C.OPTIM.lr = 1e-3
_C.OPTIM.num_total_updates = 40000
_C.OPTIM.batch_size = 20
_C.OPTIM.num_workers = 0
_C.OPTIM.lr_sched_milestones = [
    2,
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
_C.DATASET.root = "data/semantic_maps/gibson/semantic_maps/"
_C.DATASET.dset_name = "gibson"
_C.DATASET.seed = 123
_C.DATASET.output_map_size = 24.0  # meters
_C.DATASET.masking_mode = "spath"  # options: [spath]
_C.DATASET.masking_shape = "square"  # options: [square]
_C.DATASET.visibility_size = 3.0  # m
_C.DATASET.dilate_free_map = True
_C.DATASET.dilate_iters = 1
_C.DATASET.object_pf_cutoff_dist = 10.0
_C.DATASET.potential_function_masking = True
_C.DATASET.potential_function_frontier_scaling = 1.0
_C.DATASET.potential_function_non_visible_scaling = 0.0
_C.DATASET.potential_function_non_frontier_scaling = 0.0
_C.DATASET.fmm_dists_saved_root = "data/semantic_maps/gibson/fmm_dists_{}_{}".format(
    _C.DATASET.output_map_size,
    _C.DATASET.seed,
)
# Unexplored area prediction
_C.DATASET.enable_unexp_area = False
_C.DATASET.normalize_area_by_constant = False
_C.DATASET.max_unexp_area = 60.0
# Baselines
_C.DATASET.enable_directions = False
_C.DATASET.prediction_directions = [0, 45, 90, 135, 180, 225, 270, 315]
_C.DATASET.enable_locations = False
# Predict actions
_C.DATASET.enable_actions = False
_C.DATASET.num_actions = 4
_C.DATASET.turn_angle = 30


def get_config(
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

    config.freeze()
    return config
