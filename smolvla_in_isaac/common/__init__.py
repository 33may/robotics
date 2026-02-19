"""
Common utilities for Isaac Lab <-> LeRobot integration.

This package provides shared functionality for training and inference:
- Constants: Joint limits, dataset paths, camera config
- Transformations: Data conversion between Isaac Lab and LeRobot formats
- Isaac utilities: Environment setup and management
"""

# Constants
from .constants import (
    # Dataset
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_ROOT,
    # Joint ranges
    ISAACLAB_JOINT_POS_LIMIT_RANGE,
    LEROBOT_JOINT_POS_LIMIT_RANGE,
    JOINT_NAMES,
    # Camera config
    CAMERA_KEYS,
    ISAAC_CAMERA_KEYS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_CHANNELS,
    CAMERA_FPS,
    # Training defaults
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_RANDOM_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    # Inference defaults
    DEFAULT_NUM_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_HEADLESS,
    DEFAULT_LIVESTREAM,
    # Functions
    get_stats_path,
)

# Transformations
from .transformations import (
    load_normalization_stats,
    preprocess_isaac_to_lerobot,
    postprocess_lerobot_to_isaac,
    prepare_observation,
    preprocess_joint_pos_batch,
)

# Isaac Lab utilities
from .isaac_utils import (
    setup_isaac_environment,
    create_app_launcher,
    create_lift_cube_environment,
    print_environment_info,
    safe_close_environment,
)


__all__ = [
    # Constants
    "DEFAULT_DATASET_REPO_ID",
    "DEFAULT_DATASET_ROOT",
    "ISAACLAB_JOINT_POS_LIMIT_RANGE",
    "LEROBOT_JOINT_POS_LIMIT_RANGE",
    "JOINT_NAMES",
    "CAMERA_KEYS",
    "ISAAC_CAMERA_KEYS",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CAMERA_CHANNELS",
    "CAMERA_FPS",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_NUM_EPISODES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_HEADLESS",
    "DEFAULT_LIVESTREAM",
    "get_stats_path",
    # Transformations
    "load_normalization_stats",
    "preprocess_isaac_to_lerobot",
    "postprocess_lerobot_to_isaac",
    "prepare_observation",
    "preprocess_joint_pos_batch",
    # Isaac utilities
    "setup_isaac_environment",
    "create_app_launcher",
    "create_lift_cube_environment",
    "print_environment_info",
    "safe_close_environment",
]
