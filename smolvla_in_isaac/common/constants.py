"""
Constants for Isaac Lab to LeRobot data transformations.

This module centralizes all constants used across training, inference, and data conversion scripts.
"""
from pathlib import Path


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DEFAULT_DATASET_REPO_ID = "eternalmay33/pick_place_test"
DEFAULT_DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot"


# =============================================================================
# JOINT POSITION RANGES
# =============================================================================

# Isaac Lab joint position limits (in degrees)
# These represent the physical limits of the SO-101 robot arm
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),   # shoulder_pan
    (-100.0, 100.0),   # shoulder_lift
    (-100.0, 90.0),    # elbow_flex
    (-95.0, 95.0),     # wrist_flex
    (-160.0, 160.0),   # wrist_roll
    (-10.0, 100.0),    # gripper (0 = closed, 100 = open)
]

# LeRobot normalized joint position ranges (target range after conversion)
# All joints except gripper are normalized to [-100, 100]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),   # shoulder_pan
    (-100, 100),   # shoulder_lift
    (-100, 100),   # elbow_flex
    (-100, 100),   # wrist_flex
    (-100, 100),   # wrist_roll
    (0, 100),      # gripper (always positive)
]

# Joint names for reference
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

# Camera names used in the dataset
CAMERA_KEYS = [
    "observation.images.front",
    "observation.images.third_person",
    "observation.images.gripper",
]

# Isaac Lab camera observation keys (different naming)
ISAAC_CAMERA_KEYS = {
    "front": "observation.images.front",
    "front_cam_cfg": "observation.images.third_person",
    "gripper_cam_cfg": "observation.images.gripper",
}

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_CHANNELS = 3
CAMERA_FPS = 30.0


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

# Train/validation split
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42

# Common hyperparameters
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4


# =============================================================================
# NORMALIZATION STATISTICS PATH
# =============================================================================

def get_stats_path(dataset_repo_id: str = DEFAULT_DATASET_REPO_ID, root: Path | None = None) -> Path:
    """Get path to normalization statistics file."""
    if root is None:
        root = DEFAULT_DATASET_ROOT
    return root / dataset_repo_id / "meta" / "stats.json"


# =============================================================================
# INFERENCE DEFAULTS
# =============================================================================

DEFAULT_NUM_EPISODES = 10
DEFAULT_MAX_STEPS = 500
DEFAULT_HEADLESS = False
DEFAULT_LIVESTREAM = 0  # 0=off, 2=WebRTC
