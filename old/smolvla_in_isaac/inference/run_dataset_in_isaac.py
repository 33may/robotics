import os
import warnings
from pathlib import Path

# CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import cv2

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features


# from common import (
#     DEFAULT_DATASET_REPO_ID,
#     DEFAULT_DATASET_ROOT,
#     DEFAULT_BATCH_SIZE,
#     DEFAULT_NUM_WORKERS,
# )

# from common.isaac_utils import setup_isaac_environment, create_app_launcher



"""
Script to test trained ACT policy in Isaac Lab simulation.

This script:
1. Loads a trained ACT policy checkpoint
2. Initializes Isaac Lab lift_cube environment
3. Runs inference episodes with visualization
4. Evaluates success rate
"""
import os
import warnings
from pathlib import Path

# Set leisaac assets root to work from any directory
# This assumes script is run from /home/may33/projects/robotics
leisaac_assets = Path("/home/may33/projects/ml_portfolio/robotics/leisaac/assets")
if leisaac_assets.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(leisaac_assets)

# Suppress Isaac Sim warnings
os.environ["ISAAC_SUPPRESS_WARNINGS"] = "1"
os.environ["CARB_LOGGING_MAX_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# LeRobot imports
from lerobot.policies.act.modeling_act import ACTPolicy

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test ACT policy in Isaac Lab")
# parser.add_argument(
#     "--dataset",
#     type=str,
#     default="outputs/train/act_so101_test/checkpoints/last/pretrained_model",
#     help="Path to the trained policy checkpoint",
# )
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--livestream", type=int, default=0, help="Livestream mode (0=off 2=WebRTC)")

args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False

# -----------------------------
# Setup streaming and cameras
if args_cli.livestream == 2:
    args_cli.headless = True

    sys.argv.append("--/app/livestream/publicEndpointAddress=100.115.105.111")
    sys.argv.append("--/app/livestream/port=49100")

    print(f"[DEBUG] Livestream enabled (WebRTC mode)")
    print(f"[DEBUG] Stream address: 100.115.105.111:49100")
    print(f"[DEBUG] headless = {args_cli.headless}")

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Lab imports (must be after AppLauncher)
from isaaclab.envs import ManagerBasedRLEnv

from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg


dataset_id = "eternalmay33/pick_place_test"

dataset_meta = LeRobotDatasetMetadata(dataset_id)

print(dataset_meta)

features = dataset_to_policy_features(dataset_meta.features)

output_features = {key : feature for key, feature in features.items() if feature.type is FeatureType.ACTION}
input_features = {key : feature for key, feature in features.items() if feature.type is not FeatureType.ACTION}

print(input_features)
print(output_features)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

delta_timestamps = {
    "action" : make_delta_timestamps(
        list(range(20)),
        30
    )
}


delta_timestamps |= {
    k: make_delta_timestamps([-2, -1, 0], 30)
    for k in input_features
}

dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

print(dataset)


env_cfg = LiftCubeEnvCfg()

env_cfg.scene.num_envs = 1  # Single environment for testing

# Configure actions for joint position control (same as teleoperation with SO-101 leader)
env_cfg.use_teleop_device("so101leader")

env = ManagerBasedRLEnv(cfg=env_cfg)

def deg_to_rad(action_tensor):
    return action_tensor / 180 * np.pi


while True:

    obs = env.reset()

    for i in range(10000):
        actions = dataset[i]["action"]

        act_to_exec = actions[0]

        action_tensor = torch.tensor(act_to_exec, device=env.device)

        action_tensor = action_tensor.unsqueeze(0)

        # print(action_tensor)

        action_tensor = deg_to_rad(action_tensor)

        print(action_tensor)

        obs, reward, terminated, truncated, info = env.step(action_tensor)

        env.step(action_tensor)

        # print(act_to_exec)

        print("cycle")

    



