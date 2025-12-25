"""
Script to test trained SmolVLA policy in Isaac Lab simulation.

This script:
1. Loads a trained SmolVLA policy checkpoint
2. Initializes Isaac Lab lift_cube environment
3. Runs inference episodes with visualization
4. Evaluates success rate
"""
import os
import warnings
from pathlib import Path

# CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set leisaac assets root to work from any directory
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
import gc
import cv2

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test SmolVLA policy in Isaac Lab")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="outputs/finetune/smolvla_pick_place/checkpoints/best/pretrained_model",
    help="Path to the trained policy checkpoint",
)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of test episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--livestream", type=int, default=2, help="Livestream mode (0=off 2=WebRTC)")
parser.add_argument("--video_path", type=str, default="smolvla_test.mp4", help="Path to save video")

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Force headless if not specified
if not hasattr(args_cli, 'headless') or not args_cli.headless:
    args_cli.headless = True

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

# Import environment config
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg


# Load normalization stats from dataset
dataset_id = "eternalmay33/pick_place_test"
STATS_PATH = Path.home() / f".cache/huggingface/lerobot/{dataset_id}/meta/stats.json"
with open(STATS_PATH, 'r') as f:
    STATS = json.load(f)

# Extract mean/std for normalization
OBS_STATE_MEAN = np.array(STATS["observation.state"]["mean"])
OBS_STATE_STD = np.array(STATS["observation.state"]["std"])
ACTION_MEAN = np.array(STATS["action"]["mean"])
ACTION_STD = np.array(STATS["action"]["std"])

print(f"[INFO] Loaded stats from {STATS_PATH}")
print(f"[INFO] observation.state mean: {OBS_STATE_MEAN}")
print(f"[INFO] observation.state std: {OBS_STATE_STD}")
print(f"[INFO] action mean: {ACTION_MEAN}")
print(f"[INFO] action std: {ACTION_STD}")

# Joint position preprocessing functions (inverse of training conversion)
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_isaac_to_lerobot(joint_pos: np.ndarray) -> np.ndarray:
    """
    Convert joint positions from Isaac Lab to LeRobot format with MEAN_STD normalization.

    Steps:
    1. Convert radians to degrees
    2. Map from Isaac range to LeRobot range
    3. Apply MEAN_STD normalization (x - mean) / std
    """
    # Step 1: Convert radians to degrees
    joint_pos = joint_pos / np.pi * 180

    # # Step 2: Map from Isaac range to LeRobot range
    # for i in range(6):
    #     isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
    #     lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
    #     isaac_range = isaaclab_max - isaaclab_min
    #     lerobot_range = lerobot_max - lerobot_min
    #     joint_pos[i] = (joint_pos[i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min

    # # Step 3: Apply MEAN_STD normalization
    # joint_pos = (joint_pos - OBS_STATE_MEAN) / OBS_STATE_STD

    return joint_pos


def postprocess_lerobot_to_isaac(joint_pos: np.ndarray) -> np.ndarray:
    """
    Convert joint positions from LeRobot to Isaac Lab format.

    SmolVLA postprocessor already denormalizes, so we receive LeRobot degrees.
    We only need:
    1. Map from LeRobot range to Isaac range
    2. Convert from degrees to radians
    """
    # Step 1: Map from LeRobot range to Isaac range
    # for i in range(6):
    #     isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
    #     lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
    #     isaac_range = isaaclab_max - isaaclab_min
    #     lerobot_range = lerobot_max - lerobot_min
    #     joint_pos[i] = (joint_pos[i] - lerobot_min) / lerobot_range * isaac_range + isaaclab_min

    # Step 2: Convert from degrees to radians
    joint_pos = joint_pos / 180 * np.pi

    return joint_pos


def prepare_observation(obs_dict: dict, device: torch.device) -> dict:
    """
    Prepare observation from Isaac Lab format to LeRobot format.

    IMPORTANT: Immediately copies data to CPU/numpy to avoid GPU memory accumulation.

    Args:
        obs_dict: Observation dictionary from Isaac Lab environment
        device: Device to move tensors to

    Returns:
        Dictionary with LeRobot-formatted observations
    """
    # Extract observations from policy group
    policy_obs = obs_dict["policy"]

    # Get joint positions (shape: [num_envs, 6]) - copy to CPU immediately
    joint_pos = policy_obs["joint_pos"].cpu().numpy()[0].copy()

    # Convert to LeRobot format
    joint_pos_lerobot = preprocess_isaac_to_lerobot(joint_pos)

    # Get camera images and copy to CPU/numpy immediately to free GPU memory
    # Isaac Lab returns uint8 0-255, LeRobot uses float32 0-1
    front_img = policy_obs["front"].cpu().numpy()[0].copy().astype(np.float32) / 255.0
    third_person_img = policy_obs["front_cam_cfg"].cpu().numpy()[0].copy().astype(np.float32) / 255.0
    gripper_img = policy_obs["gripper_cam_cfg"].cpu().numpy()[0].copy().astype(np.float32) / 255.0

    # Prepare LeRobot observation batch (fresh tensors, no reference to obs_dict)
    observation = {
        "observation.state": torch.from_numpy(joint_pos_lerobot).float().unsqueeze(0).to(device),
        "observation.images.front": torch.from_numpy(front_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.third_person": torch.from_numpy(third_person_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.gripper": torch.from_numpy(gripper_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "task": "Pick up the cube and lift it",  # Task description for SmolVLA
    }

    return observation


def run_inference():
    """Main inference loop."""

    # Load policy
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading SmolVLA policy from: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # Clear any leftover GPU memory before starting
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()

    print(f"Policy loaded successfully on {device}")
    print(f"Policy type: {type(policy).__name__}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    # Load dataset metadata for stats
    dataset_meta = LeRobotDatasetMetadata(repo_id=dataset_id)

    # Create preprocessor/postprocessor
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats=dataset_meta.stats
    )

    print(f"Preprocessor and postprocessor created")

    # Create Isaac Lab environment
    print("\nInitializing Isaac Lab environment...")
    env_cfg = LiftCubeEnvCfg()
    env_cfg.scene.num_envs = 1  # Single environment for testing

    # Configure actions for joint position control (same as teleoperation with SO-101 leader)
    env_cfg.use_teleop_device("so101leader")

    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"Environment created: {env.unwrapped.__class__.__name__}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.keys())}")

    # Statistics
    successes = []
    episode_lengths = []

    # Video recording setup
    video_writer = None
    if args_cli.video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (640, 480)  # Front camera size
        video_writer = cv2.VideoWriter(args_cli.video_path, fourcc, fps, frame_size)
        print(f"Recording video to: {args_cli.video_path}")

    # Run episodes
    for episode_idx in range(args_cli.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx + 1}/{args_cli.num_episodes}")
        print(f"{'='*60}")

        # Reset environment
        obs_dict, _ = env.reset()

        episode_reward = 0
        step_count = 0
        done = False

        while not done and step_count < args_cli.max_steps:
            # Record frame to video
            if video_writer is not None:
                front_frame = obs_dict["policy"]["front"].cpu().numpy()[0]
                # Convert RGB to BGR for OpenCV
                front_frame_bgr = cv2.cvtColor(front_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(front_frame_bgr)

            # Wrap entire inference in no_grad to prevent memory accumulation
            with torch.no_grad():
                # Prepare observation for policy
                observation = prepare_observation(obs_dict, device)

                # Apply preprocessor (normalizes images, etc.)
                observation_processed = preprocessor(observation)

                # Delete original observation to free memory
                del observation

                # Get action from policy
                action_lerobot = policy.select_action(observation_processed)

                # CRITICAL: Immediately clear GPU cache after vision encoder
                # gc.collect()
                # torch.cuda.empty_cache()

                # Delete processed observation
                del observation_processed

                # Postprocessor denormalizes actions
                action_lerobot = postprocessor(action_lerobot)

                # Convert action from LeRobot to Isaac Lab format
                action_np = action_lerobot.cpu().numpy()[0]  # [6]

                # Delete action tensor
                del action_lerobot

                # Clear again after all tensors deleted
                gc.collect()
                torch.cuda.empty_cache()

            # Convert to Isaac format (outside no_grad)
            action_rad = postprocess_lerobot_to_isaac(action_np.copy())

            # Debug output (copy data before deleting obs_dict)
            if step_count < 10 or step_count % 50 == 0:
                current_joint_pos = obs_dict["policy"]["joint_pos"].cpu().numpy()[0].copy()
                print(f"\n  Step {step_count}:")
                print(f"    Current joints (rad): {current_joint_pos}")
                print(f"    Predicted action (deg): {action_np}")
                print(f"    Action (rad): {action_rad}")

            # Delete old obs_dict BEFORE env.step to free GPU memory
            del obs_dict

            # Execute action
            action_tensor = torch.from_numpy(action_rad).float().unsqueeze(0).to(env.device)

            # Clean up numpy arrays immediately
            del action_rad, action_np

            obs_dict, reward, terminated, truncated, info = env.step(action_tensor)

            # Clean up action tensor
            del action_tensor

            done = terminated[0] or truncated[0]
            episode_reward += reward[0].item()
            step_count += 1

            # # Clear CUDA cache and collect garbage every 5 steps to prevent OOM
            # if step_count % 5 == 0:
            #     gc.collect()
            #     torch.cuda.empty_cache()

            # if step_count % 50 == 0:
            #     mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            #     print(f"  Step {step_count}: reward={episode_reward:.3f}, GPU mem={mem_allocated:.2f}GB")

        # Episode finished
        success = terminated[0].item()  # Assuming termination means success
        successes.append(success)
        episode_lengths.append(step_count)

        print(f"\nEpisode completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Success: {success}")

        # Clear CUDA cache and garbage after episode
        gc.collect()
        torch.cuda.empty_cache()

    # Close video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {args_cli.video_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {args_cli.num_episodes}")
    print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Min/Max episode length: {min(episode_lengths)}/{max(episode_lengths)} steps")

    # Close environment
    env.close()

    return successes, episode_lengths


if __name__ == "__main__":
    try:
        successes, lengths = run_inference()
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


