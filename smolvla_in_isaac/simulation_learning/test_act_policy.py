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

# LeRobot imports
from lerobot.policies.act.modeling_act import ACTPolicy

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test ACT policy in Isaac Lab")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="outputs/train/act_so101_test/checkpoints/last/pretrained_model",
    help="Path to the trained policy checkpoint",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of test episodes")
parser.add_argument("--max_steps", type=int, default=5000, help="Maximum steps per episode")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--livestream", type=int, default=0, help="Livestream mode (0=off 2=WebRTC)")

args_cli = parser.parse_args()

# -----------------------------
# Setup streaming and cameras
if args_cli.livestream == 2:
    args_cli.headless = True
    args_cli.enable_cameras = True

    sys.argv.append("--/app/livestream/publicEndpointAddress=100.115.105.111")
    sys.argv.append("--/app/livestream/port=49100")

    print(f"[DEBUG] Livestream enabled (WebRTC mode)")
    print(f"[DEBUG] Stream address: 100.115.105.111:49100")
    print(f"[DEBUG] enable_cameras = {args_cli.enable_cameras}")
    print(f"[DEBUG] headless = {args_cli.headless}")

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports (must be after AppLauncher)
from isaaclab.envs import ManagerBasedRLEnv

# Import environment config
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg


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
    """Convert joint positions from Isaac Lab to LeRobot format."""
    joint_pos = joint_pos / np.pi * 180  # rad to deg
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min
        joint_pos[i] = (joint_pos[i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min
    return joint_pos


def postprocess_lerobot_to_isaac(joint_pos: np.ndarray) -> np.ndarray:
    """Convert joint positions from LeRobot to Isaac Lab format."""

    joint_pos = joint_pos / 180 * np.pi  # deg to rad

    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min
        joint_pos[i] = (joint_pos[i] - lerobot_min) / lerobot_range * isaac_range + isaaclab_min
    return joint_pos


def prepare_observation(obs_dict: dict, device: torch.device) -> dict:
    """
    Prepare observation from Isaac Lab format to LeRobot format.

    Args:
        obs_dict: Observation dictionary from Isaac Lab environment
        device: Device to move tensors to

    Returns:
        Dictionary with LeRobot-formatted observations
    """
    # Extract observations from policy group
    policy_obs = obs_dict["policy"]

    # Get joint positions (shape: [num_envs, 6])
    joint_pos = policy_obs["joint_pos"].cpu().numpy()[0]  # Take first environment

    # Convert to LeRobot format
    joint_pos_lerobot = preprocess_isaac_to_lerobot(joint_pos)

    # Get camera images (shape: [num_envs, height, width, channels])
    # Isaac Lab returns uint8 0-255, but ACT was trained on float32 0-1
    front_img = policy_obs["front"].cpu().numpy()[0].astype(np.float32) / 255.0
    third_person_img = policy_obs["front_cam_cfg"].cpu().numpy()[0].astype(np.float32) / 255.0
    gripper_img = policy_obs["gripper_cam_cfg"].cpu().numpy()[0].astype(np.float32) / 255.0

    # Prepare LeRobot observation batch
    observation = {
        "observation.state": torch.from_numpy(joint_pos_lerobot).float().unsqueeze(0).to(device),
        "observation.images.front": torch.from_numpy(front_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.third_person": torch.from_numpy(third_person_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.gripper": torch.from_numpy(gripper_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
    }

    return observation


def run_inference():
    """Main inference loop."""

    # Load policy
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading policy from: {checkpoint_path}")
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    print(f"Policy loaded successfully on {device}")
    print(f"Policy type: {type(policy).__name__}")

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
            # Prepare observation for policy
            observation = prepare_observation(obs_dict, device)

            # print(observation.keys())

            # front = observation["observation.images.front"]

            # third = observation["observation.images.third_person"]

            # gripper = observation["observation.images.gripper"]

            # Get action from policy
            with torch.no_grad():
                action_lerobot = policy.select_action(observation)

            # Convert action from LeRobot to Isaac Lab format
            action_np = action_lerobot.cpu().numpy()[0]  # [6]

            action_rad = postprocess_lerobot_to_isaac(action_np.copy())

            # Debug output
            if step_count == 0 or step_count % 50 == 0:
                current_joint_pos = obs_dict["policy"]["joint_pos"].cpu().numpy()[0]
                print(f"\n  Step {step_count}:")
                print(f"    Current joints (rad): {current_joint_pos}")
                # print(f"    Action from policy: {action_np}")
                # print(f"    Action min/max from policy: {action_np.min():.3f} / {action_np.max():.3f}")


                print(f"Action in rad: {action_rad}")
                print(f"    Action min/max in rad: {action_rad.min():.3f} / {action_rad.max():.3f}")

            # Try using action directly without conversion first
            action_tensor = torch.from_numpy(action_rad).float().unsqueeze(0).to(env.device)
            obs_dict, reward, terminated, truncated, info = env.step(action_tensor)

            done = terminated[0] or truncated[0]
            episode_reward += reward[0].item()
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Step {step_count}: reward={episode_reward:.3f}")

        # Episode finished
        success = terminated[0].item()  # Assuming termination means success
        successes.append(success)
        episode_lengths.append(step_count)

        print(f"\nEpisode completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Success: {success}")

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


