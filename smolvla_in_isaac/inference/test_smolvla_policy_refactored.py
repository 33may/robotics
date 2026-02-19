"""
Test trained SmolVLA policy in Isaac Lab simulation.

REFACTORED VERSION - Uses shared utilities from smolvla_in_isaac.common

This script loads a trained SmolVLA policy and evaluates it in Isaac Lab's
lift_cube environment with video recording and success metrics.
"""
import os
import warnings
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import cv2
import gc

# CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

leisaac_assets = Path("/home/may33/projects/ml_portfolio/robotics/leisaac/assets")
if leisaac_assets.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(leisaac_assets)


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import shared utilities
from common import (
    setup_isaac_environment,
    create_app_launcher,
    create_lift_cube_environment,
    load_normalization_stats,
    preprocess_isaac_to_lerobot,
    postprocess_lerobot_to_isaac,
    DEFAULT_DATASET_REPO_ID,
    LEISAAC_ASSETS_ROOT,
)

# Set leisaac assets root
if LEISAAC_ASSETS_ROOT and LEISAAC_ASSETS_ROOT.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(LEISAAC_ASSETS_ROOT)

# Setup Isaac environment (suppress warnings)
setup_isaac_environment()
warnings.filterwarnings("ignore")

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test SmolVLA policy in Isaac Lab")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/finetune/smolvla_pick_place/checkpoints/best/pretrained_model",
        help="Path to the trained policy checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset ID for loading normalization stats",
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of test episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--livestream", type=int, default=2, help="Livestream mode (0=off 2=WebRTC)")
    parser.add_argument("--video_path", type=str, default="smolvla_test.mp4", help="Path to save video")
    parser.add_argument("--task", type=str, default="Pick up the cube and lift it", help="Task description")
    return parser.parse_args()


def prepare_observation_smolvla(obs_dict: dict, device: torch.device, task_description: str) -> dict:
    """
    Prepare observation from Isaac Lab format to SmolVLA format.

    SmolVLA requires task descriptions and doesn't use MEAN_STD normalization
    (preprocessor handles this).

    Args:
        obs_dict: Observation dictionary from Isaac Lab environment
        device: Device to move tensors to
        task_description: Natural language task description

    Returns:
        Dictionary with SmolVLA-formatted observations
    """
    # Extract observations from policy group
    policy_obs = obs_dict["policy"]

    # Get joint positions (shape: [num_envs, 6]) - copy to CPU immediately
    joint_pos = policy_obs["joint_pos"].cpu().numpy()[0].copy()

    # Convert to LeRobot format (degrees, no MEAN_STD normalization yet)
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
        "task": task_description,  # SmolVLA requires task description
    }

    return observation


def run_inference(args, env, policy, preprocessor, postprocessor, device):
    """
    Run policy inference in Isaac Lab environment.

    Args:
        args: Command line arguments
        env: Isaac Lab environment
        policy: Trained SmolVLA policy
        preprocessor: SmolVLA preprocessor
        postprocessor: SmolVLA postprocessor
        device: Torch device

    Returns:
        Tuple of (successes, episode_lengths)
    """
    # Statistics
    successes = []
    episode_lengths = []

    # Video recording setup
    video_writer = None
    if args.video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (640, 480)  # Front camera size
        video_writer = cv2.VideoWriter(args.video_path, fourcc, fps, frame_size)
        print(f"Recording video to: {args.video_path}")

    # Run episodes
    for episode_idx in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        # Reset environment
        obs_dict, _ = env.reset()

        episode_reward = 0
        step_count = 0
        done = False

        while not done and step_count < args.max_steps:
            # Record frame to video
            if video_writer is not None:
                front_frame = obs_dict["policy"]["front"].cpu().numpy()[0]
                front_frame_bgr = cv2.cvtColor(front_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(front_frame_bgr)

            # Wrap entire inference in no_grad to prevent memory accumulation
            with torch.no_grad():
                # Prepare observation for policy
                observation = prepare_observation_smolvla(obs_dict, device, args.task)

                # Apply preprocessor (normalizes images, tokenizes task, etc.)
                observation_processed = preprocessor(observation)

                # Delete original observation to free memory
                del observation

                # Get action from policy
                action_lerobot = policy.select_action(observation_processed)

                # Delete processed observation
                del observation_processed

                # Postprocessor denormalizes actions
                action_lerobot = postprocessor(action_lerobot)

                # Convert action from LeRobot to Isaac Lab format
                action_np = action_lerobot.cpu().numpy()[0]  # [6]

                # Delete action tensor
                del action_lerobot

                # Clear CUDA cache
                gc.collect()
                torch.cuda.empty_cache()

            # Convert to Isaac format (outside no_grad)
            action_rad = postprocess_lerobot_to_isaac(action_np.copy())

            # Debug output
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

        # Episode finished
        success = terminated[0].item()
        successes.append(success)
        episode_lengths.append(step_count)

        print(f"\nEpisode completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Success: {success}")

        # Clear CUDA cache after episode
        gc.collect()
        torch.cuda.empty_cache()

    # Close video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {args.video_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Min/Max episode length: {min(episode_lengths)}/{max(episode_lengths)} steps")

    return successes, episode_lengths


def main():
    """Main entry point."""
    args = parse_args()

    # Force headless if not specified
    if not args.headless:
        args.headless = True

    # Setup streaming
    if args.livestream == 2:
        args.headless = True
        sys.argv.append("--/app/livestream/publicEndpointAddress=100.115.105.111")
        sys.argv.append("--/app/livestream/port=49100")
        print(f"[DEBUG] Livestream enabled (WebRTC mode)")
        print(f"[DEBUG] Stream address: 100.115.105.111:49100")

    # Create AppLauncher and start Isaac Sim
    args.enable_cameras = True
    app_launcher, simulation_app = create_app_launcher(
        headless=args.headless,
        livestream=args.livestream,
        enable_cameras=args.enable_cameras,
    )

    try:
        # Load policy
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading SmolVLA policy from: {checkpoint_path}")
        policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        policy.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)

        print(f"Policy loaded successfully on {device}")
        print(f"Policy type: {type(policy).__name__}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        # Load dataset metadata for stats
        dataset_meta = LeRobotDatasetMetadata(repo_id=args.dataset)

        # Create preprocessor/postprocessor
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            policy.config,
            dataset_stats=dataset_meta.stats
        )
        print("Preprocessor and postprocessor created")

        # Create Isaac Lab environment
        print("\nInitializing Isaac Lab environment...")
        env = create_lift_cube_environment(
            num_envs=1,
            teleop_device="so101leader",  # Joint position control
        )

        print(f"Environment created: {env.unwrapped.__class__.__name__}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space keys: {list(env.observation_space.keys())}")

        # Run inference
        successes, lengths = run_inference(
            args, env, policy, preprocessor, postprocessor, device
        )

        # Close environment
        env.close()

    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
