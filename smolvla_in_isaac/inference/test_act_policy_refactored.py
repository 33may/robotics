"""
Script to test trained ACT policy in Isaac Lab simulation.

REFACTORED VERSION - Uses common utilities from smolvla_in_isaac.common

This script:
1. Loads a trained ACT policy checkpoint
2. Initializes Isaac Lab lift_cube environment
3. Runs inference episodes with visualization
4. Evaluates success rate
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch



leisaac_assets = Path("/home/may33/projects/ml_portfolio/robotics/leisaac/assets")
if leisaac_assets.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(leisaac_assets)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import common utilities
from common import (
    # Setup
    setup_isaac_environment,
    create_app_launcher,
    create_lift_cube_environment,
    safe_close_environment,
    # Transformations
    load_normalization_stats,
    prepare_observation,
    postprocess_lerobot_to_isaac,
    # Constants
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_NUM_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_HEADLESS,
    DEFAULT_LIVESTREAM,
)

# Setup Isaac environment BEFORE importing Isaac Lab modules
setup_isaac_environment()

# LeRobot imports
from lerobot.policies.act.modeling_act import ACTPolicy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ACT policy in Isaac Lab")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/act_pick_place_eval/checkpoints/best/pretrained_model",
        help="Path to the trained policy checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo ID (for loading normalization stats)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=DEFAULT_HEADLESS,
        help="Run without GUI",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        default=DEFAULT_LIVESTREAM,
        help="Livestream mode (0=off 2=WebRTC)",
    )
    return parser.parse_args()


def run_inference(args):
    """Main inference loop."""

    # Load normalization statistics
    print(f"Loading normalization stats from dataset: {args.dataset}")
    stats = load_normalization_stats(args.dataset)

    # Load policy
    checkpoint_path = Path(args.checkpoint)
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
    env = create_lift_cube_environment(
        num_envs=1,
        teleop_device="so101leader",
    )

    # Statistics
    successes = []
    episode_lengths = []

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
            # Prepare observation for policy
            observation = prepare_observation(
                obs_dict,
                device,
                stats['obs_state_mean'],
                stats['obs_state_std']
            )

            # Get action from policy
            with torch.no_grad():
                action_lerobot = policy.select_action(observation)

            # Convert action from LeRobot to Isaac Lab format
            action_np = action_lerobot.cpu().numpy()[0]  # [6]
            action_rad = postprocess_lerobot_to_isaac(
                action_np.copy(),
                stats['action_mean'],
                stats['action_std']
            )

            # Debug output
            if step_count < 10 or step_count % 50 == 0:
                current_joint_pos = obs_dict["policy"]["joint_pos"].cpu().numpy()[0]
                print(f"\n  Step {step_count}:")
                print(f"    Current joints (rad): {current_joint_pos}")
                print(f"    Predicted action (normalized): {action_np}")
                print(f"    Action (rad): {action_rad}")

            # Execute action in environment
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
    print(f"Total episodes: {args.num_episodes}")
    print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Min/Max episode length: {min(episode_lengths)}/{max(episode_lengths)} steps")

    # Close environment
    safe_close_environment(env, None)

    return successes, episode_lengths


def main():
    """Main entry point."""
    args = parse_args()

    # Create Isaac Lab app launcher
    app_launcher, simulation_app = create_app_launcher(
        headless=args.headless,
        livestream=args.livestream,
        enable_cameras=True,
    )

    try:
        successes, lengths = run_inference(args)
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
