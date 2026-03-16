"""
Direct SmolVLA inference in Isaac Sim.

Usage:
    cd /home/may33/projects/ml_portfolio/robotics
    python -m vbti.utils.inference.run_smolvla_inference \
        --checkpoint=outputs/train/smolvla_lift_cube_3cams/best \
        --task=LeIsaac-SO101-LiftCube-v0
"""

import os
import sys
from pathlib import Path

# __file__ is vbti/utils/inference/run_smolvla_inference.py
# parents[3] is the robotics directory
_robotics_root = Path(__file__).parents[3]

# Add leisaac source to path for proper subpackage imports
_leisaac_src = _robotics_root / "leisaac" / "source" / "leisaac"
if _leisaac_src.exists() and str(_leisaac_src) not in sys.path:
    sys.path.insert(0, str(_leisaac_src))

# Set assets root for leisaac
_leisaac_assets = _robotics_root / "leisaac" / "assets"
if _leisaac_assets.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(_leisaac_assets)

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="SmolVLA inference in Isaac Sim")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-v0", help="Task environment")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--action_horizon", type=int, default=10, help="Actions to execute per inference")
parser.add_argument("--step_hz", type=int, default=30, help="Environment step rate")
parser.add_argument("--plot_actions", action="store_true", default=False, help="Plot predicted actions per joint")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of inference")
parser.add_argument("--video_path", type=str, default="/tmp/inference_video.mp4", help="Path to save video")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps in simulation")

# Isaac Sim launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# Now import everything else (after Isaac Sim is running)
import time
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors

import leisaac.tasks  # Register envs explicitly


def load_policy(checkpoint_path: str, device: torch.device):
    """Load trained SmolVLA policy, preprocessor, and postprocessor."""
    from lerobot.processor import PolicyProcessorPipeline

    # Resolve to absolute path - required for from_pretrained to recognize as local path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        # Resolve relative paths from robotics root, not CWD
        checkpoint_path = _robotics_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading policy from {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Load preprocessor (normalizes inputs)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_preprocessor.json",
    )

    # Load postprocessor (DENORMALIZES actions - critical!)
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_postprocessor.json",
    )

    return policy, preprocessor, postprocessor


# Global counter for debug image saving
_debug_img_counter = 0
_debug_img_dir = Path("/tmp/inference_debug_images")


def obs_to_policy_input(obs_dict: dict, preprocessor, device: torch.device) -> dict:
    """
    Convert Isaac Sim observations to policy input format.

    Maps:
        obs_dict["front"] -> observation.images.front
        obs_dict["front_cam_cfg"] -> observation.images.third_person
        obs_dict["gripper_cam_cfg"] -> observation.images.gripper
        obs_dict["joint_pos"] -> observation.state
    """
    global _debug_img_counter

    # Build policy input dict
    policy_input = {}

    # State - convert from radians (Isaac) to degrees (training data)
    if "joint_pos" in obs_dict:
        import math
        state_rad = obs_dict["joint_pos"]
        state_deg = state_rad * (180.0 / math.pi)  # radians → degrees
        policy_input["observation.state"] = state_deg

    # Images - map env keys to policy keys
    camera_mapping = {
        "front": "observation.images.front",
        "front_cam_cfg": "observation.images.third_person",
        "gripper_cam_cfg": "observation.images.gripper",
    }

    for env_key, policy_key in camera_mapping.items():
        if env_key in obs_dict:
            img = obs_dict[env_key]

            # Isaac returns (N, H, W, C), policy expects (N, C, H, W)
            if img.dim() == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            # Convert to float [0, 1] - policy expects float tensors
            if img.dtype == torch.uint8:
                img = img.float() / 255.0

            # Save debug images AFTER transform (first 10 frames)
            if _debug_img_counter < 10:
                _save_debug_image(img, env_key, _debug_img_counter)

            policy_input[policy_key] = img

    # Increment counter after processing all cameras for this frame
    if _debug_img_counter < 10:
        _debug_img_counter += 1
        if _debug_img_counter == 10:
            print(f"\nDebug images saved to: {_debug_img_dir}")

    # Add task description for SmolVLA
    policy_input["task"] = "Pick up the cube and lift it"

    # Preprocess (normalize)
    policy_input = preprocessor(policy_input)

    return policy_input


def _save_debug_image(img: torch.Tensor, camera_name: str, frame_idx: int):
    """Save debug image to disk for comparison with training data.

    Expected input: (N, C, H, W) float tensor in [0, 1] range after transform.
    """
    from PIL import Image

    _debug_img_dir.mkdir(parents=True, exist_ok=True)

    # Get first env image (remove batch dim): (N, C, H, W) -> (C, H, W)
    if img.dim() == 4:
        img = img[0]

    # Now img is (C, H, W), convert to (H, W, C) for PIL
    img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Convert float [0,1] to uint8 [0,255]
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype('uint8')

    pil_img = Image.fromarray(img_np)
    save_path = _debug_img_dir / f"{camera_name}_frame{frame_idx:03d}.png"
    pil_img.save(save_path)
    print(f"  Saved: {save_path.name} shape={img_np.shape} range=[{img.min():.3f}, {img.max():.3f}]")


def main():
    device = torch.device(args.device if args.device else "cuda")
    print(f"Device: {device}")

    # ============== LOAD ENVIRONMENT ==============
    print(f"\nLoading environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, device=str(device), num_envs=1)
    env_cfg.use_teleop_device("so101leader")  # Use same config as data collection
    env_cfg.seed = int(time.time())
    env_cfg.recorders = None

    # Disable timeout for manual testing
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env: ManagerBasedRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped

    # ============== LOAD POLICY ==============
    print(f"\nLoading policy from: {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, device)

    print(f"Policy loaded: {type(policy).__name__}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Postprocessor loaded: will denormalize actions")

    # ============== INFERENCE LOOP ==============
    print("\n" + "=" * 60)
    print("STARTING INFERENCE")
    print("Press R to reset episode")
    print("=" * 60)

    success_count = 0
    episode = 0
    video_frames = []  # Collect frames for video

    obs_dict, _ = env.reset()

    while simulation_app.is_running() and episode < args.num_episodes:
        episode += 1
        print(f"\n[Episode {episode}/{args.num_episodes}]")

        step = 0
        done = False

        actions = []

        while not done and simulation_app.is_running():
            with torch.inference_mode():
                # Convert observations to policy format
                policy_obs = obs_to_policy_input(obs_dict["policy"], preprocessor, device)

                # Get actions from policy (NORMALIZED)
                actions_normalized = policy.select_action(policy_obs)  # (chunk_size, action_dim)

                # DEBUG: Print observation state and raw model output
                print(f"\n=== Step {step} ===")
                if "joint_pos" in obs_dict["policy"]:
                    state_rad = obs_dict["policy"]["joint_pos"][0].cpu()
                    state_deg = state_rad * (180.0 / 3.14159)
                    print(f"Obs state (deg): {state_deg.tolist()}")
                print(f"Raw model output: {actions_normalized[0].cpu().tolist()}")

                # DENORMALIZE actions using postprocessor
                actions_dict = {"action": actions_normalized}
                actions_denorm = postprocessor(actions_dict)["action"]  # Back to raw degrees
                print(f"After postproc (deg): {actions_denorm[0].cpu().tolist()}")

                # Convert degrees to radians (training data was in degrees, env expects radians)
                import math
                actions_rad = actions_denorm * (math.pi / 180.0)
                print(f"Final radians: {actions_rad[0].cpu().tolist()}")

                # Debug: print current joint state for first few steps
                if step < 3 and "joint_pos" in obs_dict["policy"]:
                    joints_rad = obs_dict['policy']['joint_pos'][0].cpu().numpy()
                    joints_deg = joints_rad * (180.0 / 3.14159)
                    print(f"Current joints (deg): {joints_deg}")

                # Execute action_horizon steps (using RADIANS)
                for i in range(min(args.action_horizon, actions_rad.shape[0])):
                    action = actions_rad[i:i+1]  # Keep batch dim
                    actions.append(action)

                    obs_dict, reward, terminated, truncated, info = env.step(action)
                    step += 1

                    # Collect video frames from front camera
                    if args.save_video and "front" in obs_dict["policy"]:
                        frame = obs_dict["policy"]["front"][0].cpu()  # (H, W, C) or (C, H, W)
                        if frame.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                            frame = frame.permute(1, 2, 0)
                        frame = (frame.numpy() * 255).clip(0, 255).astype('uint8')
                        video_frames.append(frame)

                    # Check termination
                    if terminated.any():
                        if info.get("success", [False])[0]:
                            print(f"  SUCCESS at step {step}!")
                            success_count += 1
                        done = True
                        break

                    if truncated.any():
                        print(f"  Timeout at step {step}")
                        done = True
                        break

                    # Rate limiting
                    time.sleep(1.0 / args.step_hz)
                
            if step >= args.max_steps:
                break

        # Reset for next episode
        if episode < args.num_episodes:
            obs_dict, _ = env.reset()

    # ============== RESULTS ==============
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Success rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")

    # ============== SAVE VIDEO ==============
    if args.save_video and len(video_frames) > 0:
        save_video(video_frames, args.video_path, fps=args.step_hz)

    # ============== PLOT ACTIONS ==============
    if args.plot_actions and len(actions) > 0:
        plot_actions(actions)

    env.close()
    simulation_app.close()


def save_video(frames: list, output_path: str, fps: int = 30):
    """Save frames as MP4 video."""
    import numpy as np

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"\nVideo saved to: {output_path} ({len(frames)} frames)")
    except ImportError:
        # Fallback to OpenCV if imageio not available
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"\nVideo saved to: {output_path} ({len(frames)} frames)")


def plot_actions(actions: list):
    """Plot predicted actions for each joint over time."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Stack actions: list of (1, 6) tensors -> (N, 6) array
    actions_array = np.concatenate([a.cpu().numpy() for a in actions], axis=0)
    num_steps, num_joints = actions_array.shape

    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper"
    ]

    # Create subplot for each joint
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_joints):
        ax = axes[i]
        ax.plot(actions_array[:, i], label=f"Joint {i}", linewidth=1)
        ax.set_title(joint_names[i] if i < len(joint_names) else f"Joint {i}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action (radians)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.suptitle(f"Predicted Actions Over Time ({num_steps} steps)", fontsize=14)
    plt.tight_layout()

    # Save and show
    save_path = Path("/tmp/inference_actions_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nActions plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
