"""
Run trained SmolVLA policy on real SO-101 robot with RealSense cameras.

Features:
    - Live camera feed display during inference
    - Resilient frame capture (retries on timeout, uses last good frame)
    - Safety clamp on joint deltas
    - Periodic action printing for debugging

Usage:
    # Real robot
    python vbti/logic/inference/run_real_inference.py run \
        --checkpoint=vbti/experiments/duck_cup_smolvla/v001/checkpoints/best \
        --port=/dev/ttyACM0 \
        --task="pick up the duck and place it in the cup" \
        --max_relative_target=5.0

    # Camera preview only
    python vbti/logic/inference/run_real_inference.py preview
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import numpy as np
import torch
import cv2
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


# ── Camera setup (shared module) ─────────────────────────────────────────────

from vbti.logic.cameras.cameras import (
    CAMERA_PRESETS,
    init_cameras as _init_cameras,
    capture_frames as _capture_frames,
    stop_cameras as _stop_cameras,
    build_grid_frame as _build_grid_frame,
    show_camera_grid as _show_camera_grid_raw,
)

DEFAULT_CAMERAS = CAMERA_PRESETS["realsense"]


def _show_camera_grid(frames, camera_names, step, action=None,
                      width=640, height=480):
    """Wrapper that passes JOINT_NAMES and uses 'Inference' window."""
    return _show_camera_grid_raw(
        frames, camera_names, step, action, width, height,
        joint_names=JOINT_NAMES, window_name="Inference",
    )


def _save_video_ffmpeg(frames: list[np.ndarray], output_path: Path, fps: int):
    """Save BGR frames to mp4 using ffmpeg pipe. Produces Obsidian-compatible mp4."""
    import subprocess
    h, w = frames[0].shape[:2]
    print(f"Encoding {len(frames)} frames to {output_path}...")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "fast", "-crf", "23",
         "-movflags", "+faststart", str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        print(f"Video saved: {output_path}")
    else:
        print(f"[ERR] ffmpeg failed: {proc.stderr.read().decode()[-200:]}")


# ── Robot setup ───────────────────────────────────────────────────────────────

def _init_robot(port: str, robot_id: str = "frodeo-test", max_relative_target: float = 10.0):
    """Connect to SO-101 follower arm."""
    try:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower
    except ImportError:
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower

    config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        max_relative_target=max_relative_target,
    )
    robot = SO101Follower(config)
    robot.connect()
    return robot


def _get_state(robot) -> np.ndarray:
    """Read current joint positions as array [6] in degrees."""
    obs = robot.get_observation()
    return np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])


from vbti.logic.servos.rest import move_to_rest


# ── Policy loading ────────────────────────────────────────────────────────────

def _load_policy(checkpoint: str, device: torch.device):
    """Load SmolVLA + preprocessor + postprocessor from checkpoint."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading policy from {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_postprocessor.json",
    )

    total = sum(p.numel() for p in policy.parameters())
    print(f"Policy loaded: {total:,} params")
    return policy, preprocessor, postprocessor


# ── Observation building ──────────────────────────────────────────────────────

def _build_observation(state_deg: np.ndarray, images: dict[str, np.ndarray],
                       camera_names: list[str], task: str,
                       preprocessor, device: torch.device) -> dict:
    """Build policy input from robot state + camera frames."""
    policy_input = {}

    policy_input["observation.state"] = torch.tensor(
        state_deg, dtype=torch.float32
    ).unsqueeze(0).to(device)

    for name in camera_names:
        if name in images:
            img = torch.from_numpy(images[name]).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            img = torch.zeros(1, 3, 480, 640, dtype=torch.float32, device=device)
        policy_input[f"observation.images.{name}"] = img

    policy_input["task"] = task
    policy_input = preprocessor(policy_input)
    return policy_input


# ── Main inference loop ───────────────────────────────────────────────────────

def run(
    checkpoint: str,
    port: str = "/dev/ttyACM0",
    cameras: str = "realsense",
    camera_config: dict = None,
    camera_names: list = None,
    task: str = "pick up the object",
    robot_id: str = "frodeo-test",
    max_relative_target: float = 10.0,
    move_to_start: bool = True,
    action_horizon: int = 10,
    fps: int = 30,
    max_steps: int = 500,
    show_cameras: bool = True,
    record: str = "",
    device: str = "auto",
    print_actions_every: int = 0,
):
    """Run SmolVLA inference on real robot with live camera display.

    Args:
        checkpoint: path to checkpoint directory
        port: serial port for SO-101 follower
        cameras: camera preset ("realsense" or "opencv")
        camera_config: camera config dict (overrides preset)
        camera_names: ordered camera names matching training
        task: language task description for SmolVLA
        robot_id: robot identifier for calibration
        max_relative_target: safety clamp — max degrees per step
        action_horizon: how many actions to execute per inference call
        fps: control loop frequency
        max_steps: maximum total steps before stopping
        show_cameras: display live camera grid window
        record: path to save video (e.g. "inference_run.mp4"). Empty = no recording
        device: "auto", "cuda", "cpu"
        print_actions_every: print action values every N steps (0 = disabled)
    """
    if camera_config is None:
        camera_config = CAMERA_PRESETS.get(cameras, CAMERA_PRESETS["realsense"])
    if camera_names is None:
        camera_names = list(camera_config.keys())

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Connecting to robot on {port}...")
    robot = _init_robot(port, robot_id=robot_id)

    if robot and move_to_start:
        move_to_rest(robot, fps=fps)
    input("\nPress Enter to start inference...")

    # ── Load policy ───────────────────────────────────────────
    policy, preprocessor, postprocessor = _load_policy(checkpoint, dev)

    # ── Init hardware ─────────────────────────────────────────
    print(f"\nInitializing cameras...")
    cameras = _init_cameras(camera_config, fps=fps)

    print(f"Connecting to robot on {port}...")
    robot = _init_robot(port, robot_id, max_relative_target)
    state = _get_state(robot)
    print(f"Current position (deg): {np.round(state, 1)}")

    # ── Inference loop ────────────────────────────────────────
    print(f"\nTask: '{task}'")
    print(f"Action horizon: {action_horizon}, FPS: {fps}, Max steps: {max_steps}")
    print(f"Safety clamp: {max_relative_target} deg/step")
    print("=" * 60)
    print("Press 'q' in camera window or Ctrl+C to stop\n")

    # ── Video recorder ────────────────────────────────────────
    recorded_frames = []
    if record:
        record_path = Path(record).with_suffix(".mp4")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording to: {record_path}")

    step = 0
    last_action = None
    frame_drops = 0

    try:
        while step < max_steps:
            # Read state
            state_deg = _get_state(robot)

            # Capture images (resilient — retries, uses last frame on timeout)
            images = _capture_frames(cameras)
            if len(images) < len(camera_names):
                frame_drops += 1

            # Show camera grid + record
            if show_cameras or record:
                key = _show_camera_grid(images, camera_names, step, last_action)
                if show_cameras and key == ord("q"):
                    print("Quit via camera window.")
                    break

            # Collect frames for recording
            if record:
                grid = _build_grid_frame(images, camera_names, step, last_action)
                recorded_frames.append(grid)

            # Run policy
            with torch.inference_mode():
                obs = _build_observation(state_deg, images, camera_names,
                                         task, preprocessor, dev)
                actions_normalized = policy.select_action(obs)
                actions_deg = postprocessor({"action": actions_normalized})["action"]
                actions_deg = actions_deg.cpu().numpy()

            # Execute action horizon
            step_dt = 1.0 / fps
            for i in range(min(action_horizon, len(actions_deg))):
                t_step = time.perf_counter()
                action = actions_deg[i]
                last_action = action

                action_dict = {f"{name}.pos": float(action[j])
                               for j, name in enumerate(JOINT_NAMES)}
                robot.send_action(action_dict)

                step += 1

                # Print actions at configured interval
                if print_actions_every > 0 and step % print_actions_every == 0:
                    action_str = "  ".join(f"{n[:8]}={action[j]:7.1f}" for j, n in enumerate(JOINT_NAMES))
                    print(f"  step {step}/{max_steps}  {action_str}")

                # Rate limiting
                elapsed = time.perf_counter() - t_step
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

                if step >= max_steps:
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        print("Cleaning up...")
        # Return to rest before disconnecting to prevent arm from falling
        try:
            move_to_rest(robot, fps=fps)
        except Exception:
            pass
        _stop_cameras(cameras)
        if show_cameras:
            cv2.destroyAllWindows()
        robot.disconnect()
        print("Robot disconnected.")

        # Save video via ffmpeg (produces proper mp4 Obsidian can play)
        if record and recorded_frames:
            _save_video_ffmpeg(recorded_frames, record_path, fps)

    print(f"Done — {step} steps, {frame_drops} frame drops.")


def _ckpt_label(ckpt_path: Path) -> str:
    """Derive a readable checkpoint label from path.

    Handles both "step_006000" and "pretrained_model" (lerobot layout).
    """
    label = ckpt_path.name
    if label == "pretrained_model" and ckpt_path.parent.name.isdigit():
        label = f"step_{int(ckpt_path.parent.name):06d}"
    return label


def _resolve_checkpoint_list(checkpoint: str, experiment: str, version: str) -> list[Path]:
    """Resolve checkpoint specifier to paths. Supports comma-separated lists.

    Examples:
        "all"                          → all step checkpoints
        "best"                         → single named checkpoint
        "step_002000,step_004000,best" → specific list
        "2000,4000"                    → shorthand steps
    """
    from vbti.logic.train.experiment_utils import resolve_checkpoint

    # Comma-separated → resolve each independently
    if "," in checkpoint:
        paths = []
        for spec in checkpoint.split(","):
            spec = spec.strip()
            if spec:
                paths.extend(resolve_checkpoint(spec, experiment, version))
        return paths
    return resolve_checkpoint(checkpoint, experiment, version)


def eval(
    checkpoint: str,
    task: str = "pick up the duck and place it in the cup",
    port: str = "/dev/ttyACM1",
    robot_id="frodeo-test",
    cameras: str = "realsense",
    experiment: str = None,
    version: str = None,
    n_tries: int = 1,
    action_horizon: int = 10,
    max_steps: int = 500,
    fps: int = 30,
    move_to_start: bool = True,
    print_actions_every: int = 0,
):
    """Run evaluation on checkpoints with multiple tries each.

    Resolves checkpoint paths, runs each n_tries times, saves videos,
    and prints a summary table at the end for manual scoring.

    Args:
        checkpoint: checkpoint specifier — single name ("best", "step_002000"),
                    comma-separated list ("step_002000,step_004000,best"),
                    or "all" for every step checkpoint
        task: language instruction for the policy
        port: serial port for robot
        cameras: camera preset ("realsense" or "opencv")
        experiment: experiment name (uses active if not given)
        version: version id (uses active if not given)
        n_tries: number of evaluation runs per checkpoint
        action_horizon: how many actions to execute per inference call
        max_steps: steps per eval run
        fps: control rate
        move_to_start: move robot to rest position before each run
        print_actions_every: print action values every N steps (0 = disabled)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from vbti.logic.train.experiment_utils import _resolve_experiment, _resolve_version, _version_dir

    cam_config = CAMERA_PRESETS.get(cameras, CAMERA_PRESETS["realsense"])

    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    checkpoint_paths = _resolve_checkpoint_list(checkpoint, experiment, version)

    eval_videos_dir = _version_dir(experiment, version) / "eval" / "videos"
    eval_videos_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(checkpoint_paths) * n_tries
    print(f"\nEval: {experiment}/{version}")
    print(f"Cameras: {cameras}")
    print(f"Checkpoints: {len(checkpoint_paths)}, tries each: {n_tries}, total runs: {total_runs}")
    for p in checkpoint_paths:
        print(f"  {_ckpt_label(p)}")
    print()

    # Init hardware once, reuse across all runs
    cam_devices = _init_cameras(cam_config, fps=fps)
    camera_names = list(cam_config.keys())

    print(f"Connecting to robot on {port}...")
    robot = _init_robot(port, robot_id=robot_id)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Track results for summary table
    eval_results = []  # list of {"checkpoint", "try", "steps", "video", "status"}
    run_num = 0

    try:
        for ckpt_path in checkpoint_paths:
            label = _ckpt_label(ckpt_path)

            # Load policy once per checkpoint, reuse across tries
            print(f"\n{'='*60}")
            print(f"Loading checkpoint: {label}")
            print(f"{'='*60}")
            policy, preprocessor, postprocessor = _load_policy(str(ckpt_path), dev)

            for try_idx in range(1, n_tries + 1):
                run_num += 1
                try_label = f"try{try_idx}" if n_tries > 1 else ""

                print(f"\n{'─'*60}")
                print(f"[{run_num}/{total_runs}] {label}" + (f" — try {try_idx}/{n_tries}" if n_tries > 1 else ""))
                print(f"{'─'*60}")

                # Move to rest before each run
                if robot and move_to_start:
                    move_to_rest(robot, fps=fps)
                input("\nPress Enter to start inference...")

                # Video filename includes try number when n_tries > 1
                parts = [f"eval_{version}_{label}_ah{action_horizon}_{cameras}"]
                if try_label:
                    parts.append(try_label)
                record_path = eval_videos_dir / ("_".join(parts) + ".mp4")

                # Run inference loop inline (reusing open cameras + robot)
                recorded_frames = []
                step = 0
                last_action = None
                frame_drops = 0
                step_dt = 1.0 / fps
                run_status = "completed"

                try:
                    while step < max_steps:
                        state_deg = _get_state(robot)
                        images = _capture_frames(cam_devices)
                        if len(images) < len(camera_names):
                            frame_drops += 1

                        key = _show_camera_grid(images, camera_names, step, last_action)
                        if key == ord("q"):
                            print("Quit via camera window.")
                            run_status = "stopped"
                            break

                        grid = _build_grid_frame(images, camera_names, step, last_action)
                        recorded_frames.append(grid)

                        with torch.inference_mode():
                            obs = _build_observation(state_deg, images, camera_names,
                                                     task, preprocessor, dev)
                            actions_normalized = policy.select_action(obs)
                            actions_deg = postprocessor({"action": actions_normalized})["action"]
                            actions_deg = actions_deg.cpu().numpy()

                        for i in range(min(action_horizon, len(actions_deg))):
                            t_step = time.perf_counter()
                            action = actions_deg[i]
                            last_action = action
                            action_dict = {f"{name}.pos": float(action[j])
                                           for j, name in enumerate(JOINT_NAMES)}
                            robot.send_action(action_dict)
                            step += 1

                            if print_actions_every > 0 and step % print_actions_every == 0:
                                action_str = "  ".join(f"{n[:8]}={action[j]:7.1f}" for j, n in enumerate(JOINT_NAMES))
                                print(f"  step {step}/{max_steps}  {action_str}")

                            elapsed = time.perf_counter() - t_step
                            if elapsed < step_dt:
                                time.sleep(step_dt - elapsed)
                            if step >= max_steps:
                                break

                except KeyboardInterrupt:
                    print(f"\nRun interrupted — saving video and continuing.")
                    run_status = "interrupted"

                # Always return to rest after each run to prevent arm from falling
                print("Returning to rest position...")
                try:
                    move_to_rest(robot, fps=fps)
                except Exception as e:
                    print(f"[WARN] Failed to return to rest: {e}")

                if recorded_frames:
                    _save_video_ffmpeg(recorded_frames, record_path, fps)

                eval_results.append({
                    "checkpoint": label,
                    "try": try_idx,
                    "steps": step,
                    "status": run_status,
                    "video": record_path.name,
                })

            # Clean up policy GPU memory before next checkpoint
            del policy, preprocessor, postprocessor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # Return to rest before disconnecting to prevent arm from falling
        print("Returning to rest before shutdown...")
        try:
            move_to_rest(robot, fps=fps)
        except Exception:
            pass
        _stop_cameras(cam_devices)
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Robot disconnected.")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY — {experiment}/{version}")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Try':>4} {'Steps':>6} {'Status':<12} Video")
    print(f"{'─'*20} {'─'*4} {'─'*6} {'─'*12} {'─'*30}")
    for r in eval_results:
        print(f"{r['checkpoint']:<20} {r['try']:>4} {r['steps']:>6} {r['status']:<12} {r['video']}")
    print(f"\nVideos: {eval_videos_dir}")


def preview(camera_config: dict = None, width: int = 640, height: int = 480, fps: int = 30):
    """Live camera preview — no model, no robot. Press 'q' to quit."""
    if camera_config is None:
        camera_config = DEFAULT_CAMERAS

    print("Initializing cameras...")
    cameras = _init_cameras(camera_config, width=width, height=height, fps=fps)

    if not cameras:
        print("No cameras initialized.")
        return

    camera_names = list(camera_config.keys())
    print(f"\nShowing {len(cameras)} cameras. Press 'q' to quit.\n")

    try:
        while True:
            frames = _capture_frames(cameras)
            key = _show_camera_grid(frames, camera_names, 0)
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        _stop_cameras(cameras)
        cv2.destroyAllWindows()
        print("Cameras stopped.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "run":     run,
        "eval":    eval,
        "preview": preview,
    })
