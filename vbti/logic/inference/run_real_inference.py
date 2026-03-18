"""
Run trained SmolVLA policy on real SO-101 robot with RealSense cameras.

Features:
    - Live camera feed display during inference
    - Resilient frame capture (retries on timeout, uses last good frame)
    - Safety clamp on joint deltas
    - Dry run mode (cameras + model, no robot)

Usage:
    # Dry run with camera display
    python vbti/logic/inference/run_real_inference.py run \
        --checkpoint=vbti/experiments/duck_cup_smolvla/v001/checkpoints/best \
        --dry_run

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


# ── Camera setup ──────────────────────────────────────────────────────────────

DEFAULT_CAMERAS = {
    "top":     {"type": "realsense", "serial": "123622270073"},
    "left":    {"type": "realsense", "serial": "123622270367"},
    "right":   {"type": "realsense", "serial": "126122270644"},
    "gripper": {"type": "opencv",    "path": "/dev/video11"},
}


def _init_cameras(camera_config: dict = None, width: int = 640, height: int = 480, fps: int = 30):
    """Initialize cameras. Returns dict of {name: capture_object}."""
    import pyrealsense2 as rs

    if camera_config is None:
        camera_config = DEFAULT_CAMERAS

    cameras = {}
    for name, cfg in camera_config.items():
        cam_type = cfg.get("type", "realsense")

        if cam_type == "realsense":
            serial = cfg["serial"]
            rs_cfg = rs.config()
            rs_cfg.enable_device(serial)
            rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            pipe = rs.pipeline()
            try:
                pipe.start(rs_cfg)
                # Warmup — discard first few frames
                for _ in range(5):
                    pipe.wait_for_frames(timeout_ms=2000)
                cameras[name] = {"type": "realsense", "pipe": pipe, "last_frame": None}
                print(f"  {name}: RealSense serial={serial} OK")
            except Exception as e:
                print(f"  [ERR] {name}: RealSense {serial} — {e}")

        elif cam_type == "opencv":
            path = cfg.get("path", cfg.get("index", 0))
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if cap.isOpened():
                cameras[name] = {"type": "opencv", "cap": cap, "last_frame": None}
                print(f"  {name}: OpenCV path={path} OK")
            else:
                print(f"  [ERR] {name}: OpenCV {path} failed")

    if len(cameras) < len(camera_config):
        failed = [n for n in camera_config if n not in cameras]
        raise RuntimeError(f"Failed to init cameras: {failed}. Fix hardware before running inference.")

    print(f"Initialized {len(cameras)}/{len(camera_config)} cameras")
    return cameras


def _capture_frames(cameras: dict) -> dict[str, np.ndarray]:
    """Capture frames with retry. On failure, reuse last good frame.

    Returns {name: (H, W, 3) uint8 RGB}
    """
    frames = {}
    for name, cam in cameras.items():
        try:
            if cam["type"] == "realsense":
                ret, fs = cam["pipe"].try_wait_for_frames(timeout_ms=100)
                if ret:
                    color = fs.get_color_frame()
                    if color:
                        frame = np.asanyarray(color.get_data())
                        cam["last_frame"] = frame
                        frames[name] = frame
                        continue
                # Timeout — use last good frame
                if cam["last_frame"] is not None:
                    frames[name] = cam["last_frame"]

            elif cam["type"] == "opencv":
                ret, frame = cam["cap"].read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam["last_frame"] = frame
                    frames[name] = frame
                    continue
                if cam["last_frame"] is not None:
                    frames[name] = cam["last_frame"]

        except Exception:
            if cam["last_frame"] is not None:
                frames[name] = cam["last_frame"]

    return frames


def _build_grid_frame(frames: dict[str, np.ndarray], camera_names: list[str],
                      step: int, action: np.ndarray | None = None,
                      width: int = 640, height: int = 480) -> np.ndarray:
    """Build a 2x2 camera grid as BGR numpy array."""
    grid_imgs = []
    for name in camera_names:
        if name in frames:
            img = cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR)
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(img, f"step: {step}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        grid_imgs.append(img)

    cols = 2
    rows = (len(grid_imgs) + cols - 1) // cols
    while len(grid_imgs) < rows * cols:
        grid_imgs.append(np.zeros((height, width, 3), dtype=np.uint8))
    grid_rows = [np.hstack(grid_imgs[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    if action is not None:
        y_start = height + 50
        for j, (jname, val) in enumerate(zip(JOINT_NAMES, action)):
            text = f"{jname[:8]:>8}: {val:7.1f}"
            cv2.putText(grid, text, (width + 10, y_start + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return grid


def _show_camera_grid(frames: dict[str, np.ndarray], camera_names: list[str],
                      step: int, action: np.ndarray | None = None,
                      width: int = 640, height: int = 480):
    """Display camera grid and return key press."""
    grid = _build_grid_frame(frames, camera_names, step, action, width, height)
    cv2.imshow("Inference", grid)
    return cv2.waitKey(1) & 0xFF


def _save_video_ffmpeg(frames: list[np.ndarray], output_path: Path, fps: int):
    """Save BGR frames to mp4 using ffmpeg pipe. Produces Obsidian-compatible mp4."""
    import subprocess
    h, w = frames[0].shape[:2]
    print(f"Encoding {len(frames)} frames to {output_path}...")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "fast", "-crf", "23", str(output_path)],
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


def _stop_cameras(cameras: dict):
    for cam in cameras.values():
        if cam["type"] == "realsense":
            cam["pipe"].stop()
        elif cam["type"] == "opencv":
            cam["cap"].release()


# ── Robot setup ───────────────────────────────────────────────────────────────

def _init_robot(port: str, robot_id: str = "frodeo-test", max_relative_target: float = 10.0):
    """Connect to SO-101 follower arm."""
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


# ── Robot utils ───────────────────────────────────────────────────────────────

# Resting pose in degrees — arm tucked back, gripper open
REST_POSITION = {
    "shoulder_pan":  0.0,
    "shoulder_lift": -95.0,
    "elbow_flex":    100.0,
    "wrist_flex":    45.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}

def move_to_rest(robot, speed_deg_per_step: float = 3.0, fps: int = 30):
    """Move robot smoothly to resting position before starting inference.

    Interpolates from current position to REST_POSITION over multiple steps
    so the arm doesn't snap. Blocks until complete.

    Args:
        robot: connected SO101Follower instance
        speed_deg_per_step: max degrees to move per joint per step
        fps: control rate during the movement
    """
    current = _get_state(robot)
    target = np.array([REST_POSITION[n] for n in JOINT_NAMES])
    step_dt = 1.0 / fps

    print(f"Moving to rest position...")
    while True:
        diff = target - current
        max_diff = np.abs(diff).max()
        if max_diff < 0.5:
            break

        step = np.clip(diff, -speed_deg_per_step, speed_deg_per_step)
        current = current + step

        action_dict = {f"{name}.pos": float(current[j]) for j, name in enumerate(JOINT_NAMES)}
        robot.send_action(action_dict)
        time.sleep(step_dt)

    print(f"  At rest: {np.round(target, 1)}")


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
    camera_config: dict = None,
    camera_names: list = None,
    task: str = "pick up the object",
    robot_id: str = "frodeo-test",
    max_relative_target: float = 10.0,
    action_horizon: int = 10,
    fps: int = 30,
    max_steps: int = 500,
    dry_run: bool = False,
    show_cameras: bool = True,
    record: str = "",
    device: str = "auto",
):
    """Run SmolVLA inference on real robot with live camera display.

    Args:
        checkpoint: path to checkpoint directory
        port: serial port for SO-101 follower
        camera_config: camera config dict (see DEFAULT_CAMERAS)
        camera_names: ordered camera names matching training
        task: language task description for SmolVLA
        robot_id: robot identifier for calibration
        max_relative_target: safety clamp — max degrees per step
        action_horizon: how many actions to execute per inference call
        fps: control loop frequency
        max_steps: maximum total steps before stopping
        dry_run: cameras + model active, no robot connection
        show_cameras: display live camera grid window
        record: path to save video (e.g. "inference_run.mp4"). Empty = no recording
        device: "auto", "cuda", "cpu"
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERAS
    if camera_names is None:
        camera_names = list(camera_config.keys())

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # ── Load policy ───────────────────────────────────────────
    policy, preprocessor, postprocessor = _load_policy(checkpoint, dev)

    # ── Init hardware ─────────────────────────────────────────
    print(f"\nInitializing cameras...")
    cameras = _init_cameras(camera_config, fps=fps)

    robot = None
    if not dry_run:
        print(f"Connecting to robot on {port}...")
        robot = _init_robot(port, robot_id, max_relative_target)
        state = _get_state(robot)
        print(f"Current position (deg): {np.round(state, 1)}")
    else:
        print("[DRY RUN] — cameras + model, no robot")

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
            t_loop = time.perf_counter()

            # Read state
            state_deg = _get_state(robot) if robot else np.zeros(6)

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

                if robot:
                    action_dict = {f"{name}.pos": float(action[j])
                                   for j, name in enumerate(JOINT_NAMES)}
                    robot.send_action(action_dict)

                step += 1

                # Rate limiting
                elapsed = time.perf_counter() - t_step
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

                if step >= max_steps:
                    break

            # Progress
            if step % 30 == 0:
                inference_ms = (time.perf_counter() - t_loop) * 1000
                pos_str = f"pos: {np.round(state_deg, 1)}" if robot else ""
                print(f"  step {step}/{max_steps}  {inference_ms:.0f}ms  drops={frame_drops}  {pos_str}")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        print("Cleaning up...")
        _stop_cameras(cameras)
        if show_cameras:
            cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
            print("Robot disconnected.")

        # Save video via ffmpeg (produces proper mp4 Obsidian can play)
        if record and recorded_frames:
            _save_video_ffmpeg(recorded_frames, record_path, fps)

    print(f"Done — {step} steps, {frame_drops} frame drops.")


def eval(
    checkpoint: str,
    task: str = "pick up the duck and place it in the cup",
    port: str = "/dev/ttyACM0",
    experiment: str = None,
    version: str = None,
    max_steps: int = 500,
    fps: int = 30,
    dry_run: bool = False,
):
    """Run evaluation on one or all checkpoints of the active experiment version.

    Automatically resolves checkpoint paths and saves videos to eval/videos/.
    Between checkpoints, moves robot back to resting position.

    Args:
        checkpoint: checkpoint name ("step_002000", "best", "all")
        task: language instruction for the policy
        port: serial port for robot
        experiment: experiment name (uses active if not given)
        version: version id (uses active if not given)
        max_steps: steps per eval run
        fps: control rate
        dry_run: cameras + model only, no robot
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from vbti.logic.train.experiment_utils import resolve_checkpoint, _resolve_experiment, _resolve_version, _version_dir

    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    checkpoint_paths = resolve_checkpoint(checkpoint, experiment, version)

    eval_videos_dir = _version_dir(experiment, version) / "eval" / "videos"
    eval_videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEval: {experiment}/{version}")
    print(f"Checkpoints to run: {len(checkpoint_paths)}")
    for p in checkpoint_paths:
        print(f"  {p.name}")
    print()

    # Init hardware once, reuse across checkpoints
    cameras = _init_cameras(DEFAULT_CAMERAS, fps=fps)
    camera_names = list(DEFAULT_CAMERAS.keys())

    robot = None
    if not dry_run:
        print(f"Connecting to robot on {port}...")
        robot = _init_robot(port, robot_id="frodeo-test")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    try:
        for ckpt_path in checkpoint_paths:
            print(f"\n{'='*60}")
            print(f"Checkpoint: {ckpt_path.name}")
            print(f"{'='*60}")

            # Move to rest before each run
            if robot:
                move_to_rest(robot, fps=fps)
                print("Ready. Starting inference in 3s...")
                time.sleep(3)

            record_path = eval_videos_dir / f"eval_{version}_{ckpt_path.name}"

            # Load policy for this checkpoint
            policy, preprocessor, postprocessor = _load_policy(str(ckpt_path), dev)

            # Run inference loop inline (reusing open cameras + robot)
            recorded_frames = []
            step = 0
            last_action = None
            frame_drops = 0
            step_dt = 1.0 / fps

            try:
                while step < max_steps:
                    t_loop = time.perf_counter()

                    state_deg = _get_state(robot) if robot else np.zeros(6)
                    images = _capture_frames(cameras)
                    if len(images) < len(camera_names):
                        frame_drops += 1

                    key = _show_camera_grid(images, camera_names, step, last_action)
                    if key == ord("q"):
                        print("Quit via camera window.")
                        break

                    grid = _build_grid_frame(images, camera_names, step, last_action)
                    recorded_frames.append(grid)

                    with torch.inference_mode():
                        obs = _build_observation(state_deg, images, camera_names,
                                                 task, preprocessor, dev)
                        actions_normalized = policy.select_action(obs)
                        actions_deg = postprocessor({"action": actions_normalized})["action"]
                        actions_deg = actions_deg.cpu().numpy()

                    for i in range(min(10, len(actions_deg))):
                        t_step = time.perf_counter()
                        action = actions_deg[i]
                        last_action = action
                        if robot:
                            action_dict = {f"{name}.pos": float(action[j])
                                           for j, name in enumerate(JOINT_NAMES)}
                            robot.send_action(action_dict)
                        step += 1
                        elapsed = time.perf_counter() - t_step
                        if elapsed < step_dt:
                            time.sleep(step_dt - elapsed)
                        if step >= max_steps:
                            break

                    if step % 30 == 0:
                        print(f"  step {step}/{max_steps}  drops={frame_drops}")

            except KeyboardInterrupt:
                print(f"\nCheckpoint {ckpt_path.name} interrupted — saving video and continuing.")

            if recorded_frames:
                _save_video_ffmpeg(recorded_frames, record_path.with_suffix(".mp4"), fps)

            # Clean up policy GPU memory before next checkpoint
            del policy, preprocessor, postprocessor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        _stop_cameras(cameras)
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
            print("Robot disconnected.")

    print(f"\nEval complete. Videos saved to: {eval_videos_dir}")


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
