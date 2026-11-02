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
import threading
from dataclasses import dataclass

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

GRIPPER_IDX = 5

# Detection camera mapping — inference camera names → detection expectations
DETECTION_CAMERAS = ["left", "right", "top", "gripper"]

# State-augmentation column order — must match the dataset's observation.state layout.
# v014 dataset: 6 joints + [left,right,top,gripper] × [duck,cup] × [cx,cy] = 22 dims.
DETECTION_AUG_CAM_ORDER = ["left", "right", "top", "gripper"]
DETECTION_AUG_OBJ_ORDER = ["duck", "cup"]

REAL_LIMITS_DEG = [
    (-114.5, 125.5),   # shoulder_pan
    (-109.9, 101.0),   # shoulder_lift
    (-106.0,  89.5),   # elbow_flex
    (-103.7, 103.8),   # wrist_flex
    (-171.3, 165.2),   # wrist_roll
    (  -2.3, 110.4),   # gripper
]


# ── Camera setup (shared module) ─────────────────────────────────────────────

from vbti.logic.cameras.cameras import (
    CAMERA_PRESETS,
    init_cameras as _init_cameras,
    capture_frames as _capture_frames,
    get_latest_depth as _get_latest_depth,
    stop_cameras as _stop_cameras,
    build_grid_frame as _build_grid_frame,
    show_camera_grid as _show_camera_grid_raw,
)
from vbti.logic.depth.realtime_prepare import depth_uint16_to_turbo_rgb as _depth_to_turbo
from vbti.logic.inference.async_chunk_runner import AsyncChunkRunner

DEFAULT_CAMERAS = CAMERA_PRESETS["realsense"]


@dataclass
class RuntimePromptState:
    """Mutable prompt source shared by keyboard, voice, and GUI."""

    prompt: str
    version: int = 0
    source: str = "initial"

    def __post_init__(self):
        self._lock = threading.Lock()

    def set(self, prompt: str, source: str) -> bool:
        prompt = prompt.strip()
        if not prompt:
            return False
        with self._lock:
            if prompt == self.prompt:
                return False
            self.prompt = prompt
            self.version += 1
            self.source = source
        return True

    def snapshot(self) -> tuple[str, int, str]:
        with self._lock:
            return self.prompt, self.version, self.source


@dataclass
class RuntimeModeState:
    """Thread-safe prompt update mode, controlled by CLI or GUI."""

    mode: str

    def __post_init__(self):
        self._lock = threading.Lock()

    def set(self, mode: str) -> bool:
        if mode not in {"smooth", "responsive"}:
            raise ValueError("mode must be 'smooth' or 'responsive'")
        with self._lock:
            if mode == self.mode:
                return False
            self.mode = mode
        return True

    def snapshot(self) -> str:
        with self._lock:
            return self.mode


RED_PROMPT = "Pick up the duck and place it in the red cup"
BLACK_PROMPT = "Pick up the duck and place it in the black cup"


def _show_camera_grid(frames, camera_names, step, action=None,
                      width=640, height=480, right_column=None,
                      hud_lines=None, gato=False):
    """Wrapper that passes JOINT_NAMES and uses 'Inference' window."""
    return _show_camera_grid_raw(
        frames, camera_names, step, action, width, height,
        joint_names=JOINT_NAMES, window_name="Inference",
        right_column=right_column, hud_lines=hud_lines, gato=gato,
    )


# ── Detection overlay ───────────────────────────────────────────────────────

def _init_detector(device: str = "cuda"):
    """Load student detector for live detection overlay."""
    from vbti.logic.detection.detect import StudentDetector
    detector = StudentDetector(device=device)
    vram_mb = torch.cuda.memory_allocated() / 1e6
    print(f"[detection] StudentDetector loaded — total VRAM: {vram_mb:.0f} MB")
    return detector


def _run_detection_overlay(detector, images: dict, camera_names: list) -> dict:
    """Run detection on each camera frame, draw boxes + centers, return results.

    Modifies images IN-PLACE (draws on BGR frames).
    Returns dict of {cam: {duck: {...}, cup: {...}}} detection results.
    """
    results = {}
    for cam_name in camera_names:
        if cam_name not in images or cam_name not in DETECTION_CAMERAS:
            continue

        frame = images[cam_name]  # RGB uint8 from _capture_frames
        det = detector.detect(frame, cam_name)
        results[cam_name] = det

        h, w = frame.shape[:2]

        # Draw duck (green) and cup (red)
        colors = {"duck": (0, 255, 0), "cup": (0, 0, 255)}
        for obj_name, color in colors.items():
            obj = det[obj_name]
            if not obj["found"]:
                continue
            # Draw bounding box
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Draw center crosshair
            cx, cy = obj["center"]
            cx, cy = int(cx), int(cy)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 15, 2)
            # Label with confidence
            label = f"{obj_name} {obj['confidence']:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return results


class DetectionStateHolder:
    """Per-trial hold-last-good for the 16-d state-augmentation vector.

    Mirrors the dataset's offline `apply_confidence_hold`: when an object is
    not found in the current frame, re-emit the last-known cx/cy instead of
    zeros, so live state aug matches the training distribution.

    Output order: [left, right, top, gripper] × [duck, cup] × [cx_norm, cy_norm].
    Before any successful detection for a (cam, obj), values are 0.0 — matching
    `np.nan_to_num` of pre-first-detection NaNs in the dataset.
    """

    def __init__(self):
        self._last: dict[tuple[str, str], tuple[float, float]] = {}

    def reset(self):
        """Clear held values — call between trials / episodes."""
        self._last.clear()

    def update_and_vector(self, det_results: dict | None) -> np.ndarray:
        out: list[float] = []
        results = det_results or {}
        for cam in DETECTION_AUG_CAM_ORDER:
            cam_det = results.get(cam) or {}
            for obj in DETECTION_AUG_OBJ_ORDER:
                d = cam_det.get(obj)
                if d and d.get("found"):
                    cx, cy = float(d["center_norm"][0]), float(d["center_norm"][1])
                    self._last[(cam, obj)] = (cx, cy)
                else:
                    cx, cy = self._last.get((cam, obj), (0.0, 0.0))
                out.append(cx)
                out.append(cy)
        return np.array(out, dtype=np.float32)


def _detection_state_vector(det_results: dict | None) -> np.ndarray:
    """Stateless 16-d aug vector — zeros on not-found.

    Kept for tests/diagnostics only. Live inference must use
    `DetectionStateHolder` to match the dataset's hold-last-good behavior.
    """
    out: list[float] = []
    results = det_results or {}
    for cam in DETECTION_AUG_CAM_ORDER:
        cam_det = results.get(cam) or {}
        for obj in DETECTION_AUG_OBJ_ORDER:
            d = cam_det.get(obj)
            if d and d.get("found"):
                cx, cy = d["center_norm"]
            else:
                cx, cy = 0.0, 0.0
            out.append(float(cx))
            out.append(float(cy))
    return np.array(out, dtype=np.float32)


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

def _init_robot(port: str, robot_id: str | None = None, max_relative_target: float = 10.0):
    """Connect to SO-101 follower arm."""
    from vbti.logic.servos.profiles import get_active_profile
    if robot_id is None:
        robot_id = get_active_profile()

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
                       preprocessor, device: torch.device,
                       camera_name_map: dict[str, str] | None = None,
                       state_aug: np.ndarray | None = None) -> dict:
    """Build policy input from robot state + camera frames.

    camera_name_map: optional {physical_name: training_name} remapping.
    Example: {"gripper": "wrist_cam", "top": "top_cam"}
    state_aug: optional 1-D array concatenated to state_deg before normalization
               (e.g. detection cx/cy for v014's 22-d state).
    """
    policy_input = {}

    full_state = (np.concatenate([state_deg, state_aug])
                  if state_aug is not None else state_deg)
    policy_input["observation.state"] = torch.tensor(
        full_state, dtype=torch.float32
    ).unsqueeze(0).to(device)

    for name in camera_names:
        obs_name = camera_name_map[name] if camera_name_map and name in camera_name_map else name
        if name in images:
            img = torch.from_numpy(images[name]).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            img = torch.zeros(1, 3, 480, 640, dtype=torch.float32, device=device)
        policy_input[f"observation.images.{obs_name}"] = img

    policy_input["task"] = task
    policy_input = preprocessor(policy_input)
    return policy_input


# ── Main inference loop ───────────────────────────────────────────────────────

def run(
    checkpoint: str = "",
    port: str = "/dev/ttyACM0",
    cameras: str = "realsense",
    camera_config: dict | None = None,
    camera_names: list | None = None,
    camera_name_map: dict | None = None,
    task: str = "pick up the object",
    robot_id: str | None = None,
    max_relative_target: float = 10.0,
    move_to_start: bool = True,
    action_horizon: int = 10,
    enable_rtc: bool = True,
    fps: int = 30,
    max_steps: int = 500,
    show_cameras: bool = True,
    record: str = "",
    device: str = "auto",
    print_actions_every: int = 0,
    delta_actions: bool = False,
    detection: bool = False,
    replay_pkl: str = "",
    depth: bool = False,
    live_prompt_toggle: bool = False,
    prompt_update_mode: str = "smooth",
    red_prompt: str = RED_PROMPT,
    black_prompt: str = BLACK_PROMPT,
    voice_prompt: bool = False,
    voice_backend: str = "auto",
    voice_model: str = "base.en",
    voice_sample_rate: int = 44100,
    voice_device: int | str | None = None,
    voice_command_mode: str = "red_black",
    prompt_gui: bool = False,
    gato: bool = False,
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
        action_horizon: RTC execution horizon — actions committed per chunk
                        before the async runner re-plans
        enable_rtc: Real-Time Chunking on/off. On = the new chunk's head is
                    inpainted to match the committed trajectory (no jerk).
        fps: control loop frequency
        max_steps: maximum total steps before stopping
        show_cameras: display live camera grid window
        record: path to save video (e.g. "inference_run.mp4"). Empty = no recording
        device: "auto", "cuda", "cpu"
        print_actions_every: print action values every N steps (0 = disabled)
        replay_pkl: path to smolvla_eval.py pickle — dict {"action_0": (1,6), ...}
                    of normalized select_action outputs. Converted to degrees via postprocessor.
                    Requires checkpoint for the postprocessor stats.
        live_prompt_toggle: enable keyboard prompt switching in the camera window.
        prompt_update_mode: "smooth" waits for normal chunk refresh; "responsive"
                    resets the async runner when the prompt changes.
        red_prompt: prompt used by the red target toggle state.
        black_prompt: prompt used by the black target toggle state.
        voice_prompt: enable voice prompt input. Press 'v' to start/stop recording.
        voice_backend: "auto", "faster-whisper", or "whisper".
        voice_model: speech-to-text model name/path.
        voice_sample_rate: microphone sample rate.
        voice_device: optional sounddevice input device index/name.
        voice_command_mode: "red_black" maps recognized red/black words to the
                    canonical prompts; "direct" uses recognized text as prompt.
        prompt_gui: open a polished prompt-control GUI with custom textbox,
                    red/black quick buttons, and live update-mode controls.
    """
    if prompt_update_mode not in {"smooth", "responsive"}:
        raise ValueError("prompt_update_mode must be 'smooth' or 'responsive'")
    if voice_command_mode not in {"red_black", "direct"}:
        raise ValueError("voice_command_mode must be 'red_black' or 'direct'")
    import pickle
    from collections import deque

    if camera_config is None:
        camera_config = CAMERA_PRESETS.get(cameras, CAMERA_PRESETS["realsense"])
    assert isinstance(camera_config, dict)  # narrowed for pyright
    cam_cfg: dict = {**camera_config}  # local copy so depth-mutation doesn't leak
    # When --depth=true, splice "depth": True into the gripper config and
    # extend camera_names with the virtual "gripper_depth" key so it flows
    # through _build_observation / _build_grid_frame like any other tile.
    if depth:
        if "gripper" not in cam_cfg:
            raise ValueError(
                f"--depth=true requires a 'gripper' camera in preset; got: {list(cam_cfg.keys())}"
            )
        cam_cfg["gripper"] = {**cam_cfg["gripper"], "depth": True}
        if cam_cfg["gripper"].get("type", "realsense") != "realsense":
            raise ValueError("--depth=true is only supported for RealSense gripper (D405).")
    camera_config = cam_cfg
    if camera_names is None:
        camera_names = list(camera_config.keys())
        if depth:
            camera_names.append("gripper_depth")

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
    policy = preprocessor = postprocessor = None
    runner: AsyncChunkRunner | None = None
    if not replay_pkl:
        policy, preprocessor, postprocessor = _load_policy(checkpoint, dev)
    elif checkpoint:
        # Replay mode: only need postprocessor to convert normalized pkl → degrees
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=Path(checkpoint).expanduser().resolve(),
            config_filename="policy_postprocessor.json",
        )

    # ── Load replay actions (pkl mode) ────────────────────────
    # pkl format: {"action_0": (1,6), "action_1": (1,6), ...}
    # values are normalized select_action outputs — run through postprocessor → degrees
    replay_queue: deque | None = None
    if replay_pkl:
        with open(Path(replay_pkl).expanduser().resolve(), "rb") as f:
            pkl_data = pickle.load(f)
        sorted_keys = sorted(pkl_data.keys(), key=lambda k: int(k.split("_")[1]))
        flat = []
        for k in sorted_keys:
            arr = torch.tensor(np.asarray(pkl_data[k]), dtype=torch.float32)  # (1, 6) normalized
            if postprocessor is not None:
                deg = postprocessor({"action": arr})["action"].numpy()  # (1, 6) degrees
            else:
                deg = arr.numpy()
            flat.append(deg.reshape(-1))  # (6,) degrees
        replay_queue = deque(flat)
        print(f"[replay] Loaded {len(replay_queue)} steps from {replay_pkl}")

    # ── Async chunk runner — closed-loop RTC inference engine ────
    # Mirrors eval_engine.py: a background worker re-plans the next action
    # chunk while the current one executes, and RTC inpaints the new chunk's
    # head to match the trajectory already committed. This replaces the old
    # synchronous select_action queue. Skipped in replay mode (replay streams
    # precomputed actions and never touches the policy).
    if not replay_pkl:
        assert policy is not None and postprocessor is not None
        runner = AsyncChunkRunner(
            policy, postprocessor,
            execution_horizon=action_horizon,
            enable_rtc=enable_rtc,
        )
        runner.start()
        runner.reset()
        print(f"Async chunk runner: execution_horizon={action_horizon}, "
              f"RTC={'on' if enable_rtc else 'off'}")

    # ── Load detector (optional) ─────────────────────────────
    detector = None
    state_holder = None
    if detection:
        detector = _init_detector(device=str(dev))
        state_holder = DetectionStateHolder()

    # ── Init hardware ─────────────────────────────────────────
    print(f"\nInitializing cameras...")
    cameras = _init_cameras(camera_config, fps=fps)

    print(f"Connecting to robot on {port}...")
    robot = _init_robot(port, robot_id, max_relative_target)
    state = _get_state(robot)
    print(f"Current position (deg): {np.round(state, 1)}")

    # ── Inference loop ────────────────────────────────────────
    prompt_state = RuntimePromptState(task)
    mode_state = RuntimeModeState(prompt_update_mode)
    prompt_target = "red" if task.strip() == red_prompt.strip() else "black" if task.strip() == black_prompt.strip() else "custom"
    last_seen_prompt_version = prompt_state.version
    voice_listener = None
    prompt_gui_controller = None
    voice_status = "off"
    last_voice_text = ""
    step = 0

    def _apply_voice_text(text: str):
        nonlocal prompt_target, voice_status, last_voice_text
        last_voice_text = text
        lower = text.lower()
        if voice_command_mode == "red_black":
            if "black" in lower:
                prompt_target = "black"
                prompt_state.set(black_prompt, source="voice")
                voice_status = "recognized black"
            elif "red" in lower:
                prompt_target = "red"
                prompt_state.set(red_prompt, source="voice")
                voice_status = "recognized red"
            else:
                voice_status = "recognized text without red/black"
                print(f"[voice] ignored for red_black mode: {text!r}")
                return
        else:
            prompt_target = "custom"
            prompt_state.set(text, source="voice")
            voice_status = "recognized direct prompt"
        print(f"[voice] {text!r}")

    if voice_prompt:
        from vbti.logic.inference.voice_prompt import VoicePromptListener
        voice_listener = VoicePromptListener(
            on_text=_apply_voice_text,
            model_name=voice_model,
            sample_rate=voice_sample_rate,
            device=voice_device,
            backend=voice_backend,
        )
        voice_status = "ready"

    if prompt_gui:
        from vbti.logic.inference.prompt_gui import FullInferenceGui, PromptGuiStatus

        def _gui_status():
            p, v, src = prompt_state.snapshot()
            return PromptGuiStatus(
                prompt=p,
                version=v,
                source=src,
                mode=mode_state.snapshot(),
                step=step,
                running=True,
            )

        def _gui_set_prompt(prompt: str, source: str):
            nonlocal prompt_target
            if prompt.strip() == red_prompt.strip():
                prompt_target = "red"
            elif prompt.strip() == black_prompt.strip():
                prompt_target = "black"
            else:
                prompt_target = "custom"
            prompt_state.set(prompt, source=source)

        def _gui_set_mode(mode: str):
            if mode_state.set(mode):
                print(f"[prompt] update mode changed to {mode}")

        prompt_gui_controller = FullInferenceGui(
            get_status=_gui_status,
            on_prompt=_gui_set_prompt,
            on_mode=_gui_set_mode,
            red_prompt=red_prompt,
            black_prompt=black_prompt,
        )
        prompt_gui_controller.start()

    initial_prompt, _, _ = prompt_state.snapshot()
    print(f"\nTask: '{initial_prompt}'")
    print(f"Prompt update mode: {mode_state.snapshot()}")
    if live_prompt_toggle:
        print("Live prompt toggle: ENABLED — press 't' in camera window to toggle red/black")
    if voice_prompt:
        print("Voice prompt: ENABLED — press 'v' in camera window to start/stop recording")
    if prompt_gui:
        print("Prompt GUI: ENABLED — custom text, red/black buttons, and mode controls")
    print(f"Action horizon: {action_horizon}, FPS: {fps}, Max steps: {max_steps}")
    print(f"Safety clamp: {max_relative_target} deg/step")
    if delta_actions:
        print(f"  Delta actions: ENABLED (step-wise delta, joints reconstructed from state + delta)")
    if detection:
        vram_total = torch.cuda.memory_allocated() / 1e6
        print(f"  Detection: ENABLED — total VRAM: {vram_total:.0f} MB")
    print("=" * 60)
    if prompt_gui:
        print("Use the inference GUI for camera view and prompt commands. Close GUI or Ctrl+C to stop\n")
    elif live_prompt_toggle or voice_prompt:
        controls = []
        if live_prompt_toggle:
            controls.append("'t' toggle red/black")
        if voice_prompt:
            controls.append("'v' voice start/stop")
        controls.append("'q' quit")
        print("Press " + ", ".join(controls) + " in camera window or Ctrl+C to stop\n")
    else:
        print("Press 'q' in camera window or Ctrl+C to stop\n")

    # ── Video recorder ────────────────────────────────────────
    recorded_frames = []
    if record:
        record_path = Path(record).with_suffix(".mp4")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording to: {record_path}")

    last_action = None
    frame_drops = 0

    # When --depth=true, every capture pulls the gripper's aligned uint16 depth
    # and runs the same turbo-RGB bake the dataset prep used (clip [0.05, 0.20]
    # m → COLORMAP_TURBO via colorize_fixed_clip), splicing it into the frames
    # dict under the virtual ``gripper_depth`` key so it flows through the
    # existing _build_observation / _build_grid_frame plumbing unchanged.
    def _capture(cams):
        f = _capture_frames(cams)
        if depth:
            d = _get_latest_depth(cams).get("gripper")
            if d is not None:
                f = dict(f)
                scale = cams["gripper"].get("depth_scale", None)
                if scale and scale > 0:
                    f["gripper_depth"] = _depth_to_turbo(d, depth_scale_m=float(scale))
                else:
                    f["gripper_depth"] = _depth_to_turbo(d)
        return f

    step_dt = 1.0 / fps

    try:
        while step < max_steps:
            t_step = time.perf_counter()

            # Read state
            state_deg = _get_state(robot)

            # Capture images (resilient — retries, uses last frame on timeout).
            # The same frames feed both the policy and the display/recording.
            images = _capture(cameras)
            if len(images) < len(camera_names):
                frame_drops += 1

            # Run detection overlay (draws on frames in-place, returns results)
            det_results = None
            if detector is not None:
                t_det = time.perf_counter()
                det_results = _run_detection_overlay(detector, images, camera_names)
                det_ms = (time.perf_counter() - t_det) * 1000
                if step % 30 == 0:
                    vram_mb = torch.cuda.memory_allocated() / 1e6
                    print(f"  [det] step {step}: {det_ms:.0f}ms, VRAM {vram_mb:.0f}MB")

            current_prompt, current_prompt_version, current_prompt_source = prompt_state.snapshot()
            current_update_mode = mode_state.snapshot()
            if current_prompt_version != last_seen_prompt_version:
                print(f"[prompt] v{current_prompt_version} from {current_prompt_source}: {current_prompt}")
                if current_update_mode == "responsive" and runner is not None:
                    runner.reset()
                    print("[prompt] async runner reset for responsive update")
                last_seen_prompt_version = current_prompt_version

            # Predict ONE action — replay queue, or the async RTC runner.
            # Closed-loop: a fresh observation is built and handed to the
            # runner every step, exactly like eval_engine.py.
            if replay_queue is not None:
                if not replay_queue:
                    print("[replay] Queue exhausted — stopping.")
                    break
                action = replay_queue.popleft()
            else:
                state_aug = (state_holder.update_and_vector(det_results)
                             if state_holder is not None else None)
                obs = _build_observation(state_deg, images, camera_names,
                                         current_prompt, preprocessor, dev, camera_name_map,
                                         state_aug=state_aug)
                assert runner is not None
                action = runner.step(obs)

            if delta_actions:
                # Reconstruct absolute target from delta prediction + current state
                target = state_deg + action
                target[GRIPPER_IDX] = action[GRIPPER_IDX]  # gripper is absolute
                # Safety clamp to joint limits
                for j in range(len(REAL_LIMITS_DEG)):
                    lo, hi = REAL_LIMITS_DEG[j]
                    target[j] = np.clip(target[j], lo, hi)
                action = target

            last_action = action

            action_dict = {f"{name}.pos": float(action[j])
                           for j, name in enumerate(JOINT_NAMES)}
            robot.send_action(action_dict)
            step += 1

            if voice_listener is not None:
                while True:
                    ev = voice_listener.poll_event()
                    if ev is None:
                        break
                    if ev.kind == "recording_started":
                        voice_status = "recording"
                    elif ev.kind == "recording_stopped":
                        voice_status = "transcribing"
                    elif ev.kind == "transcribing":
                        voice_status = "transcribing"
                    elif ev.kind == "text":
                        last_voice_text = ev.text
                    elif ev.kind in {"warn", "error"}:
                        voice_status = f"{ev.kind}: {ev.text}"
                        print(f"[voice] {voice_status}")

            hud_lines = None
            if live_prompt_toggle or voice_prompt or prompt_gui:
                hud_prompt, hud_version, hud_source = prompt_state.snapshot()
                short_prompt = hud_prompt
                if len(short_prompt) > 80:
                    short_prompt = short_prompt[:77] + "..."
                controls = []
                if live_prompt_toggle:
                    controls.append("t=toggle")
                if voice_prompt:
                    controls.append("v=voice")
                controls.append("q=quit")
                hud_lines = [
                    f"prompt[{hud_version}:{hud_source}]: {short_prompt}",
                    f"mode: {mode_state.snapshot()} | " + " | ".join(controls),
                ]
                if voice_prompt:
                    hud_lines.append(f"voice: {voice_status}" + (f" | {last_voice_text[:50]}" if last_voice_text else ""))

            # Display + recording — same frames the policy just consumed.
            # In demo GUI mode, Tkinter owns the camera view and prompt controls;
            # the OpenCV window is not shown.
            grid = None
            if prompt_gui:
                grid = _build_grid_frame(images, camera_names, step, last_action,
                                          right_column=(["gripper_depth"] if depth else None),
                                          hud_lines=hud_lines, gato=gato)
                if prompt_gui_controller is not None:
                    prompt_gui_controller.update_frame(grid)
                    if prompt_gui_controller.closed:
                        print("Quit via inference GUI.")
                        break
            elif show_cameras or record:
                key = _show_camera_grid(images, camera_names, step, last_action,
                                         right_column=(["gripper_depth"] if depth else None),
                                         hud_lines=hud_lines, gato=gato)
                if show_cameras and key == ord("q"):
                    print("Quit via camera window.")
                    break
                if show_cameras and live_prompt_toggle and key == ord("t"):
                    prompt_target = "black" if prompt_target == "red" else "red"
                    next_prompt = black_prompt if prompt_target == "black" else red_prompt
                    prompt_state.set(next_prompt, source="button")
                if show_cameras and voice_prompt and key == ord("v") and voice_listener is not None:
                    try:
                        voice_listener.toggle_recording()
                        voice_status = "recording" if voice_listener.recording else "transcribing"
                    except Exception as e:
                        voice_status = f"error: {e}"
                        print(f"[voice] {voice_status}")

            if record:
                if grid is None:
                    grid = _build_grid_frame(images, camera_names, step, last_action,
                                              right_column=(["gripper_depth"] if depth else None),
                                              hud_lines=hud_lines, gato=gato)
                recorded_frames.append(grid)

            # Print actions at configured interval
            if print_actions_every > 0 and step % print_actions_every == 0:
                action_str = "  ".join(f"{n[:8]}={action[j]:7.1f}" for j, n in enumerate(JOINT_NAMES))
                print(f"  step {step}/{max_steps}  {action_str}")

            # Rate limiting — hold the fps cadence
            elapsed = time.perf_counter() - t_step
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        print("Cleaning up...")
        if voice_listener is not None:
            voice_listener.close()
        if prompt_gui_controller is not None:
            prompt_gui_controller.close()
        # Stop the async worker thread before touching anything else.
        if runner is not None:
            runner.stop()
        # Return to rest before disconnecting to prevent arm from falling
        try:
            move_to_rest(robot, fps=fps)
        except Exception:
            pass
        # Save video before disconnect — disconnect can crash (servo overload etc.)
        if record and recorded_frames:
            _save_video_ffmpeg(recorded_frames, record_path, fps)

        _stop_cameras(cameras)
        if show_cameras and not prompt_gui:
            cv2.destroyAllWindows()
        robot.disconnect()
        print("Robot disconnected.")

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
    robot_id=None,
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

                quit_requested = False
                try:
                    while step < max_steps:
                        state_deg = _get_state(robot)
                        images = _capture_frames(cam_devices)
                        if len(images) < len(camera_names):
                            frame_drops += 1

                        with torch.inference_mode():
                            obs = _build_observation(state_deg, images, camera_names,
                                                     task, preprocessor, dev, camera_name_map)
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

                            # Per-step live capture for display + recording at fps cadence
                            live_images = _capture_frames(cam_devices)
                            if len(live_images) < len(camera_names):
                                frame_drops += 1

                            # NOTE: legacy eval() doesn't expose --depth; keep right_column=None.
                            key = _show_camera_grid(live_images, camera_names, step, last_action)
                            if key == ord("q"):
                                print("Quit via camera window.")
                                run_status = "stopped"
                                quit_requested = True
                                break

                            grid = _build_grid_frame(live_images, camera_names, step, last_action)
                            recorded_frames.append(grid)

                            if print_actions_every > 0 and step % print_actions_every == 0:
                                action_str = "  ".join(f"{n[:8]}={action[j]:7.1f}" for j, n in enumerate(JOINT_NAMES))
                                print(f"  step {step}/{max_steps}  {action_str}")

                            elapsed = time.perf_counter() - t_step
                            if elapsed < step_dt:
                                time.sleep(step_dt - elapsed)
                            if step >= max_steps:
                                break

                        if quit_requested:
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
