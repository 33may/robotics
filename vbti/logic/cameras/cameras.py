"""Shared camera config, init, capture, and display utilities.

Import from anywhere:
    from vbti.logic.cameras.cameras import (
        CAMERA_PRESETS, init_cameras, capture_frames, stop_cameras,
        build_grid_frame, show_camera_grid,
    )
"""

import numpy as np
import cv2


# ── Camera presets ──────────────────────────────────────────────────────────

CAMERA_PRESETS = {
    "realsense": {
        "top":     {"type": "realsense", "serial": "123622270073"},
        "left":    {"type": "realsense", "serial": "123622270367"},
        "right":   {"type": "realsense", "serial": "126122270644"},
        "gripper": {"type": "opencv",    "path": "/dev/cam_gripper"},
    },
    "opencv": {
        "top":     {"type": "opencv", "path": "/dev/cam_top_raw"},
        "left":    {"type": "opencv", "path": "/dev/cam_left_raw"},
        "right":   {"type": "opencv", "path": "/dev/cam_right_raw"},
        "gripper": {"type": "opencv", "path": "/dev/cam_gripper_raw"},
    },
    "opencv-3cam": {
        "left":    {"type": "opencv", "path": "/dev/cam_left"},
        "right":   {"type": "opencv", "path": "/dev/cam_right"},
        "gripper": {"type": "opencv", "path": "/dev/cam_gripper"},
    },
    "sim_3_cam": {
        "wrist_cam": {"type": "opencv", "path": "/dev/cam_gripper"},
        "left_cam":  {"type": "opencv", "path": "/dev/cam_left"},
        "right_cam": {"type": "opencv", "path": "/dev/cam_right"},
    },
    "sim_3_cam_opencv": {
        "wrist_cam": {"type": "opencv", "path": "/dev/cam_gripper_raw"},
        "left_cam":  {"type": "opencv", "path": "/dev/cam_left_raw"},
        "right_cam": {"type": "opencv", "path": "/dev/cam_right_raw"},
    }
}
DEFAULT_PRESET = "realsense"


# ── Init / capture / stop ───────────────────────────────────────────────────

def init_cameras(camera_config: dict = None, width: int = 640,
                 height: int = 480, fps: int = 30) -> dict:
    """Initialize cameras. Returns dict of {name: capture_object}.

    Each entry is {"type": "realsense"|"opencv", "pipe"|"cap": ..., "last_frame": None}.
    """
    import pyrealsense2 as rs

    if camera_config is None:
        camera_config = CAMERA_PRESETS[DEFAULT_PRESET]

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
                for _ in range(5):
                    pipe.wait_for_frames(timeout_ms=2000)
                cameras[name] = {"type": "realsense", "pipe": pipe, "last_frame": None}
                print(f"  {name}: RealSense serial={serial} OK")
            except Exception as e:
                print(f"  [ERR] {name}: RealSense {serial} — {e}")

        elif cam_type == "opencv":
            path = cfg.get("path", cfg.get("index", 0))
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                for _ in range(5):
                    cap.read()
                cameras[name] = {"type": "opencv", "cap": cap, "last_frame": None}
                print(f"  {name}: OpenCV path={path} OK")
            else:
                print(f"  [ERR] {name}: OpenCV {path} failed")

    if len(cameras) < len(camera_config):
        failed = [n for n in camera_config if n not in cameras]
        raise RuntimeError(f"Failed to init cameras: {failed}. Fix hardware before running.")

    print(f"Initialized {len(cameras)}/{len(camera_config)} cameras")
    return cameras


def capture_frames(cameras: dict) -> dict[str, np.ndarray]:
    """Capture one frame per camera. Retries with last good frame on failure.

    Returns {name: (H, W, 3) uint8 RGB}.
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


def stop_cameras(cameras: dict):
    """Release all camera resources."""
    for cam in cameras.values():
        if cam["type"] == "realsense":
            cam["pipe"].stop()
        elif cam["type"] == "opencv":
            cam["cap"].release()


def build_grid_frame(frames: dict[str, np.ndarray], camera_names: list[str],
                     step: int = 0, action: np.ndarray | None = None,
                     width: int = 640, height: int = 480,
                     joint_names: list[str] | None = None) -> np.ndarray:
    """Build a 2x2 camera grid as BGR numpy array for display."""
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

    if action is not None and joint_names is not None:
        y_start = height + 50
        for j, (jname, val) in enumerate(zip(joint_names, action)):
            text = f"{jname[:8]:>8}: {val:7.1f}"
            cv2.putText(grid, text, (width + 10, y_start + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return grid


def show_camera_grid(frames: dict[str, np.ndarray], camera_names: list[str],
                     step: int = 0, action: np.ndarray | None = None,
                     width: int = 640, height: int = 480,
                     joint_names: list[str] | None = None,
                     window_name: str = "Cameras") -> int:
    """Display camera grid and return key press."""
    grid = build_grid_frame(frames, camera_names, step, action, width, height, joint_names)
    cv2.imshow(window_name, grid)
    return cv2.waitKey(1) & 0xFF
