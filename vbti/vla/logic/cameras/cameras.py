"""Shared camera config, init, capture, and display utilities.

Import from anywhere:
    from vbti.logic.cameras.cameras import (
        CAMERA_PRESETS, init_cameras, capture_frames, get_latest_depth,
        stop_cameras, build_grid_frame, show_camera_grid,
    )

Optional depth (RealSense only):
    Mark a camera entry with ``"depth": True`` in the preset (or pass an
    overridden ``camera_config`` dict). On ``init_cameras()``:
      - the depth stream is enabled alongside color
      - an ``rs.align(rs.stream.color)`` aligner is built per cam
      - the device-reported ``depth_scale`` (m / unit) is read once and stored
    On ``capture_frames()``: depth is pulled from the SAME ``fs`` blob as the
    color frame (so depth and color stay temporally + spatially registered)
    and tucked into the camera's ``last_depth`` slot. Use ``get_latest_depth``
    to fetch a ``{name: uint16 (H,W)}`` dict for inference-side processing
    (e.g. turbo-RGB bake matching the dataset prep pipeline).
"""

import numpy as np
import cv2
from pathlib import Path


GATO_IMAGE = Path.home() / "Pictures" / "gato.jpg"
_GATO_TILE: np.ndarray | None = None


# ── Camera presets ──────────────────────────────────────────────────────────

CAMERA_PRESETS = {
    # v016+ canonical preset. All four cameras on librealsense — gripper is a
    # RealSense D405 like the rest, accessed by serial. Depth enablement for
    # the gripper happens at record time via the `lerobot-record` CLI
    # (`use_depth: true` in the gripper's camera config). The preset itself
    # only carries serial routing because `init_cameras()` is an RGB-only
    # inference path; extending it to capture depth is a follow-up.
    "realsense": {
        "top":     {"type": "realsense", "serial": "123622270073"},
        "left":    {"type": "realsense", "serial": "128422270260"},
        "right":   {"type": "realsense", "serial": "126122270644"},
        "gripper": {"type": "realsense", "serial": "123622270367"},
    },
    # Same as ``realsense`` but the gripper D405 also streams aligned depth.
    # Use this preset (or pass ``--depth=true`` to inference / eval CLIs which
    # mutates the gripper entry on the fly) when the policy expects an
    # ``observation.images.gripper_depth`` input baked from live depth.
    "realsense_depth": {
        "top":     {"type": "realsense", "serial": "123622270073"},
        "left":    {"type": "realsense", "serial": "128422270260"},
        "right":   {"type": "realsense", "serial": "126122270644"},
        # Gripper throttled to 15 fps to halve USB backpressure during the
        # inter-trial gap (rest motion + ffmpeg encode = ~2-5s where nobody
        # drains frames). 30→15 doubles the time before the kernel ringbuffer
        # overflows. Color+depth share this rate — same pipeline blob.
        "gripper": {"type": "realsense", "serial": "123622270367", "depth": True, "fps": 30},
    },
    # Legacy: gripper on OpenCV V4L2 path. Pre-v016 inference setups used this
    # because `init_cameras()` exposed only color anyway. Kept for
    # backward-compat with scripts that explicitly request it.
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
    },
    "sim_4cam_opencv": {
        "top_cam":   {"type": "opencv", "path": "/dev/cam_top_raw"},
        "left_cam":  {"type": "opencv", "path": "/dev/cam_left_raw"},
        "right_cam": {"type": "opencv", "path": "/dev/cam_right_raw"},
        "wrist_cam": {"type": "opencv", "path": "/dev/cam_gripper_raw"},
    },
        "sim_4cam_realsense": {
        "top_cam":   {"type": "opencv", "path": "/dev/cam_top"},
        "left_cam":  {"type": "opencv", "path": "/dev/cam_left"},
        "right_cam": {"type": "opencv", "path": "/dev/cam_right"},
        "wrist_cam": {"type": "opencv", "path": "/dev/cam_gripper"},
    },
    # Hybrid backend used for v018+ (RGB+depth policies):
    # top/left/right stay on V4L2 (OpenCV) for low-latency single-stream capture,
    # gripper hops to librealsense so we can expose the D405's z16 depth stream
    # (V4L2 only sees the color side of the D405). Color and depth on the gripper
    # come from the same ``fs`` blob inside librealsense, so temporal+spatial
    # alignment between gripper RGB and gripper depth is preserved.
    # Camera-name keys match the v017/v018 training dataset feature names
    # (``observation.images.{top,left,right,gripper,gripper_depth}``).
    "opencv_depth": {
        "top":     {"type": "opencv",    "path": "/dev/cam_top_raw"},
        "left":    {"type": "opencv",    "path": "/dev/cam_left_raw"},
        "right":   {"type": "opencv",    "path": "/dev/cam_right_raw"},
        # Same throttle as ``realsense_depth`` — see comment there.
        "gripper": {"type": "realsense", "serial": "123622270367", "depth": True, "fps": 30},
    },
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

        # Per-camera fps override — used to throttle just the gripper D405
        # (which carries depth + has the deepest USB-hub branch and is the
        # first to stall under backpressure). Falls back to the global ``fps``.
        cam_fps = int(cfg.get("fps", fps))

        if cam_type == "realsense":
            serial      = cfg["serial"]
            wants_depth = bool(cfg.get("depth"))
            rs_cfg = rs.config()
            rs_cfg.enable_device(serial)
            rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, cam_fps)
            if wants_depth:
                # Color + depth must share the same fps within a single rs.pipeline
                # — they come back as one frameset and rs.align needs a coherent pair.
                rs_cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, cam_fps)
            pipe = rs.pipeline()
            try:
                profile = pipe.start(rs_cfg)
                for _ in range(5):
                    pipe.wait_for_frames(timeout_ms=2000)
                entry = {
                    "type":       "realsense",
                    "pipe":       pipe,
                    "last_frame": None,
                }
                if wants_depth:
                    # Aligner is what registers depth onto the color image plane.
                    # Without it, depth comes from the depth-sensor's own intrinsics
                    # and won't pixel-match the RGB the policy sees.
                    entry["align"]       = rs.align(rs.stream.color)
                    entry["depth_scale"] = profile.get_device().first_depth_sensor().get_depth_scale()
                    entry["last_depth"]  = None
                    print(f"  {name}: RealSense serial={serial} @{cam_fps}fps OK  (+depth, scale={entry['depth_scale']} m/unit)")
                else:
                    print(f"  {name}: RealSense serial={serial} @{cam_fps}fps OK")
                cameras[name] = entry
            except Exception as e:
                print(f"  [ERR] {name}: RealSense {serial} — {e}")

        elif cam_type == "opencv":
            path = cfg.get("path", cfg.get("index", 0))
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, cam_fps)
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

    For realsense cameras flagged with ``depth=True`` at init, the aligned
    depth frame is also pulled from the SAME ``fs`` blob (so it stays temporally
    + spatially registered with the color frame) and stashed into
    ``cam["last_depth"]`` as uint16 (H, W). Retrieve via :func:`get_latest_depth`.
    """
    frames = {}
    for name, cam in cameras.items():
        try:
            if cam["type"] == "realsense":
                ret, fs = cam["pipe"].try_wait_for_frames(timeout_ms=100)
                if ret:
                    if "align" in cam:
                        fs = cam["align"].process(fs)
                        df = fs.get_depth_frame()
                        if df:
                            cam["last_depth"] = np.asanyarray(df.get_data()).copy()
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


def get_latest_depth(cameras: dict) -> dict[str, np.ndarray]:
    """Return uint16 depth for every camera that has depth enabled.

    Reads from the ``last_depth`` slot populated by :func:`capture_frames`
    (no I/O, no extra device fetch — just an accessor). Cameras without
    depth or that haven't yielded a depth frame yet are omitted.

    Returns:
        {name: (H, W) uint16}  — hardware units; multiply by
        ``cameras[name]["depth_scale"]`` to get meters.
    """
    return {
        name: cam["last_depth"]
        for name, cam in cameras.items()
        if cam.get("last_depth") is not None
    }


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
                     joint_names: list[str] | None = None,
                     right_column: list[str] | None = None,
                     hud_lines: list[str] | None = None,
                     gato: bool = False) -> np.ndarray:
    """Build a 2x2 camera grid as BGR numpy array for display.

    Args:
        right_column: optional list of camera names to peel off and stack as
            a TALL RIGHT COLUMN at full grid height (resized to match). Use
            this for the live depth tile so it sits next to the gripper RGB
            instead of dropping under the 2x2 grid as a half-empty 3rd row.
    """
    right_set     = set(right_column or [])
    main_names    = [n for n in camera_names if n not in right_set]
    right_names   = [n for n in (right_column or []) if n in camera_names]

    def _tile(name: str) -> np.ndarray:
        if name in frames:
            img = cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR)
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(img, f"step: {step}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img

    # ── Main 2-col grid for the RGB cams ──
    grid_imgs = [_tile(n) for n in main_names]
    cols = 2
    rows = max(1, (len(grid_imgs) + cols - 1) // cols)
    while len(grid_imgs) < rows * cols:
        grid_imgs.append(np.zeros((height, width, 3), dtype=np.uint8))
    grid_rows = [np.hstack(grid_imgs[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    # ── Right column: each tile keeps natural (height, width); pad ABOVE ──
    # with zeros so the right tile sits in the BOTTOM row, visually paired with
    # the gripper RGB tile that lives in the bottom-right of the main 2x2 grid.
    if right_names:
        gh       = grid.shape[0]
        col_imgs = [_tile(n) for n in right_names]
        used_h   = sum(t.shape[0] for t in col_imgs)
        if used_h < gh:
            col_imgs.insert(0, _gato_placeholder(width, gh - used_h) if gato
                            else np.zeros((gh - used_h, width, 3), dtype=np.uint8))
        elif used_h > gh:
            # Only happens if right column has more tiles than main rows.
            # Trim the last tile down to fit exactly.
            excess = used_h - gh
            last   = col_imgs[-1]
            col_imgs[-1] = last[: max(1, last.shape[0] - excess)]
        right_col = np.vstack(col_imgs)
        grid = np.hstack([grid, right_col])

    if action is not None and joint_names is not None:
        y_start = height + 50
        for j, (jname, val) in enumerate(zip(joint_names, action)):
            text = f"{jname[:8]:>8}: {val:7.1f}"
            cv2.putText(grid, text, (width + 10, y_start + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if hud_lines:
        y = 55
        for line in hud_lines:
            cv2.putText(grid, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            y += 24

    return grid


def _gato_placeholder(width: int, height: int) -> np.ndarray:
    global _GATO_TILE
    if _GATO_TILE is None:
        _GATO_TILE = cv2.imread(str(GATO_IMAGE), cv2.IMREAD_COLOR)
    if _GATO_TILE is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = max(width / _GATO_TILE.shape[1], height / _GATO_TILE.shape[0])
    resized = cv2.resize(_GATO_TILE, (
        int(_GATO_TILE.shape[1] * scale),
        int(_GATO_TILE.shape[0] * scale),
    ))
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return resized[y0:y0 + height, x0:x0 + width]


def show_camera_grid(frames: dict[str, np.ndarray], camera_names: list[str],
                     step: int = 0, action: np.ndarray | None = None,
                     width: int = 640, height: int = 480,
                     joint_names: list[str] | None = None,
                     window_name: str = "Cameras",
                     right_column: list[str] | None = None,
                     hud_lines: list[str] | None = None,
                     gato: bool = False) -> int:
    """Display camera grid and return key press."""
    grid = build_grid_frame(frames, camera_names, step, action, width, height,
                            joint_names, right_column=right_column,
                            hud_lines=hud_lines, gato=gato)
    cv2.imshow(window_name, grid)
    return cv2.waitKey(1) & 0xFF
