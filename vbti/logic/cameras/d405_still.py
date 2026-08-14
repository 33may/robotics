#!/usr/bin/env python3
"""Still-frame capture for a single D405: color + depth + raw IR stereo pair.

The raw IR pair is the point: it lets learned stereo (Fast-FoundationStereo)
be compared against native depth later on byte-identical frames.
"""

import time

import numpy as np

# 848x480 is the D405's optimal depth resolution (per Intel tuning guide).
WIDTH, HEIGHT, FPS = 848, 480, 15


def open_camera(serial: str, width: int = WIDTH, height: int = HEIGHT,
                fps: int = FPS):
    """Start one D405 pipeline with color + depth + both IR imagers.

    Returns (pipe, profile, align, depth_scale).
    """
    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    cfg.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Let auto-exposure converge before anyone captures.
    for _ in range(10):
        pipe.wait_for_frames(timeout_ms=2000)
    return pipe, profile, align, depth_scale


def _intrinsics_dict(intr) -> dict:
    return {"width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "model": str(intr.model), "coeffs": list(intr.coeffs)}


def session_metadata(profile, depth_scale: float, serial: str) -> dict:
    """Intrinsics for every stream + IR1->IR2 extrinsics (stereo baseline)."""
    import pyrealsense2 as rs

    def stream(s, idx=0):
        for sp in profile.get_streams():
            if sp.stream_type() == s and (idx == 0 or sp.stream_index() == idx):
                return sp.as_video_stream_profile()
        raise RuntimeError(f"stream {s} idx {idx} not found")

    color, depth = stream(rs.stream.color), stream(rs.stream.depth)
    ir1, ir2 = stream(rs.stream.infrared, 1), stream(rs.stream.infrared, 2)
    ext = ir1.get_extrinsics_to(ir2)  # baseline lives in translation[0]

    ivs = color.get_intrinsics()
    return {
        "serial": serial,
        "resolution": [ivs.width, ivs.height],
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {
            "color": _intrinsics_dict(color.get_intrinsics()),
            "depth": _intrinsics_dict(depth.get_intrinsics()),
            "ir_left": _intrinsics_dict(ir1.get_intrinsics()),
            "ir_right": _intrinsics_dict(ir2.get_intrinsics()),
        },
        "extrinsics_ir1_to_ir2": {
            "rotation": list(ext.rotation),
            "translation_m": list(ext.translation),
        },
    }


def capture_bundle(pipe, align, settle_s: float = 0.1, flush: int = 3) -> dict:
    """Grab one complete frame bundle. Call only with the camera stationary.

    Flushes stale frames first so the bundle postdates arrival at the pose.
    Returns {"rgb", "ir_left", "ir_right", "depth_raw", "depth_aligned",
    "timestamp"} — arrays are copies, safe to hold.
    """
    time.sleep(settle_s)
    for _ in range(flush):
        pipe.wait_for_frames(timeout_ms=2000)

    fs = pipe.wait_for_frames(timeout_ms=2000)
    ts = time.time()

    depth_raw = np.asanyarray(fs.get_depth_frame().get_data()).copy()
    ir_left = np.asanyarray(fs.get_infrared_frame(1).get_data()).copy()
    ir_right = np.asanyarray(fs.get_infrared_frame(2).get_data()).copy()

    fs_aligned = align.process(fs)
    rgb = np.asanyarray(fs_aligned.get_color_frame().get_data()).copy()
    depth_aligned = np.asanyarray(fs_aligned.get_depth_frame().get_data()).copy()

    return {"rgb": rgb, "ir_left": ir_left, "ir_right": ir_right,
            "depth_raw": depth_raw, "depth_aligned": depth_aligned,
            "timestamp": ts}
