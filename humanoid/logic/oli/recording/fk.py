"""fk.py — brain-side camera FK + rig description (MAY-173 slam-demo-loop 1.4).

The Robot-side recorder cannot read camera prims, so each frame's `T_world_cam`
(USD convention: camera views down local -Z, +Y up; column-vector 4×4) is derived
by FK: GT base pose (x, y, yaw from the debug-pose channel) ∘ the static mount
from `camera_mounts` — the SAME numbers `build_camera_usd._local_transform`
authored into the robot USD, so the composition mirrors what the renderer used.

Known constant offsets brain-side cannot see (glide hover height, settled torso
pitch) cancel by construction: `rosbag_synth.recover_static_mount` estimates the
base→cam mount FROM the recorded pairs, so only base↔cam CONSISTENCY matters —
which FK guarantees exactly (see test_static_mount_is_constant_across_poses).

Pure numpy + camera_mounts. No isaacsim.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from humanoid.logic.oli.camera_mounts import (
    CameraMount,
    D435I_STEREO_BASELINE_M,
    HEAD_CAM,
    STEREO_CAMERAS,
    rgb_intrinsics,
)


def T_world_base(x: float, y: float, yaw: float) -> np.ndarray:
    """Planar base pose → 4×4 world transform (z=0 — constant offsets are absorbed
    by the bake's mount recovery, see module docstring)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3] = float(x), float(y)
    return T


def T_base_cam_usd(mount: CameraMount) -> np.ndarray:
    """Static base→camera transform, USD camera convention — the column-vector
    twin of `build_camera_usd._local_transform` (which authors row-vector Gf):
    view axis +X tilted `pitch_down_deg` below horizontal, no roll; camera looks
    down local -Z; translation = the mount's base-frame position."""
    th = math.radians(mount.pitch_down_deg)
    view = np.array([math.cos(th), 0.0, -math.sin(th)])
    right = np.cross(view, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, view)
    T = np.eye(4)
    T[:3, 0] = right          # +X_local
    T[:3, 1] = up             # +Y_local
    T[:3, 2] = -view          # +Z_local (USD camera views down -Z)
    T[:3, 3] = np.asarray(mount.pos_base, dtype=float)
    return T


def cam_world(x: float, y: float, yaw: float, mount: CameraMount) -> np.ndarray:
    """Per-frame `T_world_cam` for the dump row: base FK ∘ static mount."""
    return T_world_base(x, y, yaw) @ T_base_cam_usd(mount)


def rig_dict(res: Tuple[int, int]) -> dict:
    """rig.json payload — everything the bake-side converter needs to build
    camera_info + extrinsics without importing this repo. Mirrors the scripted
    coverage drive's rig so the SAME container tooling eats both dumps."""
    def mount_entry(m: CameraMount) -> dict:
        intr = rgb_intrinsics(width=res[0], height=res[1], hfov_deg=m.hfov_deg)
        return {
            "parent_link": m.parent_link,
            "pos_base": [float(v) for v in m.pos_base],
            "pitch_down_deg": float(m.pitch_down_deg),
            "intrinsics": {"width": intr.width, "height": intr.height,
                           "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
        }
    return {
        "camera_axes": "usd (-Z view, +Y up); convert to ROS optical (+Z view) at bag synth",
        "stamp": "sim time ns (world.current_time)",
        "baseline_m": D435I_STEREO_BASELINE_M,
        "stereo_pair": ["head_left", "head_right"],
        "rgbd": ["head"],
        "cameras": {m.name: mount_entry(m) for m in (HEAD_CAM, *STEREO_CAMERAS)},
    }
