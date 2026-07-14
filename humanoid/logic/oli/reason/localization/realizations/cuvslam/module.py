"""cuvslam — `LocalizationModule` adapter around PyCuVSLAM standalone visual odometry.

Pure L1 (local odometry), known-start: no loop closure, no relocalization — the map never
re-anchors the pose after t=0, so drift is unbounded BY DESIGN (see README hypothesis).

Input mapping (decided it-2, journal): cuVSLAM RGBD mode fed by ONE camera (config
`input.camera`, default head — forward-looking, pitch 0). `CameraFrame.rgb` (H,W,3 uint8)
goes in as the color image; `CameraFrame.depth` (float32 meters) is converted to uint16
depth units (cuVSLAM hard-requires uint16) using `odometry.rgbd_settings.depth_scale_factor`
(units per meter, default 1000 = millimeters). Intrinsics ride on each frame, so the rig +
tracker are built lazily on the FIRST stepped frame, not in `start()`. IMU is off in v0.

Frames: the rig frame is the camera OPTICAL frame (x right, y down, z forward;
`rig_from_camera` = identity). The constant base←camera mount comes from
`oli.camera_mounts` (D10 single source of truth; head joints sit at 0 in glide). The fixed
VO→map SE(3) is anchored on the first tracked frame: T_map_vo = T_map_base0 · T_base_cam ·
T_vo_cam(t0)⁻¹, then every step answers T_map_base(t) = T_map_vo · T_vo_cam(t) · T_base_cam⁻¹,
flattened SE(3)→SE(2) (x, y, yaw about +Z).

Status semantics: a valid VO pose is DRIFTING (propagated since the t=0 map fix — the honest
reading of the contract for pure VO; the scorer counts any non-None pose as coverage),
`last_fix_stamp_ns` = the anchor stamp. Tracking failure (world_from_rig is None) or a missing
warm start is LOST. A raising tracker (bad rig, malformed input) marks the module dead →
LOST for the rest of the episode (the host gets a fresh instance next episode).

Tunables all live in `config.yaml`; contract + invariants: `../../AGENTS.md`; loop
discipline: `../AGENTS.md`.
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import numpy as np

from ...contracts import (
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)
from .....camera_mounts import CAMERAS, CameraMount

import cuvslam
from cuvslam.utils import _apply_odometry_section


# ── SE(3) helpers (numpy only — no scipy in bench-cuvslam) ───────────────────────────


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _se3(rot: np.ndarray, trans) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(trans, dtype=float)
    return t


def _inv(t: np.ndarray) -> np.ndarray:
    rot, trans = t[:3, :3], t[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def _base_from_camera(mount: CameraMount) -> np.ndarray:
    """Constant base←camera-optical SE(3) from the shared mount table.

    Optical axes in the base frame (x fwd, y left, z up), level camera:
    x_opt(right) = -y_base, y_opt(down) = -z_base, z_opt(fwd) = +x_base; then the mount's
    pitch-down tilt about +Y_base. Valid in glide: head/waist joints hold 0 (nominal chain).
    """
    r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    a = math.radians(mount.pitch_down_deg)
    ry = np.array([
        [math.cos(a), 0.0, math.sin(a)],
        [0.0, 1.0, 0.0],
        [-math.sin(a), 0.0, math.cos(a)],
    ])
    return _se3(ry @ r0, mount.pos_base)


def _pose_to_se3(pose: RobotPose) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return _se3(rot, [pose.x, pose.y, 0.0])


class CuvslamModule:
    """PyCuVSLAM RGBD visual odometry behind the `LocalizationModule` protocol."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = dict(config or {})
        inp = dict(self._config.get("input") or {})
        self._camera_name = str(inp.get("camera", "head"))
        mounts = {m.name: m for m in CAMERAS}
        if self._camera_name not in mounts:
            raise ValueError(
                f"input.camera {self._camera_name!r} not in mounts {sorted(mounts)}")
        self._t_base_cam = _base_from_camera(mounts[self._camera_name])
        self._t_cam_base = _inv(self._t_base_cam)

        odo = dict(self._config.get("odometry") or {})
        rgbd = dict(odo.get("rgbd_settings") or {})
        self._depth_units_per_m = float(rgbd.get("depth_scale_factor", 1000.0))

        # episode state (start() resets; the host builds a fresh instance per episode)
        self._initial_pose: Optional[RobotPose] = None
        self._tracker = None
        self._t_map_vo: Optional[np.ndarray] = None
        self._anchor_stamp: Optional[int] = None
        self._last_pose: Optional[RobotPose] = None
        self._dead: Optional[str] = None

    # ── LocalizationModule ────────────────────────────────────────────────────────

    def start(self, setup: LocalizationSetup) -> None:
        # Pure VO: setup.map_dir is unused (no map artifacts); known-start seed is the
        # whole anchor. Tracker construction is lazy — intrinsics arrive with frames.
        self._initial_pose = setup.initial_pose
        self._tracker = None
        self._t_map_vo = None
        self._anchor_stamp = None
        self._last_pose = None
        self._dead = None

    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        stamp = loc_in.stamp_ns
        if self._dead is not None or self._initial_pose is None:
            return self._lost(stamp)

        frame = loc_in.frames.get(self._camera_name)
        if frame is None:
            return self._carry(stamp)   # our camera didn't tick — age the last verdict

        try:
            if self._tracker is None:
                self._build_tracker(frame)
            assert self._tracker is not None
            est, _ = self._tracker.track(
                stamp,
                images=[np.ascontiguousarray(frame.rgb)],
                depths=[self._depth_to_units(frame.depth)],
            )
        except Exception as exc:  # noqa: BLE001 — structural failure: dead for the episode
            self._dead = f"cuvslam tracker failed: {exc}"
            print(f"[cuvslam] {self._dead}", file=sys.stderr)
            return self._lost(stamp)

        world_from_rig = est.world_from_rig
        if world_from_rig is None:
            return self._lost(stamp)    # tracking lost this tick (cuVSLAM may recover)

        pose = world_from_rig.pose
        t_vo_cam = _se3(_quat_xyzw_to_mat(pose.rotation), pose.translation)
        if self._t_map_vo is None:      # first tracked frame = the map anchor (known start)
            self._t_map_vo = (
                _pose_to_se3(self._initial_pose) @ self._t_base_cam @ _inv(t_vo_cam)
            )
            self._anchor_stamp = stamp
        t_map_base = self._t_map_vo @ t_vo_cam @ self._t_cam_base
        self._last_pose = RobotPose(
            stamp_ns=stamp,
            x=float(t_map_base[0, 3]),
            y=float(t_map_base[1, 3]),
            yaw=math.atan2(float(t_map_base[1, 0]), float(t_map_base[0, 0])),
        )
        return LocalizationOut(
            stamp_ns=stamp,
            pose=self._last_pose,
            status=LocalizationStatus.DRIFTING,
            last_fix_stamp_ns=self._anchor_stamp,
        )

    def stop(self) -> None:
        # Must tolerate stop() after a failed/absent start() (testing.py teardown contract).
        self._tracker = None            # drops the GPU tracker; GC frees the CUDA context
        self._t_map_vo = None
        self._last_pose = None

    # ── internals ─────────────────────────────────────────────────────────────────

    def _build_tracker(self, frame) -> None:
        intr = frame.intrinsics
        cam = cuvslam.Camera()
        cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
        cam.focal = (intr.fx, intr.fy)
        cam.principal = (intr.cx, intr.cy)
        cam.size = (intr.width, intr.height)
        rig = cuvslam.Rig()
        rig.cameras = [cam]

        odom_cfg = cuvslam.Tracker.OdometryConfig()
        section = dict(self._config.get("odometry") or {})
        if section:
            _apply_odometry_section(odom_cfg, section)
        self._tracker = cuvslam.Tracker(rig, odom_cfg)

    def _depth_to_units(self, depth: np.ndarray) -> np.ndarray:
        units = np.nan_to_num(
            np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        ) * self._depth_units_per_m
        return np.ascontiguousarray(np.clip(units, 0.0, 65535.0).astype(np.uint16))

    def _lost(self, stamp: int) -> LocalizationOut:
        return LocalizationOut(
            stamp_ns=stamp, pose=None, status=LocalizationStatus.LOST,
            last_fix_stamp_ns=self._anchor_stamp,
        )

    def _carry(self, stamp: int) -> LocalizationOut:
        if self._last_pose is None:
            return self._lost(stamp)
        return LocalizationOut(
            stamp_ns=stamp,
            pose=RobotPose(stamp_ns=stamp, x=self._last_pose.x, y=self._last_pose.y,
                           yaw=self._last_pose.yaw),
            status=LocalizationStatus.DRIFTING,
            last_fix_stamp_ns=self._anchor_stamp,
        )


def build(config: dict) -> CuvslamModule:
    """Registry entrypoint: `config` = parsed config.yaml + caller overrides."""
    return CuvslamModule(config)
