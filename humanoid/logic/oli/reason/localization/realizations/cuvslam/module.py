"""cuvslam — `LocalizationModule` adapter around PyCuVSLAM.

Two modes (config `mode`, it-3 2026-07-17):

**map_relative** (the demo candidate): stereo pair (`input.stereo`) against a prior map
built by `build_map.py` (`LocalizationSetup.map_dir` = the pycuvslam_map dir). First
complete frame pair: build the VEHICLE-frame rig (rig_from_camera = mount-table
base←optical, the exact rig the map builder used — rig identity is part of the map
contract), track once, then `localize_in_map` around the hint. The hint arrives in the
WORLD frame (`setup.initial_pose`, GT start or map click) and is converted into the map
frame via the map's `registration_gt.json` (rigid SE(2), computed by build_map's audit);
every emitted pose converts back — so downstream sees WORLD-frame (occupancy-grid-frame)
poses, per the RobotPose contract, and the module owns all alignment internally.

    hint_W ──inv(reg)──► hint_M ──localize_in_map──► SLAM anchored to map
    track(t) ──► slam_pose (vehicle in M) ──reg──► RobotPose in W   [TRACKING]

With `localization.relocalize_period_s` > 0 (it-8), tracking additionally self-relocalizes
every period: mid-tracking `localize_in_map` with the module's OWN current slam_pose as the
prior — reconciles accumulated drift on success, non-destructive on refusal (cuVSLAM keeps
the current map; nvidia-corpus://cuvslam-api/cpp/a00138#section-2). No GT anywhere.
MEASURED CAVEAT (it-8, JOURNAL): in vendored v16 the reloc FIX is mm-accurate but the
post-localize state swap (async_slam_localize.cpp swaps the whole SLAM graph) degrades
subsequent tracking on this replay — see the journal before trusting a nonzero period.

Status semantics: TRACKING = SLAM pose valid post-localization (continuously
re-anchored by stored-landmark matches); LOST = tracking invalid this tick (vendored
v16 has NO self-recovery — the operator re-hint path boots a FRESH instance via the
host). A failed localize or missing registration marks the module dead → LOST forever
this episode, never a raise (re-hint safety). DRIFTING is not yet distinguished
(needs per-frame anchor telemetry; refinement noted in JOURNAL).

**rgbd_vo** (the frozen it-2 L1 baseline): single-camera RGBD visual odometry,
known-start anchored, unbounded drift by design. Kept verbatim for locbench
reproducibility — see the it-2 journal entry and README hypothesis.

Tunables all live in `config.yaml`; contract + invariants: `../../AGENTS.md`; loop
discipline: `../AGENTS.md`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ...contracts import (
    CameraFrame,
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)
from .....camera_mounts import CAMERAS, STEREO_CAMERAS, CameraMount

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


def _mat_to_quat_xyzw(m: np.ndarray) -> list:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-8:
        return [(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                (m[1, 0] - m[0, 1]) / (4 * w), w]
    x = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    return [x, (m[0, 1] + m[1, 0]) / (4 * x), (m[0, 2] + m[2, 0]) / (4 * x),
            (m[2, 1] - m[1, 2]) / (4 * x)]


def _yaw_from_quat_xyzw(q) -> float:
    return math.atan2(2 * (q[3] * q[2] + q[0] * q[1]),
                      1 - 2 * (q[1] ** 2 + q[2] ** 2))


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


class _Registration:
    """Rigid SE(2) world←map bridge from the map's registration_gt.json (build_map audit).

    pose_W = R @ [x, y]_M + t ;  yaw_W = yaw_M + theta.  Pure data — the module applies
    it both ways so all alignment stays internal (contract: downstream never refits).
    """

    def __init__(self, R: np.ndarray, t: np.ndarray, theta: float) -> None:
        self.R, self.t, self.theta = R, t, theta

    @classmethod
    def load(cls, path: Path) -> "_Registration":
        d = json.loads(path.read_text())
        return cls(np.array(d["R"], dtype=float), np.array(d["t"], dtype=float),
                   float(d["theta_rad"]))

    @classmethod
    def identity(cls) -> "_Registration":
        """W ≡ M — for maps whose artifacts were baked into the cuVSLAM frame."""
        return cls(np.eye(2), np.zeros(2), 0.0)

    def w_from_m(self, x: float, y: float, yaw: float) -> tuple:
        p = self.R @ np.array([x, y]) + self.t
        return float(p[0]), float(p[1]), yaw + self.theta

    def m_from_w(self, x: float, y: float, yaw: float) -> tuple:
        p = self.R.T @ (np.array([x, y]) - self.t)
        return float(p[0]), float(p[1]), yaw - self.theta


class CuvslamModule:
    """PyCuVSLAM behind the `LocalizationModule` protocol (map_relative | rgbd_vo)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = dict(config or {})
        self._mode = str(self._config.get("mode", "rgbd_vo"))
        if self._mode not in ("map_relative", "rgbd_vo"):
            raise ValueError(f"unknown mode {self._mode!r}")

        inp = dict(self._config.get("input") or {})
        mounts = {m.name: m for m in (*CAMERAS, *STEREO_CAMERAS)}

        if self._mode == "rgbd_vo":
            self._camera_name = str(inp.get("camera", "head"))
            if self._camera_name not in mounts:
                raise ValueError(
                    f"input.camera {self._camera_name!r} not in mounts {sorted(mounts)}")
            self._t_base_cam = _base_from_camera(mounts[self._camera_name])
            self._t_cam_base = _inv(self._t_base_cam)
            odo = dict(self._config.get("odometry") or {})
            rgbd = dict(odo.get("rgbd_settings") or {})
            self._depth_units_per_m = float(rgbd.get("depth_scale_factor", 1000.0))
        else:
            self._stereo = [str(n) for n in
                            (inp.get("stereo") or ["head_left", "head_right"])]
            missing = [n for n in self._stereo if n not in mounts]
            if missing:
                raise ValueError(f"input.stereo {missing} not in mounts {sorted(mounts)}")
            self._rig_from_cam = {n: _base_from_camera(mounts[n]) for n in self._stereo}

        # episode state (start() resets; the host builds a fresh instance per episode)
        self._setup: Optional[LocalizationSetup] = None
        self._initial_pose: Optional[RobotPose] = None
        self._tracker = None
        self._dead: Optional[str] = None
        self._last_pose: Optional[RobotPose] = None
        self._anchor_stamp: Optional[int] = None
        # rgbd_vo:
        self._t_map_vo: Optional[np.ndarray] = None
        # map_relative:
        self._reg: Optional[_Registration] = None
        self._localized = False
        self._pending: Dict[str, CameraFrame] = {}   # latest frame per stereo side
        self._last_pair_stamp = -1                   # last stamp fed to track()
        self._reloc_period_ns = 0                    # 0 = periodic self-reloc off
        self._reloc_ok = 0                           # it-8 counters (diagnostics)
        self._reloc_fail = 0
        self._last_reloc_stamp: Optional[int] = None  # last localize attempt (SIM time)
        # display-only side channel (dev_app Localization panel): dict cached during
        # step() on the host thread, read cross-thread via `diagnostics()` — a GIL-atomic
        # ref swap, never mutated in place. NOT part of the LocalizationModule contract.
        self._latest_diag: Optional[dict] = None

    # ── LocalizationModule ────────────────────────────────────────────────────────

    def start(self, setup: LocalizationSetup) -> None:
        self._setup = setup
        self._initial_pose = setup.initial_pose
        self._tracker = None
        self._dead = None
        self._last_pose = None
        self._anchor_stamp = None
        self._t_map_vo = None
        self._reg = None
        self._localized = False
        self._pending = {}
        self._last_pair_stamp = -1
        self._reloc_ok = 0
        self._reloc_fail = 0
        self._last_reloc_stamp = None
        if self._mode == "map_relative":
            loc_cfg = dict(self._config.get("localization") or {})
            self._reloc_period_ns = int(
                float(loc_cfg.get("relocalize_period_s", 0.0)) * 1e9)
            reg_name = loc_cfg.get("registration_file", "registration_gt.json")
            if not reg_name:
                # registration_file: null → M IS the world (bake-time alignment, Anton
                # 17-07): every map artifact was baked into the cuVSLAM frame, so poses
                # pass through raw and the hint arrives already in M (start_pose.json).
                self._reg = _Registration.identity()
            else:
                reg_path = Path(setup.map_dir) / str(reg_name)
                try:
                    self._reg = _Registration.load(reg_path)
                except Exception as exc:  # noqa: BLE001 — bad map artifacts = dead episode
                    self._die(f"registration load failed ({reg_path}): {exc}")

    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        if self._mode == "map_relative":
            return self._step_map_relative(loc_in)
        return self._step_rgbd_vo(loc_in)

    def stop(self) -> None:
        # Must tolerate stop() after a failed/absent start() (testing.py teardown contract).
        self._tracker = None            # drops the GPU tracker; GC frees the CUDA context
        self._t_map_vo = None
        self._last_pose = None
        self._pending = {}

    # ── map_relative ──────────────────────────────────────────────────────────────

    def _step_map_relative(self, loc_in: LocalizationIn) -> LocalizationOut:
        stamp = loc_in.stamp_ns
        if self._dead is not None or self._initial_pose is None or self._reg is None:
            return self._lost(stamp)

        for name in self._stereo:
            f = loc_in.frames.get(name)
            if f is not None:
                self._pending[name] = f
        left = self._pending.get(self._stereo[0])
        right = self._pending.get(self._stereo[1])
        if left is None or right is None or left.stamp_ns != right.stamp_ns:
            return self._carry(stamp)   # incomplete/mismatched pair — age last verdict
        if left.stamp_ns <= self._last_pair_stamp:
            return self._carry(stamp)   # an unrelated stream ticked (chest/head) — the
            # host steps us anyway; re-tracking a stale pair violates cuVSLAM's
            # strictly-increasing-timestamp requirement (live-fire find, 17-07)
        self._last_pair_stamp = left.stamp_ns

        pair = (left, right)
        imgs = [np.ascontiguousarray(f.rgb) for f in pair]
        pair_stamp = left.stamp_ns
        try:
            if self._tracker is None:
                self._build_stereo_tracker(pair)
            assert self._tracker is not None
            est, slam_pose = self._tracker.track(pair_stamp, imgs)
            if not self._localized:
                h = self._initial_pose
                loc_pose = self._localize(
                    pair_stamp, imgs, self._reg.m_from_w(h.x, h.y, h.yaw))
                if loc_pose is not None:
                    slam_pose = loc_pose   # emit the anchored fix, not the pre-seat pose
            elif (self._reloc_period_ns > 0 and est.world_from_rig is not None
                    and self._last_reloc_stamp is not None
                    and pair_stamp - self._last_reloc_stamp >= self._reloc_period_ns):
                # periodic self-relocalize (it-8): prior = the tracker's OWN slam_pose —
                # already in M, no registration roundtrip, no GT anywhere in this path.
                guess_m = (float(slam_pose.translation[0]),
                           float(slam_pose.translation[1]),
                           _yaw_from_quat_xyzw(slam_pose.rotation))
                loc_pose = self._localize(pair_stamp, imgs, guess_m)
                if loc_pose is not None:
                    slam_pose = loc_pose   # emit the reconciled pose this step
        except Exception as exc:  # noqa: BLE001 — structural failure: dead for the episode
            self._die(f"cuvslam tracker failed: {exc}")
            return self._lost(stamp)
        if self._dead is not None:
            return self._lost(stamp)

        if est.world_from_rig is None:
            return self._lost(stamp)    # tracking lost this tick (re-hint = fresh episode)

        self._cache_diagnostics(stamp, imgs[0])

        p = slam_pose  # map-frame vehicle pose (localize_in_map re-seated the SLAM state)
        yaw_m = _yaw_from_quat_xyzw(p.rotation)
        x_w, y_w, yaw_w = self._reg.w_from_m(
            float(p.translation[0]), float(p.translation[1]), yaw_m)
        self._last_pose = RobotPose(stamp_ns=stamp, x=x_w, y=y_w, yaw=yaw_w)
        return LocalizationOut(
            stamp_ns=stamp,
            pose=self._last_pose,
            status=LocalizationStatus.TRACKING,
            last_fix_stamp_ns=self._anchor_stamp,
        )

    def _build_stereo_tracker(self, pair) -> None:
        cams = []
        for f, name in zip(pair, self._stereo):
            intr = f.intrinsics
            cam = cuvslam.Camera()
            cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
            cam.focal = (intr.fx, intr.fy)
            cam.principal = (intr.cx, intr.cy)
            cam.size = (intr.width, intr.height)
            t = self._rig_from_cam[name]
            cam.rig_from_camera = cuvslam.Pose(
                translation=list(t[:3, 3]), rotation=_mat_to_quat_xyzw(t[:3, :3]))
            cams.append(cam)
        slam_cfg_d = dict(self._config.get("slam") or {})
        odom_cfg = cuvslam.Tracker.OdometryConfig(
            async_sba=bool(slam_cfg_d.get("async_sba", False)),
            rectified_stereo_camera=bool(
                slam_cfg_d.get("rectified_stereo_camera", True)),
        )
        slam_cfg = cuvslam.Tracker.SlamConfig(
            sync_mode=bool(slam_cfg_d.get("sync_mode", True)),
            max_map_size=int(slam_cfg_d.get("max_map_size", 0)),  # 0 = unlimited —
            # MUST match build_map (default 300 caps the loaded map; localize refuses)
        )
        self._tracker = cuvslam.Tracker(cuvslam.Rig(cams), odom_cfg, slam_cfg)

    def _cache_diagnostics(self, stamp: int, left_rgb) -> None:
        """Cache the display-only diagnostics dict (tracked features + SLAM metrics + the
        EXACT frame the tracker consumed, 2× downscaled — dots and pixels must be the same
        instant or the overlay visibly trails during motion; dev_app finding 17-07).

        Called on the host thread right after track(); failures are swallowed — the
        panel losing a frame of dots must never darken the pose path.
        """
        try:
            obs = self._tracker.get_last_observations(0)   # camera 0 = head_left
            m = self._tracker.get_slam_metrics()
            # len(get_loop_closure_poses()) is a MONOTONE event counter: it grows only when
            # a loop closure actually re-anchored us against the map — the drift-diagnosis
            # signal (lc_status is just the latest-sample flag) (17-07)
            lc_events = len(self._tracker.get_loop_closure_poses() or [])
            self._latest_diag = {
                "stamp_ns": stamp,
                "observations": [(float(o.u), float(o.v), int(o.id)) for o in obs],
                "rgb": np.ascontiguousarray(left_rgb[::2, ::2]),  # tracker's own frame, ½ res
                "rgb_scale": 0.5,                                 # obs u/v are full-res
                "lc_status": bool(getattr(m, "lc_status", False)) if m else None,
                "pgo_status": bool(getattr(m, "pgo_status", False)) if m else None,
                "lc_events": lc_events,
                "lc_good_landmarks": int(getattr(m, "lc_good_landmarks_count", 0)) if m else 0,
                "lc_selected_landmarks": int(
                    getattr(m, "lc_selected_landmarks_count", 0)) if m else 0,
                "lc_pnp_landmarks": int(getattr(m, "lc_pnp_landmarks_count", 0)) if m else 0,
                "localized": self._localized,
                "reloc_ok": self._reloc_ok,
                "reloc_fail": self._reloc_fail,
                "last_reloc_stamp": self._last_reloc_stamp,
            }
        except Exception:  # noqa: BLE001 — diagnostics are best-effort by design
            pass

    def diagnostics(self) -> Optional[dict]:
        """Latest display-only diagnostics (NOT part of the module contract)."""
        return self._latest_diag

    def _localize(self, stamp: int, imgs, guess_m: tuple) -> Optional[object]:
        """Run localize_in_map around `guess_m` = (x, y, yaw) prior in the map frame M;
        returns the anchored rig pose or None.

        Two callers, two refusal semantics: the START fix (prior = hint W→M) dies —
        no anchor means the episode is unusable; a PERIODIC self-reloc (prior = own
        slam_pose) swallows the refusal and counts it — cuVSLAM keeps the current map
        (corpus §LocalizeInMap), tracking continues, retry next period.
        """
        assert (self._reg is not None and self._tracker is not None
                and self._setup is not None)
        periodic = self._localized
        loc_cfg = dict(self._config.get("localization") or {})
        settings = cuvslam.Tracker.SlamLocalizationSettings(
            horizontal_search_radius=float(loc_cfg.get("horizontal_search_radius", 4.0)),
            vertical_search_radius=float(loc_cfg.get("vertical_search_radius", 1.0)),
            horizontal_step=float(loc_cfg.get("horizontal_step", 0.25)),
            vertical_step=float(loc_cfg.get("vertical_step", 0.25)),
            angular_step_rads=float(loc_cfg.get("angular_step_rads", 0.1)),
        )
        x_m, y_m, yaw_m = guess_m
        half = yaw_m / 2.0
        guess = cuvslam.Pose(translation=[x_m, y_m, 0.0],
                             rotation=[0.0, 0.0, math.sin(half), math.cos(half)])
        result: Dict[str, object] = {}

        def finish(pose, err):
            result["pose"], result["err"] = pose, err

        # sync_mode=True → the finish callback fires within this call
        self._tracker.localize_in_map(
            str(self._setup.map_dir), stamp, guess, imgs, settings,
            lambda: None, finish)
        # attempt-paced timer: a refusal also waits a full period before the retry;
        # the start fix seeds the period clock
        self._last_reloc_stamp = stamp
        pose = result.get("pose")
        if pose is None:
            if periodic:
                self._reloc_fail += 1
            else:
                self._die(f"localize_in_map failed: {result.get('err')!r}")
            return None
        if periodic:
            self._reloc_ok += 1
        self._localized = True
        self._anchor_stamp = stamp
        return pose

    # ── rgbd_vo (it-2 baseline, verbatim) ─────────────────────────────────────────

    def _step_rgbd_vo(self, loc_in: LocalizationIn) -> LocalizationOut:
        stamp = loc_in.stamp_ns
        if self._dead is not None or self._initial_pose is None:
            return self._lost(stamp)

        frame = loc_in.frames.get(self._camera_name)
        if frame is None:
            return self._carry(stamp)   # our camera didn't tick — age the last verdict

        try:
            if self._tracker is None:
                self._build_rgbd_tracker(frame)
            assert self._tracker is not None
            est, _ = self._tracker.track(
                stamp,
                images=[np.ascontiguousarray(frame.rgb)],
                depths=[self._depth_to_units(frame.depth)],
            )
        except Exception as exc:  # noqa: BLE001 — structural failure: dead for the episode
            self._die(f"cuvslam tracker failed: {exc}")
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

    def _build_rgbd_tracker(self, frame) -> None:
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

    # ── shared verdict helpers ────────────────────────────────────────────────────

    def _die(self, message: str) -> None:
        self._dead = message
        print(f"[cuvslam] {message}", file=sys.stderr)

    def _lost(self, stamp: int) -> LocalizationOut:
        return LocalizationOut(
            stamp_ns=stamp, pose=None, status=LocalizationStatus.LOST,
            last_fix_stamp_ns=self._anchor_stamp,
        )

    def _carry(self, stamp: int) -> LocalizationOut:
        if self._last_pose is None:
            return self._lost(stamp)
        status = (LocalizationStatus.TRACKING if self._mode == "map_relative"
                  and self._localized else LocalizationStatus.DRIFTING)
        return LocalizationOut(
            stamp_ns=stamp,
            pose=RobotPose(stamp_ns=stamp, x=self._last_pose.x, y=self._last_pose.y,
                           yaw=self._last_pose.yaw),
            status=status,
            last_fix_stamp_ns=self._anchor_stamp,
        )


def build(config: dict) -> CuvslamModule:
    """Registry entrypoint: `config` = parsed config.yaml + caller overrides."""
    return CuvslamModule(config)
