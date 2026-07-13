"""Contracts of the localization module (design.md D2–D6, may-173-reason-module-separation).

`LocalizationIn` is the frame-paced input bundle; `LocalizationOut` the map-frame verdict with
an explicit status machine (pose None iff LOST); `LocalizationSetup` the start-time materials;
`RobotPose` lives HERE now (localization's output type — deps point nav→localization).
Pure: runs in the `brain` env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Intent, Mode, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _frame(stamp_ns=1, name="head", w=4, h=3):
    return CameraFrame(
        stamp_ns=stamp_ns, name=name,
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=np.ones((h, w), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=w, height=h, fx=2.0, fy=2.0, cx=w / 2, cy=h / 2),
    )


# ── RobotPose (moved here from nav/types.py, behavior identical) ─────────────

def test_robot_pose_coerces_types():
    p = RobotPose(stamp_ns=1.0, x=1, y=2, yaw=0)
    assert isinstance(p.stamp_ns, int) and isinstance(p.x, float)
    assert (p.x, p.y, p.yaw) == (1.0, 2.0, 0.0)


def test_robot_pose_importable_from_nav_for_downstream():
    # nav consumes localization's output type — the import direction of the split
    from humanoid.logic.oli.reason.localization.contracts import RobotPose as RP
    assert RP is RobotPose


# ── LocalizationIn — the frame-paced bundle (D2) ─────────────────────────────

def test_localization_in_holds_frames_obs_and_optional_intent():
    li = LocalizationIn(stamp_ns=5, frames={"head": _frame(5)}, observation=_obs(4))
    assert li.stamp_ns == 5 and li.intent is None
    assert set(li.frames) == {"head"}
    li2 = LocalizationIn(
        stamp_ns=5, frames={"head": _frame(5)}, observation=_obs(4),
        intent=Intent(mode=Mode.WALK, v_x=0.1),
    )
    assert li2.intent is not None and li2.intent.v_x == pytest.approx(0.1)


def test_localization_in_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        LocalizationIn(stamp_ns=5, frames={}, observation=_obs(4))


def test_localization_in_coerces_stamp_and_is_frozen():
    li = LocalizationIn(stamp_ns=5.0, frames={"head": _frame(5)}, observation=_obs(4))
    assert isinstance(li.stamp_ns, int)
    with pytest.raises(Exception):
        li.stamp_ns = 6  # type: ignore[misc]


# ── LocalizationOut — map-frame verdict + status machine (D3, D4) ────────────

def test_out_tracking_carries_pose():
    out = LocalizationOut(
        stamp_ns=7, pose=RobotPose(stamp_ns=7, x=1.0), status=LocalizationStatus.TRACKING,
        last_fix_stamp_ns=7,
    )
    assert out.pose is not None and out.status is LocalizationStatus.TRACKING


def test_out_lost_means_no_pose_and_vice_versa():
    ok = LocalizationOut(stamp_ns=7, pose=None, status=LocalizationStatus.LOST)
    assert ok.pose is None and ok.last_fix_stamp_ns is None
    with pytest.raises(ValueError):
        LocalizationOut(stamp_ns=7, pose=None, status=LocalizationStatus.TRACKING)
    with pytest.raises(ValueError):
        LocalizationOut(
            stamp_ns=7, pose=RobotPose(stamp_ns=7), status=LocalizationStatus.LOST
        )


def test_out_drifting_carries_pose_and_older_fix():
    out = LocalizationOut(
        stamp_ns=9, pose=RobotPose(stamp_ns=9), status=LocalizationStatus.DRIFTING,
        last_fix_stamp_ns=3,
    )
    assert out.last_fix_stamp_ns == 3


# ── LocalizationSetup — start-time materials (D4, D5) ────────────────────────

def test_setup_holds_map_dir_hint_and_calibration(tmp_path: Path):
    s = LocalizationSetup(
        map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=1.0), calibration={"cam": "head"},
    )
    assert s.map_dir == tmp_path and s.initial_pose.x == 1.0
    s2 = LocalizationSetup(map_dir=tmp_path)  # hint + calibration optional
    assert s2.initial_pose is None and s2.calibration == {}
