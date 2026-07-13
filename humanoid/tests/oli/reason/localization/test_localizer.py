"""TDD for the localization `Localizer` seam + GroundTruthLocalizer (debug/eval backend).

Localization is *required*, not optional: `Observation` carries no world pose by design
(contracts.py — "the real robot has no ground-truth pose"), so the planner has no pose to
plan from without a Localizer. The Localizer Protocol abstracts the pose source so the
planner is backend-agnostic:

  - GroundTruthLocalizer — reads a fenced debug/eval side channel (day-1, this file).
  - CuVslamLocalizer     — RGBD visual localization against the map (day-2, same seam).

Both return the same `RobotPose`, so day-2 swaps in without touching the planner, and the
GT stream doubles as the estimated-vs-truth error baseline. Pure: runs in the `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Observation
from humanoid.logic.oli.reason.localization import (
    DebugPoseLocalizer,
    GroundTruthLocalizer,
    Localizer,
    RobotPose,
)

pytestmark = pytest.mark.brain


def _obs(stamp_ns: int = 1) -> Observation:
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


# ── RobotPose contract ───────────────────────────────────────────────────────


def test_robotpose_holds_and_coerces_planar_pose():
    p = RobotPose(stamp_ns=5, x=1, y=2, yaw=0.5)  # ints coerce to float
    assert (p.x, p.y, p.yaw) == (1.0, 2.0, 0.5)
    assert isinstance(p.x, float) and p.stamp_ns == 5


def test_robotpose_defaults_to_origin():
    p = RobotPose(stamp_ns=0)
    assert (p.x, p.y, p.yaw) == (0.0, 0.0, 0.0)


def test_robotpose_is_frozen():
    p = RobotPose(stamp_ns=1, x=0.0, y=0.0, yaw=0.0)
    with pytest.raises(Exception):
        p.x = 3.0  # type: ignore[misc]


# ── GroundTruthLocalizer ─────────────────────────────────────────────────────


def test_gt_localizer_returns_latest_pose_from_reader():
    pose = RobotPose(stamp_ns=10, x=1.0, y=2.0, yaw=0.3)
    loc = GroundTruthLocalizer(pose_reader=lambda: pose)
    assert loc.estimate(_obs()) == pose


def test_gt_localizer_returns_none_before_first_pose():
    loc = GroundTruthLocalizer(pose_reader=lambda: None)
    assert loc.estimate(_obs()) is None


def test_gt_localizer_reads_debug_channel_not_the_pose_less_observation():
    """The GT source is the injected debug reader — never the (pose-less) Observation."""
    seen = []

    def reader():
        seen.append(1)
        return RobotPose(stamp_ns=1, x=7.0, y=0.0, yaw=0.0)

    loc = GroundTruthLocalizer(pose_reader=reader)
    out = loc.estimate(_obs(stamp_ns=999))
    assert seen and out.x == 7.0  # reader consulted regardless of Observation content


def test_gt_localizer_satisfies_localizer_protocol():
    loc = GroundTruthLocalizer(pose_reader=lambda: None)
    assert isinstance(loc, Localizer)  # runtime_checkable Protocol — day-2 backend swaps here


# ── DebugPoseLocalizer (debug-mode realization: reads the debug pose stream) ──


class _FakeClient:
    """Stands in for comm.DebugPoseClient — yields the newest (stamp_ns, x, y, yaw) or None."""

    def __init__(self, sample):
        self._sample = sample

    def latest(self):
        return self._sample


def test_debug_pose_localizer_maps_sample_to_robotpose():
    loc = DebugPoseLocalizer(_FakeClient((7, 1.0, 2.0, 0.3)))
    p = loc.estimate(_obs())
    assert isinstance(p, RobotPose)
    assert (p.stamp_ns, p.x, p.y, p.yaw) == (7, 1.0, 2.0, 0.3)


def test_debug_pose_localizer_none_before_first_sample():
    loc = DebugPoseLocalizer(_FakeClient(None))
    assert loc.estimate(_obs()) is None


def test_debug_pose_localizer_satisfies_localizer_protocol():
    assert isinstance(DebugPoseLocalizer(_FakeClient(None)), Localizer)
