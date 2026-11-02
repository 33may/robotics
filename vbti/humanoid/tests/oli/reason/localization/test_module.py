"""`LocalizationModule` protocol + conformance helper (design.md D5, D13).

The protocol is the host-agnostic surface a candidate implements (bench replays a bag into it;
the live node feeds it from sockets). `verify_module_contract` is the reusable checker any
implementation must pass — the bench and candidate test suites import it. Pure (`brain` env).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationModule,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.testing import verify_module_contract

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _loc_in(stamp_ns):
    frame = CameraFrame(
        stamp_ns=stamp_ns, name="head",
        rgb=np.zeros((3, 4, 3), dtype=np.uint8), depth=np.ones((3, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=2.0, cy=1.5),
    )
    return LocalizationIn(stamp_ns=stamp_ns, frames={"head": frame}, observation=_obs(stamp_ns))


class _HoldStillModule:
    """Minimal conforming module: answers every frame with a TRACKING pose at the hint."""

    def __init__(self) -> None:
        self._pose_xy = (0.0, 0.0)

    def start(self, setup: LocalizationSetup) -> None:
        if setup.initial_pose is not None:
            self._pose_xy = (setup.initial_pose.x, setup.initial_pose.y)

    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        x, y = self._pose_xy
        return LocalizationOut(
            stamp_ns=loc_in.stamp_ns,
            pose=RobotPose(stamp_ns=loc_in.stamp_ns, x=x, y=y),
            status=LocalizationStatus.TRACKING,
            last_fix_stamp_ns=loc_in.stamp_ns,
        )

    def stop(self) -> None:
        pass


class _WrongStampModule(_HoldStillModule):
    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        out = super().step(loc_in)
        return LocalizationOut(
            stamp_ns=out.stamp_ns + 999, pose=out.pose, status=out.status,
            last_fix_stamp_ns=out.last_fix_stamp_ns,
        )


def test_protocol_is_runtime_checkable():
    assert isinstance(_HoldStillModule(), LocalizationModule)
    assert not isinstance(object(), LocalizationModule)


def test_conforming_module_passes_the_checker(tmp_path: Path):
    outs = verify_module_contract(
        _HoldStillModule(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=2.0, y=3.0)),
        [_loc_in(10), _loc_in(20), _loc_in(30)],
    )
    assert len(outs) == 3
    assert all(o.pose is not None and o.pose.x == 2.0 for o in outs)


def test_checker_rejects_stamp_mismatch(tmp_path: Path):
    with pytest.raises(AssertionError, match="stamp"):
        verify_module_contract(
            _WrongStampModule(), LocalizationSetup(map_dir=tmp_path), [_loc_in(10)],
        )


def test_checker_rejects_non_monotonic_input_use(tmp_path: Path):
    # the checker itself guards against a bad HARNESS too: inputs must be monotonic
    with pytest.raises(AssertionError, match="monotonic"):
        verify_module_contract(
            _HoldStillModule(), LocalizationSetup(map_dir=tmp_path),
            [_loc_in(20), _loc_in(10)],
        )
