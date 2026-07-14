"""Conformance gate for THIS candidate — runs inside its own `bench-cuvslam` env:

    conda run -n bench-cuvslam python -m pytest logic/oli/reason/localization/realizations/cuvslam/

Deliberately NOT part of the repo's brain-marked suite (playbook §Conformance): dep-heavy
candidates must never be importable from the brain env. Green here = phase-4 "refine" done on
the contract; accuracy is locbench's verdict, never this file's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli.contracts import CameraFrame, CameraIntrinsics, Observation
from humanoid.logic.oli.reason.localization import (
    LocalizationIn,
    LocalizationModule,
    LocalizationSetup,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.testing import verify_module_contract

# /loc-new rewrites `_template` and the class name below. ABSOLUTE import on purpose: a
# relative `.module` makes standalone pytest runs import the package under a second namespace
# (`logic.…` vs `humanoid.logic.…`) and isinstance checks then fail on identity.
from humanoid.logic.oli.reason.localization.realizations.cuvslam.module import (
    CuvslamModule,
)


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


def test_module_satisfies_the_protocol():
    assert isinstance(CuvslamModule(), LocalizationModule)


def test_module_passes_the_conformance_checker(tmp_path: Path):
    outs = verify_module_contract(
        CuvslamModule(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=0.0, y=0.0)),
        [_loc_in(10), _loc_in(20), _loc_in(30)],
    )
    assert len(outs) == 3


def test_stop_tolerates_failed_or_absent_start():
    CuvslamModule().stop()
