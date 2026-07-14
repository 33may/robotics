"""The `_template` realization — the `/loc-new` scaffold source (may-173-locdev-flow §3.1).

The template must be a CONFORMING no-op so every scaffolded candidate starts life green on the
contract and red only on accuracy — bring-up starts from "runs at all", not from a broken shell.

This suite deliberately imports a realization from the brain env: `_template` (like
`reference/`) is one of the PURE realizations (stdlib/numpy only) that the playbook's
conformance-split allows in the brain suite. Dep-heavy candidates are the ones that must never
be imported here — the architecture guard (locbench change, task 11.3) should scope its
no-realization-imports rule to `logic/oli/` sources, not to this file. The protocol
canary/isinstance checks for the template live HERE (not in test_architecture.py) to keep this
change's footprint disjoint from the in-flight locbench build.
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
    LocalizationStatus,
    RobotPose,
)
from humanoid.logic.oli.reason.localization.realizations._template.module import (
    TemplateModule,
)
from humanoid.logic.oli.reason.localization.testing import verify_module_contract

pytestmark = pytest.mark.brain

# static canary — the declaration-site conformance check (../AGENTS.md, the Protocol pattern)
_static_template: LocalizationModule = TemplateModule()


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


def test_template_satisfies_the_protocol():
    assert isinstance(TemplateModule(), LocalizationModule)


def test_template_passes_the_conformance_checker(tmp_path: Path):
    outs = verify_module_contract(
        TemplateModule(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=1.5, y=-2.0, yaw=0.3)),
        [_loc_in(10), _loc_in(20), _loc_in(30)],
    )
    assert len(outs) == 3


def test_template_holds_the_warm_start_pose(tmp_path: Path):
    """The no-op behavior contract: answer every frame with the warm-start hint, TRACKING.

    This makes a freshly scaffolded candidate produce a *scoreable* run immediately (poses at
    the spawn, 100% coverage) — its accuracy failure is then purely algorithmic, which is the
    correct starting red for the loop.
    """
    outs = verify_module_contract(
        TemplateModule(),
        LocalizationSetup(map_dir=tmp_path, initial_pose=RobotPose(stamp_ns=0, x=1.5, y=-2.0, yaw=0.3)),
        [_loc_in(10), _loc_in(20)],
    )
    for out in outs:
        assert out.status is LocalizationStatus.TRACKING
        assert out.pose is not None
        assert (out.pose.x, out.pose.y, out.pose.yaw) == (1.5, -2.0, 0.3)


def test_template_without_hint_reports_lost(tmp_path: Path):
    """No warm start (no `initial_pose`) ⇒ the template honestly knows nothing: LOST, pose None.

    Guards the pose⇔LOST invariant path a scaffolded candidate inherits — and documents that
    v1 realizations are known-start only (../AGENTS.md: no kidnapped-robot recovery).
    """
    outs = verify_module_contract(
        TemplateModule(), LocalizationSetup(map_dir=tmp_path), [_loc_in(10)],
    )
    assert outs[0].pose is None
    assert outs[0].status is LocalizationStatus.LOST


def test_template_stop_tolerates_failed_start(tmp_path: Path):
    """`stop()` after a failed/partial `start()` must not raise (testing.py teardown contract)."""
    module = TemplateModule()
    module.stop()  # never started — must be a no-op, not an AttributeError
