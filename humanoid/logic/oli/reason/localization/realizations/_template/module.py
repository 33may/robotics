"""<name> — `LocalizationModule` adapter.

TEMPLATE: `/loc-new` copies this folder to start a candidate. The stub below is a CONFORMING
no-op — it answers every frame with the warm-start pose (TRACKING), or LOST when no hint was
given. A fresh candidate is therefore green on the contract and red only on accuracy: replace
the body of `step()` (and `start()`/`stop()` around your vendored algorithm) and keep every
tunable in `config.yaml`, never hardcoded here. Contract + invariants: `../../AGENTS.md`;
loop discipline: `../AGENTS.md`.
"""

from __future__ import annotations

from typing import Optional

from ...contracts import (
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)


class TemplateModule:
    """Conforming no-op: holds the warm-start pose forever; honest LOST without one.

    `stop()` is safe before/after a failed `start()` (teardown contract in `testing.py`).
    """

    def __init__(self) -> None:
        self._hint: Optional[RobotPose] = None

    def start(self, setup: LocalizationSetup) -> None:
        # Real candidates: load setup.map_dir artifacts, boot the algorithm, seed it with
        # setup.initial_pose (v1 is known-start only), read setup.calibration.
        self._hint = setup.initial_pose

    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        if self._hint is None:
            return LocalizationOut(
                stamp_ns=loc_in.stamp_ns, pose=None, status=LocalizationStatus.LOST,
            )
        return LocalizationOut(
            stamp_ns=loc_in.stamp_ns,
            pose=RobotPose(
                stamp_ns=loc_in.stamp_ns, x=self._hint.x, y=self._hint.y, yaw=self._hint.yaw,
            ),
            status=LocalizationStatus.TRACKING,
            last_fix_stamp_ns=loc_in.stamp_ns,
        )

    def stop(self) -> None:
        self._hint = None


def build(config: dict) -> TemplateModule:
    """Registry entrypoint: `config` = parsed config.yaml + caller overrides. The no-op
    template has no tunables; real candidates route EVERY knob through here."""
    return TemplateModule()
