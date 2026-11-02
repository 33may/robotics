"""reference — GT replay with injectable degradation (locbench D13, the harness's proof).

Not a localizer: a MEASURING STICK for the bench itself. It replays ground truth through the
full hosting path (in-brain host → telemetry → evaluator → scorer) and degrades it on demand
(config.yaml / bench overrides), so the harness can prove its own gates before any SLAM
exists — the acceptance triplet: clean → PASS; 0.2 m bias → fails max-pos; 20% dropout →
fails coverage.

GT arrives on a BENCH-ONLY feed socket: the evaluator republishes the GT stream to
`Setup.calibration["gt_feed_socket"]` (the World's own debug-pose socket has exactly one
reader — the brain's Nav). Real candidates ignore that key; this module requires it.
Pure stdlib — runs in the plain `brain` env (proves the contract needs nothing special).
"""

from __future__ import annotations

import random
from typing import Optional

from .....comm.debug_pose import DebugPoseClient
from ...contracts import (
    LocalizationIn,
    LocalizationOut,
    LocalizationSetup,
    LocalizationStatus,
    RobotPose,
)


class ReferenceModule:
    """Echo the newest GT sample, degraded per `config["inject"]`:

    | key           | effect                                        |
    |---------------|-----------------------------------------------|
    | `bias_x_m` / `bias_y_m` / `bias_yaw_rad` | constant offset (the E1 regime) |
    | `noise_sigma_m`                          | Gaussian position noise, seeded |
    | `dropout`                                | P(answer LOST), seeded          |
    """

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        inject = dict(self.config.get("inject") or {})
        self._bias_x = float(inject.get("bias_x_m", 0.0))
        self._bias_y = float(inject.get("bias_y_m", 0.0))
        self._bias_yaw = float(inject.get("bias_yaw_rad", 0.0))
        self._sigma = float(inject.get("noise_sigma_m", 0.0))
        self._dropout = float(inject.get("dropout", 0.0))
        self._rng = random.Random(int(inject.get("seed", 0)))
        self._client: Optional[DebugPoseClient] = None
        self._last_fix_ns: Optional[int] = None

    def start(self, setup: LocalizationSetup) -> None:
        path = setup.calibration.get("gt_feed_socket")
        if not path:
            raise ValueError(
                "reference candidate needs Setup.calibration['gt_feed_socket'] — the "
                "locbench evaluator provides it (this candidate is bench-only)")
        self._client = DebugPoseClient(str(path))
        self._last_fix_ns = None

    def step(self, loc_in: LocalizationIn) -> LocalizationOut:
        assert self._client is not None, "step() before start()"
        sample = self._client.latest()
        if sample is None or self._rng.random() < self._dropout:
            return LocalizationOut(stamp_ns=loc_in.stamp_ns, pose=None,
                                   status=LocalizationStatus.LOST)
        _, x, y, yaw = sample
        if self._sigma:
            x += self._rng.gauss(0.0, self._sigma)
            y += self._rng.gauss(0.0, self._sigma)
        # last_fix must be ≤ the answered stamp and monotonic (contract): clamp to input.
        self._last_fix_ns = min(loc_in.stamp_ns, max(self._last_fix_ns or 0, sample[0]))
        return LocalizationOut(
            stamp_ns=loc_in.stamp_ns,
            pose=RobotPose(loc_in.stamp_ns, x + self._bias_x, y + self._bias_y,
                           yaw + self._bias_yaw),
            status=LocalizationStatus.TRACKING,
            last_fix_stamp_ns=self._last_fix_ns,
        )

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build(config: dict) -> ReferenceModule:
    return ReferenceModule(config)
