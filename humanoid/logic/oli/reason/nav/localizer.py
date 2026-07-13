"""nav/localizer.py — the pose-source seam and its day-1 ground-truth backend.

`Observation` deliberately carries no world pose (`contracts.py`: "the real robot has no
ground-truth pose"), so the planner cannot plan without a pose estimate. `Localizer` is the
seam that produces one; backends are swappable behind it and the planner never knows which is
live:

  - `GroundTruthLocalizer` — reads a fenced debug/eval side channel (day-1, here).
  - `CuVslamLocalizer`     — RGBD visual localization against the map (day-2, same seam).

Pure: no isaacsim/limxsdk (the `brain` invariant holds).
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from ...contracts import CameraFrame, Observation
from .types import RobotPose


@runtime_checkable
class Localizer(Protocol):
    """(observation[, camera_frame]) → current best `RobotPose`, or None until one exists.

    The single seam the planner depends on. Keeping it uniform (both invariant brain inputs in
    the signature) lets a day-2 visual localizer — which *does* consume `camera_frame` + FK from
    `observation` — drop in where the day-1 ground-truth backend sits, with zero planner change.
    """

    def estimate(
        self, observation: Observation, camera_frame: Optional[CameraFrame] = None
    ) -> Optional[RobotPose]: ...


class GroundTruthLocalizer:
    """Debug/eval `Localizer`: surfaces the World's ground-truth pose from a side channel.

    Fenced OUTSIDE the invariance boundary on purpose: the GT pose never rides the `Observation`
    contract (the real robot has no such signal) — it arrives on a separate debug channel that
    production simply never launches. `pose_reader` yields the latest `RobotPose` (or None) and
    is injected, so this class stays pure and unit-testable — tests pass a fake callable, the
    runtime passes the debug-socket client's reader. It ignores `observation`/`camera_frame` by
    design; those are the *real* localizer's inputs. Because it satisfies `Localizer`, the day-2
    visual backend drops into the same planner seam, and this same GT stream becomes the
    estimated-vs-truth error baseline for evaluating that backend.
    """

    def __init__(self, pose_reader: Callable[[], Optional[RobotPose]]) -> None:
        self._pose_reader = pose_reader

    def estimate(
        self, observation: Observation, camera_frame: Optional[CameraFrame] = None
    ) -> Optional[RobotPose]:
        return self._pose_reader()


class DebugPoseLocalizer:
    """`Localizer` backed by the fenced debug ground-truth pose stream (world → brain).

    The DEBUG-mode realization of the pose-source abstraction. Wraps a client exposing
    ``latest() -> (stamp_ns, x, y, yaw) | None`` (the comm `DebugPoseClient`) and turns the newest
    sample into a `RobotPose`. Today it carries Isaac's ground truth, so the planner can be verified
    on perfect coordinates; once verified, a localization-module `Localizer` (cuVSLAM / RTAB-Map)
    drops into this exact seam and nothing downstream changes. Ignores `observation`/`camera_frame`
    — its coordinates come from the stream, not proprioception. Duck-types the client (no `comm`
    import), so it stays unit-testable with a fake.
    """

    def __init__(self, client) -> None:
        self._client = client

    def estimate(
        self, observation: Observation, camera_frame: Optional[CameraFrame] = None
    ) -> Optional[RobotPose]:
        sample = self._client.latest()
        if sample is None:
            return None
        stamp_ns, x, y, yaw = sample
        return RobotPose(stamp_ns=stamp_ns, x=x, y=y, yaw=yaw)
