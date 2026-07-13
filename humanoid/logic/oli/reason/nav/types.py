"""nav/types.py — the navigation reasoning stack's internal contracts.

These are *brain-internal* nav types (localizer→planner seam), distinct from the world-edge
contracts in `oli/contracts.py`. Pure: numpy/stdlib only, never isaacsim/limxsdk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional


class GoalCoordinate(NamedTuple):
    """A map-frame navigation goal — the clean contract INTO the nav layer (e.g. a dev_app click).

    Planar (x, y) with an optional final heading `yaw` (None = don't care, the 2D PoC default).
    A `NamedTuple` so it reads as a contract (`goal.x`) yet still unpacks/subscripts like a point
    (`goal[0]`, `x, y, *_ = goal`) for the renderer and planner. Named `GoalCoordinate` (not
    `Goal`) to leave room for higher-level reasoning/task "goals" later in the system.
    """

    x: float
    y: float
    yaw: Optional[float] = None


@dataclass(frozen=True)
class RobotPose:
    """Planar robot pose in the map frame (SE(2)) — the Localizer's output, the planner's input.

    `map frame` = the fixed ground-truth scene frame the occupancy grid is baked in (the scene
    loaded by `--scene`). `stamp_ns` is SIM-time (the same D8 clock as `Observation`), so a pose
    aligns with the observation/camera frame it was derived from. Planar (x, y, yaw) for the 2D
    nav PoC; a future SE(3) pose extends this **without changing the `Localizer` seam** (the
    planner consumes whatever pose type the localizer emits).
    """

    stamp_ns: int       # SIM-time nanoseconds — the D8 pacing clock
    x: float = 0.0      # map-frame X [m]
    y: float = 0.0      # map-frame Y [m]
    yaw: float = 0.0    # map-frame heading [rad], CCW about +Z

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "yaw", float(self.yaw))
