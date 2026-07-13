"""nav/types.py — the navigation module's own input contract.

Brain-internal nav types, distinct from the world-edge contracts in `oli/contracts.py`.
`RobotPose` moved to `reason/localization/contracts.py` (it is localization's OUTPUT type;
dependencies point nav→localization — change `may-173-reason-module-separation`).
Pure: stdlib only, never isaacsim/limxsdk.
"""

from __future__ import annotations

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
