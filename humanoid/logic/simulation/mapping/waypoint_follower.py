"""waypoint_follower.py — pure coverage-drive motion source (MAY-173 locdev T2).

GT pose (x, y, yaw) → body-frame twist (vx, vy, wz) chasing a waypoint list:
turn-toward + creep-while-turning (vx scales with cos of the heading error), so
the head cameras sweep the scene the way a driven robot would. This is the
SWAPPABLE motion source of the data-collection flow — the recorder is blind to
it, which is what lets the walk-motion mimic replace/augment it later.

Pure numpy/stdlib. Consumed by `coverage_drive_main.py` (isaac env).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointFollower:
    """Chase waypoints on a ground-truth SE(2) pose; done after the last one
    (or wrap when `loop=True`)."""

    def __init__(
        self,
        waypoints: Sequence[Tuple[float, float]],
        *,
        speed: float = 0.8,
        yaw_gain: float = 1.5,
        arrive_radius: float = 0.5,
        max_wz: float = 1.0,
        loop: bool = False,
    ) -> None:
        if not waypoints:
            raise ValueError("waypoints must be non-empty")
        self._wps = [(float(wx), float(wy)) for wx, wy in waypoints]
        self._speed = float(speed)
        self._yaw_gain = float(yaw_gain)
        self._arrive_radius = float(arrive_radius)
        self._max_wz = float(max_wz)
        self._loop = bool(loop)
        self._i = 0
        self._done = False

    @property
    def index(self) -> int:
        return self._i

    @property
    def done(self) -> bool:
        return self._done

    def command(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        """Body-frame twist (vx, vy, wz) for the current pose; (0, 0, 0) when done."""
        if self._done:
            return (0.0, 0.0, 0.0)

        # Advance past every waypoint already within reach (cap at one lap so a
        # degenerate all-close list can't spin forever when looping).
        for _ in range(len(self._wps)):
            tx, ty = self._wps[self._i]
            if math.hypot(tx - x, ty - y) > self._arrive_radius:
                break
            self._i += 1
            if self._i >= len(self._wps):
                if self._loop:
                    self._i = 0
                else:
                    self._done = True
                    return (0.0, 0.0, 0.0)

        tx, ty = self._wps[self._i]
        err = _wrap_pi(math.atan2(ty - y, tx - x) - yaw)
        wz = max(-self._max_wz, min(self._max_wz, self._yaw_gain * err))
        vx = self._speed * max(0.0, math.cos(err))  # creep while turning
        return (vx, 0.0, wz)
