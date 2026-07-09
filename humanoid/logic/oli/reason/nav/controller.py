"""nav/controller.py — pure-pursuit path follower → body-frame velocity command.

The local controller: given the robot's current pose and a planned path (world waypoints), pick
a lookahead point and produce a body-frame twist `(v_x, v_y, w_z)` that carries the robot along
the path. The glide base is holonomic (it accepts lateral `v_y`), so we drive straight toward
the lookahead point in the body frame AND turn to face the travel direction — natural-looking
walking that still strafes through turns. Speed eases to zero inside `goal_tol`.

This is the exact `(v_x, v_y, w_z)` the existing glide path consumes (`Intent` → `GlideAction`
→ `GLIDE_CMD`), so the planner plugs into locomotion with no new contract. Pure: stdlib only.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .types import RobotPose

Point = Tuple[float, float]


def _wrap(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PurePursuit:
    """Holonomic pure-pursuit follower with heading alignment.

    `max_lin` caps linear speed [m/s], `max_wz` caps yaw rate [rad/s], `lookahead` [m] is how far
    ahead on the path to aim, `goal_tol` [m] is the arrival radius (inside it → stop), `k_yaw`
    scales the turn-to-face-travel term. Stateless: `command` is a pure function of (pose, path).
    """

    def __init__(
        self,
        max_lin: float = 1.0,
        max_wz: float = 1.5,
        lookahead: float = 0.5,
        goal_tol: float = 0.15,
        k_yaw: float = 1.5,
    ) -> None:
        self.max_lin = float(max_lin)
        self.max_wz = float(max_wz)
        self.lookahead = float(lookahead)
        self.goal_tol = float(goal_tol)
        self.k_yaw = float(k_yaw)

    def command(self, pose: RobotPose, path: List[Point]) -> Tuple[float, float, float]:
        if not path:
            return (0.0, 0.0, 0.0)

        gx, gy = path[-1]
        dist_goal = math.hypot(gx - pose.x, gy - pose.y)
        if dist_goal <= self.goal_tol:
            return (0.0, 0.0, 0.0)  # arrived — stop

        # Lookahead point: first waypoint at least `lookahead` away, else the final goal.
        look: Point = path[-1]
        for px, py in path:
            if math.hypot(px - pose.x, py - pose.y) >= self.lookahead:
                look = (px, py)
                break

        dx, dy = look[0] - pose.x, look[1] - pose.y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return (0.0, 0.0, 0.0)
        ux, uy = dx / d, dy / d  # unit direction to lookahead (world)

        speed = min(self.max_lin, dist_goal)  # ease down as we near the goal
        cos_y, sin_y = math.cos(pose.yaw), math.sin(pose.yaw)
        # rotate the world-frame direction into the body frame (R(-yaw))
        v_x = speed * (ux * cos_y + uy * sin_y)
        v_y = speed * (-ux * sin_y + uy * cos_y)

        yaw_err = _wrap(math.atan2(dy, dx) - pose.yaw)
        w_z = max(-self.max_wz, min(self.max_wz, self.k_yaw * yaw_err))
        return (v_x, v_y, w_z)
