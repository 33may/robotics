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

from ..localization import RobotPose

Point = Tuple[float, float]


def _wrap(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PurePursuit:
    """Differential-drive pure-pursuit follower — faces where it goes; never strafes or reverses.

    Only a **front cone** of ±`front_cone` (default 60°, so a 120° arc) is drivable: within it the
    robot moves forward AND turns toward the target, with forward speed easing to zero as the
    heading error approaches the cone edge; outside the cone (target to the side or behind) it
    **rotates in place** until the target enters the cone, then moves. `v_y` is always 0 (no lateral
    strafe) and `v_x` is never negative (no reverse) — natural humanoid locomotion.

    `max_lin` caps linear speed [m/s], `max_wz` caps yaw rate [rad/s], `lookahead` [m] is how far
    ahead on the path to aim, `goal_tol` [m] is the arrival radius (inside it → stop), `k_yaw`
    scales the turn term, `front_cone` [rad] is the half-arc within which motion is allowed.
    Stateless: `command` is a pure function of (pose, path).
    """

    def __init__(
        self,
        max_lin: float = 1.0,
        max_wz: float = 1.5,
        lookahead: float = 0.5,
        goal_tol: float = 0.15,
        k_yaw: float = 1.5,
        front_cone: float = math.pi / 3.0,   # 60° half-arc → 120° drivable front
    ) -> None:
        self.max_lin = float(max_lin)
        self.max_wz = float(max_wz)
        self.lookahead = float(lookahead)
        self.goal_tol = float(goal_tol)
        self.k_yaw = float(k_yaw)
        self.front_cone = float(front_cone)

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
        if math.hypot(dx, dy) < 1e-9:
            return (0.0, 0.0, 0.0)

        yaw_err = _wrap(math.atan2(dy, dx) - pose.yaw)   # heading error to the lookahead
        w_z = max(-self.max_wz, min(self.max_wz, self.k_yaw * yaw_err))
        if abs(yaw_err) > self.front_cone:
            return (0.0, 0.0, w_z)   # target outside the front cone → rotate in place, no motion

        # Inside the cone: drive forward (never strafe/reverse), easing as the heading error grows
        # so speed is full when aligned and tapers to 0 at the cone edge.
        speed = min(self.max_lin, dist_goal)
        v_x = speed * (1.0 - abs(yaw_err) / self.front_cone)
        return (v_x, 0.0, w_z)
