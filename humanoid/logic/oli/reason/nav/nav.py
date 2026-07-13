"""nav/nav.py — the autonomous-navigation Reason module (goal → localize → plan → follow → intent).

`Nav` is the single place path planning lives: it consumes a `GoalCoordinate` (`set_goal`, e.g. a
dev_app click), owns its costmap (inflated footprint + soft clearance), and produces the planned
path OUT (`plan(pose)` / `.path`) for observers to render. As a Reason module it is also a drop-in
for `Teleop` in the Orchestrator — the same `to_policy_in(observation[, camera_frame])` seam runs
localize → plan → pure-pursuit to a body-frame `(v_x, v_y, w_z)` `Intent`, so everything downstream
(glide → `GLIDE_CMD` → World) is unchanged. With no goal / no pose / no path it emits a
**zero-velocity hold** — never a joint command (that is PolicyRunner's job).

Two clean seams: **goal in** (`GoalCoordinate` → `set_goal`) and **path out** (`plan` → `.path`).
The inner state (costmap, clearance layer, localizer, planned path) is private — the "black-box
module, clean contract" shape. Pure: no isaacsim/limxsdk.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ...contracts import CameraFrame, Intent, Mode, Observation, PolicyIn
from ..localization import Localizer
from ..mapping import OccupancyGrid
from .controller import PurePursuit
from .planner import plan_path
from .types import GoalCoordinate

Point = Tuple[float, float]


class Nav:
    """Compose a `Localizer` + owned costmap + planner + pure-pursuit into a navigation module.

    Pass the RAW `costmap`; Nav builds its own robot layer internally: a hard `inflate` by
    `robot_radius_m` (impassable footprint) plus a soft `clearance_cost(inflation_radius_m,
    clearance_weight)` gradient so paths prefer the open middle but still take a tight gap when
    forced. These knobs are robot/policy — kept here, not in the map bake. Defaults (0.0) are a
    no-op point planner. `mode` is the `Intent.mode` stamped on emitted commands (glide ignores it;
    the real walk policy would honor `WALK`). Re-plans on each `plan`/`to_policy_in` call.
    """

    def __init__(
        self,
        costmap: OccupancyGrid,
        localizer: Localizer,
        mode: Mode = Mode.WALK,
        controller: Optional[PurePursuit] = None,
        *,
        robot_radius_m: float = 0.0,
        inflation_radius_m: float = 0.0,
        clearance_weight: float = 0.0,
        heuristic_weight: float = 1.0,
        horizon_m: float = 2.0,
    ) -> None:
        self._costmap = costmap                              # raw map: coord transforms + clearance src
        self._plan_grid = costmap.inflate(robot_radius_m)    # hard footprint (impassable band)
        self._cost = costmap.clearance_cost(inflation_radius_m, clearance_weight)  # soft gradient
        self._localizer = localizer
        self._mode = mode
        self._controller = controller or PurePursuit()
        # heuristic_weight>1 = weighted A* (fewer nodes, path ≤ weight×optimal); horizon_m bounds
        # each local re-plan so compute is flat regardless of how far the goal is.
        self._weight = heuristic_weight
        self._horizon_m = horizon_m
        self._goal: Optional[GoalCoordinate] = None
        self._path: Optional[List[Point]] = None

    # ── goal IN ──────────────────────────────────────────────────────────────
    def set_goal(self, goal: GoalCoordinate) -> None:
        """Set the navigation goal (map-frame `GoalCoordinate`, e.g. from a dev_app click).

        Clears the cached path so the next `plan` is a FULL re-plan from the robot to this new
        goal; subsequent same-goal `plan`s do cheap LOCAL re-plans (near-horizon only)."""
        self._goal = goal
        self._path = None

    def clear_goal(self) -> None:
        self._goal = None
        self._path = None

    # ── path OUT ─────────────────────────────────────────────────────────────
    @property
    def path(self) -> Optional[List[Point]]:
        """The most recently planned path (world waypoints), or None — for observers to render."""
        return self._path

    def _full_plan(self, pose) -> Optional[List[Point]]:
        return plan_path(
            self._plan_grid, (pose.x, pose.y), (self._goal.x, self._goal.y),
            cost=self._cost, weight=self._weight,
        )

    def plan(self, pose) -> Optional[List[Point]]:
        """Plan from `pose` to the current goal on the owned costmap; store + return the path.

        First call after a new goal = a FULL plan (robot→goal). While the goal holds, each call is
        a cheap LOCAL re-plan: re-solve only the near `horizon_m` from the robot and splice the
        untouched far tail — so compute is flat no matter how distant the goal is. Falls back to a
        full plan if the local solve fails (robot drifted far off the path). None when there is no
        goal or it is blocked/unreachable. The "path out" seam — dev_app renders `.path`."""
        if self._goal is None:
            self._path = None
            return None
        if not self._path:                        # new/cleared goal → full plan
            self._path = self._full_plan(pose)
            return self._path
        horizon, tail = self._split_ahead(pose, self._path, self._horizon_m)
        local = plan_path(
            self._plan_grid, (pose.x, pose.y), horizon, cost=self._cost, weight=self._weight
        )
        self._path = (local + tail) if local else self._full_plan(pose)
        return self._path

    @staticmethod
    def _split_ahead(pose, path: List[Point], horizon_m: float) -> Tuple[Point, List[Point]]:
        """Walk `path` from the waypoint nearest the robot until `horizon_m` accumulated; return
        (that horizon waypoint, the remaining tail after it). The local re-plan targets the horizon;
        the tail is spliced on unchanged."""
        i0 = min(range(len(path)), key=lambda i: (path[i][0] - pose.x) ** 2 + (path[i][1] - pose.y) ** 2)
        acc, ih = 0.0, i0
        for i in range(i0, len(path) - 1):
            acc += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            ih = i + 1
            if acc >= horizon_m:
                break
        return path[ih], path[ih + 1:]

    # ── Reason seam (localize → plan → follow → intent) ──────────────────────
    def to_policy_in(
        self, observation: Observation, camera_frame: Optional[CameraFrame] = None
    ) -> PolicyIn:
        if self._goal is None:
            return self._hold(observation)
        pose = self._localizer.estimate(observation, camera_frame)
        if pose is None:
            return self._hold(observation)  # no localization yet → stay put
        path = self.plan(pose)
        if not path:
            return self._hold(observation)  # goal blocked / unreachable → stay put
        v_x, v_y, w_z = self._controller.command(pose, path)
        return PolicyIn(
            observation=observation,
            intent=Intent(mode=self._mode, v_x=v_x, v_y=v_y, w_z=w_z),
        )

    def _hold(self, observation: Observation) -> PolicyIn:
        return PolicyIn(
            observation=observation,
            intent=Intent(mode=self._mode, v_x=0.0, v_y=0.0, w_z=0.0),
        )
