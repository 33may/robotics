"""nav/nav.py — the autonomous-navigation Reason module (localize → plan → follow → intent).

`Nav` is a drop-in for `Teleop` in the Orchestrator: it produces `PolicyIn` from the same
`to_policy_in(observation[, camera_frame])` seam, so everything downstream (glide → `GLIDE_CMD`
→ World) is unchanged — the only difference is the velocity comes from a planner instead of a
joystick. Each call it asks the `Localizer` where the robot is, plans a path to the current goal
on the (inflated) costmap, and runs pure-pursuit to a body-frame `(v_x, v_y, w_z)` wrapped as an
`Intent`. With no goal / no pose / no path it emits a **zero-velocity hold** — never a joint
command (that is PolicyRunner's job).

The inner state (costmap, localizer backend, planned path) is private; callers only set a goal
and read `PolicyIn` — the "black-box module, clean contract" shape. Pure: no isaacsim/limxsdk.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ...contracts import CameraFrame, Intent, Mode, Observation, PolicyIn
from .controller import PurePursuit
from .costmap import OccupancyGrid
from .localizer import Localizer
from .planner import plan_path


class Nav:
    """Compose a `Localizer` + costmap + planner + pure-pursuit into a navigation Reason module.

    `costmap` should already be inflated by the robot radius (`OccupancyGrid.inflate`), so the
    planner can treat the robot as a point. `mode` is the `Intent.mode` stamped on emitted
    commands (glide ignores it; the real walk policy would honor `WALK`). Re-plans every call —
    fine for a static PoC-sized grid and it also picks up live costmap changes for free; a replan
    throttle is a later optimization.
    """

    def __init__(
        self,
        costmap: OccupancyGrid,
        localizer: Localizer,
        mode: Mode = Mode.WALK,
        controller: Optional[PurePursuit] = None,
    ) -> None:
        self._costmap = costmap
        self._localizer = localizer
        self._mode = mode
        self._controller = controller or PurePursuit()
        self._goal: Optional[Tuple[float, float]] = None

    def set_goal(self, x: float, y: float) -> None:
        """Set the navigation goal in map-frame world coords (e.g. from a dev_app click)."""
        self._goal = (float(x), float(y))

    def clear_goal(self) -> None:
        self._goal = None

    def to_policy_in(
        self, observation: Observation, camera_frame: Optional[CameraFrame] = None
    ) -> PolicyIn:
        if self._goal is None:
            return self._hold(observation)
        pose = self._localizer.estimate(observation, camera_frame)
        if pose is None:
            return self._hold(observation)  # no localization yet → stay put
        path = plan_path(self._costmap, (pose.x, pose.y), self._goal)
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
