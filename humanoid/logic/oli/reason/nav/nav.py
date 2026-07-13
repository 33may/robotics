"""nav/nav.py — the thin navigation orchestrator (goal → localize → map → plan → follow → intent).

Post-split (design.md D10), `Nav` holds ONLY: the goal state (its "goal in" seam), the module
refs, the per-tick chain, the zero-velocity hold semantics, and the `Intent.mode` stamping. All
planning state (robot layer, path cache, full/local re-plan) lives in `Planner`; the world lives
behind `MappingModule.latest()`; the pose behind the `Localizer` seam. Nav pulls the newest map
each tick and hands it to the planner as an explicit value — the orchestrator's job (D8).

As a Reason module it is a drop-in for `Teleop`: the same `to_policy_in(observation[,
camera_frame])` runs the chain to a body-frame `(v_x, v_y, w_z)` `Intent`, so everything
downstream (glide → `GLIDE_CMD` → World) is unchanged. With no goal / no pose / no map / no path
it emits a **zero-velocity hold** — never a joint command (that is PolicyRunner's job).

Two clean seams: **goal in** (`GoalCoordinate` → `set_goal`) and **path out** (`plan` / `.path`)
for observers to render. Pure: no isaacsim/limxsdk.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...contracts import CameraFrame, Intent, Mode, Observation, PolicyIn
from ..localization import Localizer, RobotPose
from ..mapping import MappingModule
from .controller import PurePursuit
from .planner import Planner
from .types import GoalCoordinate

Point = Tuple[float, float]


class Nav:
    """Compose localization + mapping + planning + following behind the Reason seam.

    `mapping` and `localizer` are the module refs Nav pulls from each tick; `planner` carries
    the robot/policy knobs (footprint, clearance, re-plan horizon — default: a no-op point
    planner); `controller` is the follower. `mode` is the `Intent.mode` stamped on emitted
    commands (glide ignores it; the real walk policy would honor `WALK`).
    """

    def __init__(
        self,
        mapping: MappingModule,
        localizer: Localizer,
        mode: Mode = Mode.WALK,
        controller: Optional[PurePursuit] = None,
        planner: Optional[Planner] = None,
    ) -> None:
        self._mapping = mapping
        self._localizer = localizer
        self._planner = planner or Planner()
        self._controller = controller or PurePursuit()
        self._mode = mode
        self._goal: Optional[GoalCoordinate] = None

    # ── goal IN ──────────────────────────────────────────────────────────────
    def set_goal(self, goal: GoalCoordinate) -> None:
        """Set the navigation goal (map-frame `GoalCoordinate`, e.g. from a dev_app click).

        Clears the planner's cached path so the next `plan` is a FULL re-plan; subsequent
        same-goal `plan`s do cheap LOCAL re-plans (near-horizon only)."""
        self._goal = goal
        self._planner.clear()

    def clear_goal(self) -> None:
        self._goal = None
        self._planner.clear()

    # ── path OUT ─────────────────────────────────────────────────────────────
    @property
    def path(self) -> Optional[List[Point]]:
        """The most recently planned path (world waypoints), or None — for observers to render."""
        return self._planner.path

    def plan(self, pose: RobotPose) -> Optional[List[Point]]:
        """Plan from `pose` to the current goal on the newest map snapshot; None when there is
        no goal / no map yet / the goal is blocked. The "path out" seam — dev_app renders it."""
        world_map = self._mapping.latest()
        if world_map is None:
            return None
        return self._planner.plan(pose, self._goal, world_map)

    # ── Reason seam (localize → map → plan → follow → intent) ────────────────
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
            return self._hold(observation)  # no map / goal blocked / unreachable → stay put
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
