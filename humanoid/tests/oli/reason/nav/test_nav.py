"""TDD for the Nav reason module (nav/nav.py) — the localize→plan→follow→intent composition.

Verifies it behaves as a drop-in Reason module: emits `PolicyIn` wrapping the given Observation,
drives toward a set goal, and holds zero-velocity when there is no goal / no pose / no path. Pure:
runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation, PolicyIn
from humanoid.logic.oli.reason.localization import GroundTruthLocalizer, RobotPose
from humanoid.logic.oli.reason.mapping import OccupancyGrid, StaticMapping
from humanoid.logic.oli.reason.nav import GoalCoordinate, Planner
from humanoid.logic.oli.reason.nav.nav import Nav

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _empty_map(n=10, res=1.0):
    return StaticMapping.from_grid(OccupancyGrid(np.zeros((n, n), dtype=bool), res))


def _loc_at(x, y, yaw=0.0):
    return GroundTruthLocalizer(pose_reader=lambda: RobotPose(stamp_ns=0, x=x, y=y, yaw=yaw))


def test_no_goal_holds_zero_velocity():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    pin = nav.to_policy_in(_obs())
    assert isinstance(pin, PolicyIn)
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)
    assert pin.intent.mode == Mode.WALK


def test_policy_in_wraps_the_given_observation():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    obs = _obs(stamp_ns=42)
    assert nav.to_policy_in(obs).observation is obs


def test_no_pose_yet_holds():
    nav = Nav(_empty_map(), GroundTruthLocalizer(pose_reader=lambda: None))
    nav.set_goal(GoalCoordinate(8.5, 0.5))
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_drives_toward_goal_ahead():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5, yaw=0.0))
    nav.set_goal(GoalCoordinate(8.5, 0.5))  # straight along +x
    pin = nav.to_policy_in(_obs())
    assert pin.intent.v_x > 0.5
    assert pin.intent.mode == Mode.WALK


def test_stops_when_at_goal():
    nav = Nav(_empty_map(), _loc_at(8.5, 0.5))
    nav.set_goal(GoalCoordinate(8.5, 0.5))
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_unreachable_goal_holds():
    arr = np.zeros((10, 10), dtype=bool)
    arr[0, 8] = True  # goal cell occupied → no path
    nav = Nav(StaticMapping.from_grid(OccupancyGrid(arr, 1.0)), _loc_at(0.5, 0.5))
    nav.set_goal(GoalCoordinate(8.5, 0.5))
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_clear_goal_returns_to_hold():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    nav.set_goal(GoalCoordinate(8.5, 0.5))
    assert nav.to_policy_in(_obs()).intent.v_x > 0.0
    nav.clear_goal()
    assert nav.to_policy_in(_obs()).intent.v_x == 0.0


# ── path OUT of the nav layer (plan() + .path) — consumed by the dev_app renderer ────

def test_plan_returns_path_and_exposes_it():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    nav.set_goal(GoalCoordinate(8.5, 0.5))
    path = nav.plan(RobotPose(stamp_ns=0, x=0.5, y=0.5))
    assert path and len(path) >= 2
    assert path[0] == (0.5, 0.5) and path[-1] == (8.5, 0.5)  # world waypoints
    assert nav.path is path                                  # exposed for the panel to render


def test_plan_without_goal_is_none():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    assert nav.plan(RobotPose(stamp_ns=0, x=0.5, y=0.5)) is None
    assert nav.path is None


def test_nav_owns_clearance_and_prefers_open_route():
    # raw wall along row 0; Nav builds its own inflated + clearance costmap from robot params
    arr = np.zeros((3, 5), dtype=bool)
    arr[0, :] = True
    grid = OccupancyGrid(arr, 1.0)
    nav = Nav(StaticMapping.from_grid(grid), _loc_at(0.5, 1.5),
              planner=Planner(inflation_radius_m=2.0, clearance_weight=5.0))
    nav.set_goal(GoalCoordinate(4.5, 1.5))
    path = nav.plan(RobotPose(stamp_ns=0, x=0.5, y=1.5))
    assert path is not None
    # clearance cost pushes the route off the wall (row 1) up to row 2 where there's room
    assert any(grid.world_to_cell(x, y)[0] == 2 for x, y in path)


# ── local re-plan: full on new goal, then cheap near-horizon splice keeping the tail ─

def test_second_plan_is_local_and_reuses_far_tail():
    nav = Nav(_empty_map(20), _loc_at(0.5, 0.5), planner=Planner(horizon_m=2.0))
    nav.set_goal(GoalCoordinate(18.5, 0.5))
    full = nav.plan(RobotPose(stamp_ns=0, x=0.5, y=0.5))     # FULL plan (new goal)
    assert full and full[-1] == (18.5, 0.5)
    moved = nav.plan(RobotPose(stamp_ns=0, x=2.5, y=0.5))     # LOCAL re-plan from a moved pose
    assert moved[-1] == (18.5, 0.5)                           # still reaches the goal
    assert moved[-3:] == full[-3:]                            # far tail reused verbatim


def test_set_goal_forces_full_replan():
    nav = Nav(_empty_map(20), _loc_at(0.5, 0.5))
    nav.set_goal(GoalCoordinate(18.5, 0.5))
    nav.plan(RobotPose(stamp_ns=0, x=0.5, y=0.5))
    nav.set_goal(GoalCoordinate(0.5, 18.5))                   # new goal clears the cached path
    assert nav.path is None
    p2 = nav.plan(RobotPose(stamp_ns=0, x=0.5, y=0.5))
    assert p2[-1] == (0.5, 18.5)
