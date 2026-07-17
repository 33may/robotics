"""TDD for the stateful `Planner` module (nav/planner.py — design.md D8, D9).

The planner consumes the EMITTED `Map` value (`plan(pose, goal, world_map)`) — it never holds a
mapping-module ref. It owns its derivations privately: the robot layer (inflate + clearance)
cached keyed on `map.version`, and the path cache with goal-change detection (new goal → full A*,
same goal → local horizon re-plan + tail splice). A version bump rebuilds the layer AND drops the
cached path. Pure: runs in `brain`.
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid.logic.oli.reason.localization import RobotPose
from humanoid.logic.oli.reason.mapping import Map, OccupancyGrid
from humanoid.logic.oli.reason.nav import GoalCoordinate
from humanoid.logic.oli.reason.nav.planner import Planner

pytestmark = pytest.mark.brain


class _CountingGrid(OccupancyGrid):
    """OccupancyGrid that counts robot-layer derivations (inflate calls)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.inflate_calls = 0

    def inflate(self, radius_m):
        self.inflate_calls += 1
        return super().inflate(radius_m)


def _grid(n=10, res=1.0):
    return _CountingGrid(np.zeros((n, n), dtype=bool), res)


def _map(grid, version=1):
    return Map(grid=grid, version=version, stamp_ns=version)


def _pose(x=0.5, y=0.5):
    return RobotPose(stamp_ns=0, x=x, y=y)


def test_full_plan_reaches_goal():
    p = Planner()
    path = p.plan(_pose(), GoalCoordinate(8.5, 8.5), _map(_grid()))
    assert path is not None
    assert path[-1] == pytest.approx((8.5, 8.5))
    assert p.path is path


def test_no_goal_clears_path_and_returns_none():
    p = Planner()
    p.plan(_pose(), GoalCoordinate(8.5, 8.5), _map(_grid()))
    assert p.plan(_pose(), None, _map(_grid())) is None
    assert p.path is None


def test_blocked_goal_returns_none():
    g = _grid()
    g.grid[8, 8] = True
    assert Planner().plan(_pose(), GoalCoordinate(8.5, 8.5), _map(g)) is None


def test_same_goal_replans_locally_from_moved_pose(monkeypatch):
    # Pin the MECHANISM, not just the endpoints: the second call must run A* to the near
    # HORIZON waypoint (a local re-plan), never to the final goal (which would mean a silent
    # regression to always-full-planning — endpoint asserts alone cannot tell the difference).
    from humanoid.logic.oli.reason.nav import planner as planner_mod

    calls = []
    real_plan_path = planner_mod.plan_path

    def spy(grid, start, goal, **kw):
        calls.append(goal)
        return real_plan_path(grid, start, goal, **kw)

    monkeypatch.setattr(planner_mod, "plan_path", spy)
    p = Planner(horizon_m=2.0)
    g = _grid()
    m = _map(g)
    goal = GoalCoordinate(8.5, 0.5)
    first = p.plan(_pose(0.5, 0.5), goal, m)
    moved = p.plan(_pose(2.5, 0.5), goal, m)     # same goal → local re-plan + tail splice
    assert first is not None and moved is not None
    assert moved[0] == pytest.approx((2.5, 0.5))  # re-plan starts at the moved pose
    assert moved[-1] == pytest.approx((8.5, 0.5))
    assert calls[0] == (goal.x, goal.y)           # first call: FULL plan to the goal
    assert len(calls) == 2 and calls[1] != (goal.x, goal.y), (
        "second same-goal plan must target the local HORIZON waypoint, not the goal — "
        "the local re-plan machinery regressed to full planning"
    )


def test_clear_preserves_robot_layer():
    # clear() forgets goal+path (next plan is FULL) but must NOT throw away the derived robot
    # layer — it depends on the map, not the goal (planner.py clear() docstring).
    g = _grid()
    p = Planner(robot_radius_m=0.5)
    m = _map(g)
    p.plan(_pose(), GoalCoordinate(8.5, 8.5), m)
    p.clear()
    p.plan(_pose(), GoalCoordinate(0.5, 8.5), m)
    assert g.inflate_calls == 1                   # layer survived the clear


def test_goal_change_forces_full_replan():
    p = Planner()
    m = _map(_grid())
    p.plan(_pose(), GoalCoordinate(8.5, 8.5), m)
    changed = p.plan(_pose(), GoalCoordinate(0.5, 8.5), m)
    assert changed is not None and changed[-1] == pytest.approx((0.5, 8.5))


def test_robot_layer_derived_once_per_version():
    g = _grid()
    p = Planner(robot_radius_m=0.5)
    m = _map(g, version=1)
    goal = GoalCoordinate(8.5, 8.5)
    for x in (0.5, 1.5, 2.5):
        p.plan(_pose(x, 0.5), goal, m)
    assert g.inflate_calls == 1                   # cached on version — derived exactly once


def test_version_bump_rebuilds_layer_and_drops_path():
    g = _grid()
    p = Planner()
    goal = GoalCoordinate(8.5, 8.5)
    assert p.plan(_pose(0.5, 0.5), goal, _map(g, version=1)) is not None
    # The world changed: a wall now fully seals the map (same grid object, bumped version).
    # If the planner failed to rebuild the layer OR kept the stale cached path, it would still
    # return a path — None proves BOTH invalidations happened.
    g.grid[4, :] = True
    assert p.plan(_pose(0.5, 0.5), goal, _map(g, version=2)) is None
    assert g.inflate_calls == 2                   # robot layer re-derived on the bump


# ── escape corridor: start blocked by inflation only (2026-07-17, Anton) ──────
# A baked map can leave the robot's spawn inside an inflation-locked pocket
# (start/end camera blind spot). The planner must escape over RAW-free cells
# (physically observed/swept floor) to the nearest legally-free cell — never
# through a raw obstacle.


def _pocket_world(n=20, res=0.1):
    """Free room with a walled pocket rows/cols 2..8 and a 1-cell door at
    (5, 8). With robot_radius ≥ 1 cell the door is inflation-blocked."""
    g = np.zeros((n, n), dtype=bool)
    g[2, 2:9] = True   # bottom wall
    g[8, 2:9] = True   # top wall
    g[2:9, 2] = True   # left wall
    g[2:9, 8] = True   # right wall
    g[5, 8] = False    # the door
    return _CountingGrid(g, res)


def test_escape_corridor_plans_out_of_inflation_locked_pocket():
    world = _pocket_world()
    p = Planner(robot_radius_m=0.15)
    path = p.plan(_pose(0.55, 0.55), GoalCoordinate(1.55, 1.55), _map(world))
    assert path is not None
    # ends at the goal cell
    assert abs(path[-1][0] - 1.55) < 0.1 and abs(path[-1][1] - 1.55) < 0.1
    # every waypoint is raw-free: escape never tunnels a real wall
    for x, y in path:
        assert not world.is_occupied_cell(*world.world_to_cell(x, y))


def test_escape_returns_none_when_start_sealed_by_raw_walls():
    world = _pocket_world()
    world.grid[5, 8] = True  # brick up the door — pocket truly sealed
    p = Planner(robot_radius_m=0.15)
    path = p.plan(_pose(0.55, 0.55), GoalCoordinate(1.55, 1.55), _map(world))
    assert path is None


def test_unblocked_start_needs_no_escape_and_path_is_legal():
    world = _pocket_world()
    p = Planner(robot_radius_m=0.15)
    # start in open space: normal planning, all waypoints legal in the PLAN grid
    path = p.plan(_pose(1.25, 1.25), GoalCoordinate(1.85, 1.85), _map(world))
    assert path is not None
