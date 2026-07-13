"""TDD for grid A* path planning (nav/planner.py).

Proves the core of Albert's doubt at the unit level: given a map with obstacles, find a
collision-free route. Covers a straight path, adjacency of steps, routing through a gap in a
wall, and the None cases (goal blocked / enclosed). Pure: runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping import OccupancyGrid
from humanoid.logic.oli.reason.nav.planner import plan_path

pytestmark = pytest.mark.brain


def _free(rows, cols, res=1.0, origin=(0.0, 0.0)):
    return OccupancyGrid(np.zeros((rows, cols), dtype=bool), res, origin)


def test_straight_path_in_empty_grid():
    g = _free(5, 5)
    path = plan_path(g, (0.5, 0.5), (4.5, 4.5))
    assert path is not None
    assert path[0] == g.cell_to_world(0, 0)
    assert path[-1] == g.cell_to_world(4, 4)


def test_path_steps_are_8connected_adjacent():
    g = _free(6, 6)
    path = plan_path(g, (0.5, 0.5), (5.5, 5.5))
    cells = [g.world_to_cell(x, y) for x, y in path]
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1


def test_path_routes_around_wall_through_gap():
    arr = np.zeros((5, 5), dtype=bool)
    arr[0:4, 2] = True  # vertical wall on col 2, gap only at row 4
    g = OccupancyGrid(arr, 1.0)
    path = plan_path(g, (0.5, 0.5), (4.5, 0.5))  # cell (0,0) → (0,4), must cross col 2
    assert path is not None
    assert all(not g.is_occupied(x, y) for x, y in path)  # never steps on the wall


def test_goal_occupied_returns_none():
    arr = np.zeros((3, 3), dtype=bool)
    arr[2, 2] = True
    g = OccupancyGrid(arr, 1.0)
    assert plan_path(g, (0.5, 0.5), (2.5, 2.5)) is None


def test_no_path_when_goal_enclosed():
    arr = np.zeros((5, 5), dtype=bool)
    arr[1, 1:4] = True
    arr[3, 1:4] = True
    arr[2, 1] = arr[2, 3] = True  # ring around free goal cell (2,2)
    g = OccupancyGrid(arr, 1.0)
    assert plan_path(g, (0.5, 0.5), (2.5, 2.5)) is None


# ── soft clearance cost: prefer the open route, still allow tight ────────────

def test_plan_path_hugs_wall_without_cost_bulges_with_cost():
    occ = np.zeros((3, 5), dtype=bool)
    occ[0, :] = True                       # wall along row 0
    g = OccupancyGrid(occ, resolution=1.0)
    start = g.cell_to_world(1, 0)
    goal = g.cell_to_world(1, 4)

    # no cost → shortest path hugs the wall (stays on row 1)
    base = plan_path(g, start, goal)
    assert base is not None
    assert all(g.world_to_cell(x, y)[0] == 1 for x, y in base)

    # clearance cost → path bulges to row 2 (farther from the wall) when there's room
    cost = g.clearance_cost(inflation_radius_m=2.0, weight=5.0)
    clr = plan_path(g, start, goal, cost=cost)
    assert clr is not None
    assert any(g.world_to_cell(x, y)[0] == 2 for x, y in clr)


def test_plan_path_cost_still_squeezes_through_only_gap():
    # one-cell gap in a wall: even with a huge clearance cost, the only route is the gap
    occ = np.zeros((5, 3), dtype=bool)
    occ[2, :] = True
    occ[2, 1] = False                      # single opening at (row 2, col 1)
    g = OccupancyGrid(occ, resolution=1.0)
    cost = g.clearance_cost(inflation_radius_m=3.0, weight=100.0)
    path = plan_path(g, g.cell_to_world(0, 1), g.cell_to_world(4, 1), cost=cost)
    assert path is not None
    assert (2, 1) in [g.world_to_cell(x, y) for x, y in path]


def test_plan_path_weighted_astar_still_valid():
    g = _free(12, 12)
    s, go = g.cell_to_world(0, 0), g.cell_to_world(11, 11)
    exact = plan_path(g, s, go)
    weighted = plan_path(g, s, go, weight=2.0)
    assert exact and weighted
    assert weighted[0] == exact[0] and weighted[-1] == exact[-1]  # same endpoints, valid route
