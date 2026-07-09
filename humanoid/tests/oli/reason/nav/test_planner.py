"""TDD for grid A* path planning (nav/planner.py).

Proves the core of Albert's doubt at the unit level: given a map with obstacles, find a
collision-free route. Covers a straight path, adjacency of steps, routing through a gap in a
wall, and the None cases (goal blocked / enclosed). Pure: runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli.reason.nav import OccupancyGrid
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
