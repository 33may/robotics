"""TDD for the 2D occupancy grid (nav/costmap.py).

The grid is the map the planner searches: world↔cell transforms, occupancy queries (with
out-of-bounds blocked), and circle-footprint `inflate` (a box on the floor = occupied cells,
dilated by the robot radius so the planner treats the robot as a point). Pure: runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli.reason.nav import OccupancyGrid

pytestmark = pytest.mark.brain


def _free(rows, cols, res=1.0, origin=(0.0, 0.0)):
    return OccupancyGrid(np.zeros((rows, cols), dtype=bool), res, origin)


# ── construction + transforms ────────────────────────────────────────────────


def test_grid_requires_2d():
    with pytest.raises(ValueError):
        OccupancyGrid(np.zeros(4, dtype=bool), 1.0)


def test_cell_center_math():
    g = _free(3, 3, res=2.0)
    assert g.cell_to_world(0, 0) == (1.0, 1.0)  # center of first cell


def test_world_cell_roundtrip_at_center():
    g = _free(5, 5, res=0.5, origin=(1.0, 2.0))
    x, y = g.cell_to_world(3, 2)
    assert g.world_to_cell(x, y) == (3, 2)


# ── occupancy queries ────────────────────────────────────────────────────────


def test_is_occupied_reads_grid():
    arr = np.zeros((3, 3), dtype=bool)
    arr[1, 2] = True
    g = OccupancyGrid(arr, 1.0)
    assert g.is_occupied_cell(1, 2) is True
    assert g.is_occupied_cell(0, 0) is False


def test_out_of_bounds_is_blocked():
    g = _free(2, 2)
    assert g.is_occupied(-5.0, -5.0) is True
    assert g.is_occupied_cell(10, 10) is True


def test_is_occupied_world_coords():
    arr = np.zeros((4, 4), dtype=bool)
    arr[2, 3] = True
    g = OccupancyGrid(arr, 1.0)
    x, y = g.cell_to_world(2, 3)
    assert g.is_occupied(x, y) is True


# ── circle-footprint inflation ───────────────────────────────────────────────


def test_inflate_grows_obstacle_by_radius():
    arr = np.zeros((5, 5), dtype=bool)
    arr[2, 2] = True
    infl = OccupancyGrid(arr, 1.0).inflate(1.0)  # radius = 1 cell → 4-neighbors
    for r, c in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
        assert infl.is_occupied_cell(r, c) is True
    assert infl.is_occupied_cell(1, 1) is False  # diagonal (√2 > 1) stays free


def test_inflate_is_nondestructive():
    arr = np.zeros((3, 3), dtype=bool)
    arr[1, 1] = True
    g = OccupancyGrid(arr, 1.0)
    _ = g.inflate(1.0)
    assert g.is_occupied_cell(0, 1) is False  # original untouched


def test_inflate_zero_radius_is_copy():
    arr = np.zeros((3, 3), dtype=bool)
    arr[1, 1] = True
    g = OccupancyGrid(arr, 1.0)
    assert np.array_equal(g.inflate(0.0).grid, g.grid)
