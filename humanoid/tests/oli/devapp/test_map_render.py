"""TDD for the pure nav-map rasterizer (devapp/map_render.py).

Verifies the north-up orientation + world→pixel mapping + non-destructive overlay — the geometry
that must be right for the robot dot to land where Oli actually is. Pure numpy: runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli.devapp.map_render import (
    base_rgb,
    compose,
    pixel_to_world,
    world_to_pixel,
)
from humanoid.logic.oli.reason.localization import RobotPose
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain


def test_pixel_to_world_inverts_world_to_pixel():
    g = OccupancyGrid(np.zeros((4, 3), dtype=bool), 1.0, origin=(0.0, 0.0))
    for x, y in [(0.5, 0.5), (2.5, 3.5), (1.5, 1.5)]:
        r, c = world_to_pixel(g, x, y)          # world → north-up pixel
        assert pixel_to_world(g, r, c) == (x, y)  # pixel → world (cell centers, exact)


def test_pixel_to_world_offset_origin():
    g = OccupancyGrid(np.zeros((4, 3), dtype=bool), 0.5, origin=(-2.0, -1.0))
    r, c = world_to_pixel(g, -0.75, 0.25)
    assert pixel_to_world(g, r, c) == (-0.75, 0.25)


def _grid(occ_rc=()):
    arr = np.zeros((4, 3), dtype=bool)
    for r, c in occ_rc:
        arr[r, c] = True
    return OccupancyGrid(arr, 1.0, origin=(0.0, 0.0))


def test_base_rgb_shape_and_north_up():
    g = _grid(occ_rc=[(0, 0)])  # occupied at grid row 0 = min-y
    img = base_rgb(g)
    assert img.shape == (4, 3, 3) and img.dtype == np.uint8
    assert tuple(img[-1, 0]) == (40, 40, 48)  # min-y occupied cell → BOTTOM image row
    assert tuple(img[0, 0]) != (40, 40, 48)   # top row is free/light


def test_world_to_pixel_north_up():
    g = _grid()
    assert world_to_pixel(g, 0.5, 0.5) == (3, 0)   # cell(0,0) center → bottom-left image
    assert world_to_pixel(g, 2.5, 3.5) == (0, 2)   # cell(3,2) center → top-right image


def test_compose_marks_robot_red_at_pose():
    g = _grid()
    base = base_rgb(g)
    img = compose(g, base, pose=RobotPose(stamp_ns=0, x=0.5, y=0.5, yaw=0.0))
    r, c = world_to_pixel(g, 0.5, 0.5)
    assert tuple(img[r, c]) == (230, 30, 30)


def test_compose_is_nondestructive():
    g = _grid()
    base = base_rgb(g)
    before = base.copy()
    compose(g, base, pose=RobotPose(stamp_ns=0, x=0.5, y=0.5, yaw=0.0), goal=(2.5, 3.5))
    assert np.array_equal(base, before)
