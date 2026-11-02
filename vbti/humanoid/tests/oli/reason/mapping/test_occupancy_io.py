"""TDD for the baked occupancy artifact loader/saver (mapping/occupancy_io.py).

The brain loads a plain npy+json artifact (baked offline from the scene USD) — never importing
isaac. Round-trips a grid through save→load, preserving occupancy, resolution, and origin. Pure:
runs in `brain`.
"""

import json

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping import OccupancyGrid
from humanoid.logic.oli.reason.mapping.occupancy_io import (
    load_occupancy,
    occupancy_from_image,
    save_occupancy,
)

pytestmark = pytest.mark.brain


def test_occupancy_roundtrip(tmp_path):
    arr = np.zeros((4, 5), dtype=bool)
    arr[1, 2] = True
    arr[3, 4] = True
    g = OccupancyGrid(arr, 0.05, origin=(-1.0, 2.5))

    save_occupancy(g, str(tmp_path / "map"))
    g2 = load_occupancy(str(tmp_path / "map"))

    assert np.array_equal(g2.grid, g.grid)
    assert g2.resolution == pytest.approx(0.05)
    assert g2.origin == (-1.0, 2.5)
    assert g2.is_occupied_cell(1, 2) is True
    assert g2.is_occupied_cell(0, 0) is False


def test_saved_meta_is_plain_json(tmp_path):
    g = OccupancyGrid(np.zeros((2, 2), dtype=bool), 0.1, origin=(0.0, 0.0))
    save_occupancy(g, str(tmp_path / "m"))
    with open(str(tmp_path / "m" / "occupancy.json")) as f:
        meta = json.load(f)
    assert meta == {"resolution": 0.1, "origin": [0.0, 0.0]}


# ── ROS/omap PNG → OccupancyGrid conversion ──────────────────────────────────


def test_image_classifies_three_values():
    img = np.array([[0], [127], [255]], dtype=np.uint8)  # top black, mid grey, bottom white
    g = occupancy_from_image(img, 1.0, (0.0, 0.0), unknown_occupied=False)
    # flipud: grid row0 = bottom (white/free) ... row2 = top (black/occupied)
    assert g.grid[0, 0] == False
    assert g.grid[1, 0] == False  # grey, unknown→free here
    assert g.grid[2, 0] == True


def test_unknown_folds_to_occupied_by_default():
    g = occupancy_from_image(np.array([[127]], dtype=np.uint8), 1.0, (0.0, 0.0))
    assert g.grid[0, 0] == True


def test_flip_puts_png_top_row_at_grid_max_y():
    img = np.array([[0], [255]], dtype=np.uint8)  # top black, bottom white
    g = occupancy_from_image(img, 1.0, (0.0, 0.0))
    assert g.grid[-1, 0] == True  # PNG top → max-y → last grid row
    assert g.grid[0, 0] == False  # PNG bottom → min-y → row 0


def test_rgba_uses_one_channel_and_carries_meta():
    g = occupancy_from_image(np.zeros((1, 1, 4), dtype=np.uint8), 0.05, (-26.925, -23.425))
    assert g.grid[0, 0] == True  # black RGBA → occupied
    assert g.resolution == pytest.approx(0.05)
    assert g.origin == (-26.925, -23.425)


def test_negate_inverts_meaning():
    g = occupancy_from_image(np.array([[255]], dtype=np.uint8), 1.0, (0.0, 0.0), negate=1)
    assert g.grid[0, 0] == True  # negate=1 → white is occupied
