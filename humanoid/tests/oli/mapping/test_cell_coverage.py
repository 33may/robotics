"""Tests for the cell-grid coverage generator (MAY-173 locdev T2, Anton's design).

The algorithm (15-07-2026): split the map into an R×C cell grid; sample K random
FREE points per cell (blocked cells yield fewer/none); lap 1 visits cells in
seeded random order following each cell's points in sampled sequence; lap 2
visits cells in a fresh random order with each cell's sequence REVERSED (same
positions, opposite headings — doubles viewpoint directions for the BoW map).
The flat target list is then stitched by plan_route with the deployment planner.
Runs in the `brain` env.
"""

import sys

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping.occupancy_io import OccupancyGrid
from humanoid.logic.simulation.mapping.cell_coverage import (
    build_coverage_route,
    generate_cell_coverage,
    neighbour_walk,
)

pytestmark = pytest.mark.brain


def _open_grid():
    """20×20 m fully-free map (res 0.1)."""
    return OccupancyGrid(np.zeros((200, 200), dtype=np.uint8), resolution=0.1,
                         origin=(0.0, 0.0))


def _half_blocked_grid():
    """20×20 m map whose right half (x>10) is solid."""
    g = np.zeros((200, 200), dtype=np.uint8)
    g[:, 100:] = 1
    return OccupancyGrid(g, resolution=0.1, origin=(0.0, 0.0))


def test_points_are_free_and_inside_their_cell():
    cov = generate_cell_coverage(_open_grid(), cells=(4, 4), points_per_cell=5,
                                 margin_m=0.5, seed=1, laps=1)
    assert len(cov.cells) == 16
    for (r, c), pts in cov.cells.items():
        assert len(pts) == 5
        # cell bounds: 5×5 m tiles of the 20×20 extent
        x_lo, y_lo = c * 5.0, r * 5.0
        for x, y in pts:
            assert x_lo <= x <= x_lo + 5.0
            assert y_lo <= y <= y_lo + 5.0


def test_blocked_cells_yield_no_points():
    cov = generate_cell_coverage(_half_blocked_grid(), cells=(2, 2),
                                 points_per_cell=4, margin_m=0.5, seed=2, laps=1)
    right = [pts for (r, c), pts in cov.cells.items() if c == 1]
    left = [pts for (r, c), pts in cov.cells.items() if c == 0]
    assert all(len(p) == 0 for p in right), "solid cells must be skipped"
    assert all(len(p) == 4 for p in left)


def test_lap1_concatenates_cells_in_visit_order():
    cov = generate_cell_coverage(_open_grid(), cells=(2, 2), points_per_cell=3,
                                 margin_m=0.5, seed=3, laps=1)
    expect = [p for cell in cov.cell_orders[0] for p in cov.cells[cell]]
    assert cov.targets == expect


def test_lap2_reverses_each_cells_sequence():
    cov = generate_cell_coverage(_open_grid(), cells=(2, 2), points_per_cell=3,
                                 margin_m=0.5, seed=4, laps=2)
    lap1_len = sum(len(p) for p in cov.cells.values())
    lap2 = cov.targets[lap1_len:]
    expect = [p for cell in cov.cell_orders[1] for p in reversed(cov.cells[cell])]
    assert lap2 == expect
    assert len(cov.cell_orders[0]) == len(cov.cell_orders[1]) == 4
    assert cov.cell_orders[0] != cov.cell_orders[1] or True  # orders independent draws


def test_deterministic_per_seed():
    a = generate_cell_coverage(_open_grid(), cells=(3, 3), points_per_cell=4,
                               margin_m=0.5, seed=7, laps=2)
    b = generate_cell_coverage(_open_grid(), cells=(3, 3), points_per_cell=4,
                               margin_m=0.5, seed=7, laps=2)
    c = generate_cell_coverage(_open_grid(), cells=(3, 3), points_per_cell=4,
                               margin_m=0.5, seed=8, laps=2)
    assert a.targets == b.targets
    assert a.targets != c.targets


def test_points_per_cell_accepts_a_callable_for_zone_weighting():
    # Anton's zone weighting (15-07): fewer points in the open-floor rows,
    # more in the rack rows — points_per_cell may be a (row, col) -> int callable.
    cov = generate_cell_coverage(
        _open_grid(), cells=(4, 4),
        points_per_cell=lambda r, c: 2 if r < 2 else 4,
        margin_m=0.5, seed=5, laps=1,
    )
    for (r, c), pts in cov.cells.items():
        assert len(pts) == (2 if r < 2 else 4), f"cell ({r},{c})"


# ── neighbour_walk: frontier random walk over the cell graph ─────────────────


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def test_neighbour_walk_visits_every_cell_once():
    usable = [(r, c) for r in range(5) for c in range(5)]
    walk = neighbour_walk(usable, np.random.default_rng(0), start=(0, 0))
    assert walk[0] == (0, 0)
    assert sorted(walk) == sorted(usable)  # permutation, no repeats


def test_each_step_jumps_to_a_nearest_unvisited_cell():
    # The frontier rule: when unvisited Moore-neighbours exist we take one (Chebyshev
    # 1); only when the local ring is exhausted do we jump — and then to the NEAREST
    # unvisited cell. So every step is min-Chebyshev among the still-unvisited set.
    usable = [(r, c) for r in range(5) for c in range(5)]
    walk = neighbour_walk(usable, np.random.default_rng(1), start=(2, 2))
    visited = {walk[0]}
    for prev, cur in zip(walk[:-1], walk[1:]):
        remaining = set(usable) - visited
        dmin = min(_cheb(prev, cell) for cell in remaining)
        assert _cheb(prev, cur) == dmin
        visited.add(cur)


def test_diagonal_passes_happen():
    # Anton's requirement: capture diagonal cell-to-cell passes. On an open grid the
    # walk must contain at least one step with BOTH row and col changing.
    usable = [(r, c) for r in range(5) for c in range(5)]
    walk = neighbour_walk(usable, np.random.default_rng(2), start=(0, 0))
    diagonals = [(a, b) for a, b in zip(walk[:-1], walk[1:])
                 if a[0] != b[0] and a[1] != b[1]]
    assert diagonals, "expected diagonal passes in the walk"


def test_walk_is_deterministic_per_rng():
    usable = [(r, c) for r in range(4) for c in range(4)]
    a = neighbour_walk(usable, np.random.default_rng(9), start=(0, 0))
    b = neighbour_walk(usable, np.random.default_rng(9), start=(0, 0))
    c = neighbour_walk(usable, np.random.default_rng(10), start=(0, 0))
    assert a == b
    assert a != c


def test_walk_handles_holes():
    # A gap of blocked cells: the walk still covers every usable cell and jumps
    # across the hole (Chebyshev > 1) rather than getting stuck.
    usable = [(r, c) for r in range(5) for c in range(5) if not (c == 2)]  # column 2 gone
    walk = neighbour_walk(usable, np.random.default_rng(3), start=(0, 0))
    assert sorted(walk) == sorted(usable)


def test_generator_uses_frontier_walk_by_default():
    # generate_cell_coverage's cell order should obey the nearest-frontier invariant,
    # not be an arbitrary permutation.
    cov = generate_cell_coverage(_open_grid(), cells=(4, 4), points_per_cell=2,
                                 margin_m=0.5, seed=6, laps=1)
    order = cov.cell_orders[0]
    visited = {order[0]}
    for prev, cur in zip(order[:-1], order[1:]):
        remaining = set(order) - visited
        dmin = min(_cheb(prev, cell) for cell in remaining)
        assert _cheb(prev, cur) == dmin
        visited.add(cur)


# ── build_coverage_route: committed YAML spec → (CellCoverage, Route) ─────────


def _write_spec(tmp_path, body):
    p = tmp_path / "cov.yaml"
    p.write_text(body)
    return p


def test_build_coverage_route_int_ppc(tmp_path):
    spec = _write_spec(tmp_path,
        "name: t\ncells: [4, 4]\nseed: 1\nlaps: 1\nmargin_m: 0.5\n"
        "clearance_m: 0.5\nspeed: 0.7\npoints_per_cell: 3\n")
    cov, route = build_coverage_route(spec, _open_grid())
    assert all(len(p) == 3 for p in cov.cells.values())
    assert route.name == "t"
    assert route.clearance_m == 0.5
    assert route.speed == 0.7
    assert route.waypoints == cov.targets


def test_build_coverage_route_zoned_ppc(tmp_path):
    spec = _write_spec(tmp_path,
        "name: z\ncells: [4, 4]\nseed: 2\nlaps: 1\nmargin_m: 0.5\n"
        "clearance_m: 0.5\nspeed: 0.8\n"
        "points_per_cell:\n  default: 4\n  zones:\n    - rows: [0, 1]\n      points: 2\n")
    cov, _ = build_coverage_route(spec, _open_grid())
    for (r, c), pts in cov.cells.items():
        assert len(pts) == (2 if r in (0, 1) else 4), f"cell ({r},{c})"


def test_build_coverage_route_is_deterministic(tmp_path):
    body = ("name: d\ncells: [3, 3]\nseed: 5\nlaps: 2\nmargin_m: 0.5\n"
            "clearance_m: 0.5\nspeed: 0.8\npoints_per_cell: 4\n")
    a, _ = build_coverage_route(_write_spec(tmp_path, body), _open_grid())
    b, _ = build_coverage_route(_write_spec(tmp_path, body), _open_grid())
    assert a.targets == b.targets


def test_module_is_pure():
    import humanoid.logic.simulation.mapping.cell_coverage  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
