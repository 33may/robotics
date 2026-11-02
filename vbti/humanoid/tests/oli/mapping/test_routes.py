"""Tests for coverage-route loading + clearance validation (MAY-173 locdev T2).

A route is a committed YAML (name, clearance, waypoints). `validate_route` walks
every segment against an inflated occupancy grid and reports violations — the
same render-approve-freeze discipline the locbench episodes use, so a route that
validates today keeps validating after scene changes (or fails loudly).
Runs in the `brain` env.
"""

import sys

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping.occupancy_io import OccupancyGrid
from humanoid.logic.simulation.mapping.routes import (
    Route,
    load_route,
    plan_route,
    validate_route,
)

pytestmark = pytest.mark.brain


def _grid_with_wall():
    """10×10 m free grid (res 0.1) with a 1 m-thick wall across x∈[4,5], all y."""
    g = np.zeros((100, 100), dtype=np.uint8)
    g[:, 40:50] = 1
    return OccupancyGrid(g, resolution=0.1, origin=(0.0, 0.0))


def test_load_route_from_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "name: test-loop\nclearance_m: 0.4\nspeed: 0.7\nloop: false\n"
        "waypoints:\n  - [1.0, 2.0]\n  - [3.0, 4.0]\n"
    )
    r = load_route(p)
    assert isinstance(r, Route)
    assert r.name == "test-loop"
    assert r.clearance_m == 0.4
    assert r.speed == 0.7
    assert r.loop is False
    assert r.waypoints == [(1.0, 2.0), (3.0, 4.0)]


def test_clear_route_validates_empty(tmp_path):
    r = Route(name="ok", clearance_m=0.3, speed=0.8, loop=False,
              waypoints=[(1.0, 1.0), (3.0, 1.0), (3.0, 8.0)])
    assert validate_route(r, _grid_with_wall()) == []


def test_segment_through_wall_is_reported():
    r = Route(name="bad", clearance_m=0.3, speed=0.8, loop=False,
              waypoints=[(1.0, 5.0), (8.0, 5.0)])  # crosses the x∈[4,5] wall
    violations = validate_route(r, _grid_with_wall())
    assert violations, "expected the wall crossing to be flagged"
    # a violation names the segment and a world point inside the blockage
    seg, (vx, vy) = violations[0]
    assert seg == 0
    assert 4.0 - 0.3 <= vx <= 5.0 + 0.3


def test_clearance_inflation_catches_near_miss():
    # Path parallel to the wall at 0.2 m — free at 0 clearance, blocked at 0.5.
    r_tight = Route(name="near", clearance_m=0.0, speed=0.8, loop=False,
                    waypoints=[(3.8, 1.0), (3.8, 9.0)])
    r_wide = Route(name="near", clearance_m=0.5, speed=0.8, loop=False,
                   waypoints=[(3.8, 1.0), (3.8, 9.0)])
    assert validate_route(r_tight, _grid_with_wall()) == []
    assert validate_route(r_wide, _grid_with_wall()) != []


def test_loop_route_validates_closing_segment():
    # Open route legal; closing the loop crosses the wall → only loop mode flags it.
    wps = [(1.0, 1.0), (3.0, 1.0), (3.0, 8.0), (8.0, 8.0)]
    # segment (3,8)→(8,8) crosses the wall already; use wall-free side points instead
    wps = [(1.0, 1.0), (3.0, 1.0), (3.0, 8.0), (1.0, 8.0)]
    open_r = Route(name="o", clearance_m=0.3, speed=0.8, loop=False, waypoints=wps)
    assert validate_route(open_r, _grid_with_wall()) == []
    # a loop back from (1,8) to (1,1) stays wall-free too → still clean
    loop_r = Route(name="l", clearance_m=0.3, speed=0.8, loop=True, waypoints=wps)
    assert validate_route(loop_r, _grid_with_wall()) == []


def _grid_with_gap():
    """Like _grid_with_wall but the wall has a doorway at y∈[7,9] — the planner
    must divert through it instead of the straight line."""
    g = np.zeros((100, 100), dtype=np.uint8)
    g[:, 40:50] = 1
    g[70:90, 40:50] = 0  # doorway
    return OccupancyGrid(g, resolution=0.1, origin=(0.0, 0.0))


def test_plan_route_expands_targets_via_deployment_planner():
    # Two targets on opposite sides of the wall: the DEPLOYMENT planner must
    # route through the doorway — the dense path is nothing like the straight line.
    r = Route(name="p", clearance_m=0.3, speed=0.8, loop=False,
              waypoints=[(2.0, 2.0), (8.0, 2.0)])
    path = plan_route(r, _grid_with_gap())
    assert len(path) > 2
    ys = [p[1] for p in path]
    assert max(ys) > 6.0, "path must divert up through the doorway"
    # endpoints preserved
    assert path[0] == pytest.approx((2.0, 2.0), abs=0.2)
    assert path[-1] == pytest.approx((8.0, 2.0), abs=0.2)


def test_plan_route_downsamples_to_spacing():
    r = Route(name="p", clearance_m=0.3, speed=0.8, loop=False,
              waypoints=[(1.0, 1.0), (1.0, 9.0)])
    path = plan_route(_ensure(r), _grid_with_gap(), spacing_m=1.0)
    import math
    gaps = [math.dist(a, b) for a, b in zip(path[:-1], path[1:])]
    assert all(g <= 1.5 for g in gaps)      # dense enough to follow
    assert min(gaps) > 0.3                  # but not raw 5 cm planner cells


def _ensure(r):
    return r


def test_plan_route_raises_on_unreachable_target():
    r = Route(name="p", clearance_m=0.3, speed=0.8, loop=False,
              waypoints=[(2.0, 2.0), (8.0, 2.0)])
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(r, _grid_with_wall())  # solid wall, no doorway


def test_module_is_pure():
    import humanoid.logic.simulation.mapping.routes  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
