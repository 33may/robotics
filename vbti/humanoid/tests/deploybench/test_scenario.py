"""TDD for deploybench scenarios (logic/deploybench/scenario.py).

An interactive, per-run eval course: an operator-picked start + goal on a baked map, KNOWN or
KIDNAPPED start, saved as reproducible JSON. Validation reuses `nav.plan_path` (the route Nav
actually drives) and enforces FR2 — start/goal must lie in the mapped free space with a
drivable route between them. Pure numpy/stdlib → `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.deploybench.scenario import (
    Scenario,
    StartMode,
    load_scenario,
    save_scenario,
    validate,
)
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain


def _split_grid(n=40, res=0.5, origin=(-10.0, -10.0)) -> OccupancyGrid:
    """20×20 m room: border wall + a full horizontal wall across the middle row (no gap),
    so the top (y > 0) and bottom (y < 0) halves are unreachable from each other."""
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[n // 2, :] = True
    return OccupancyGrid(occ, res, origin)


def test_round_trip_known(tmp_path):
    s = Scenario(name="demo-known", map_dir="data/maps/x",
                 start=(-5.0, -5.0), goal=(5.0, -5.0),
                 start_mode=StartMode.KNOWN, start_yaw=1.57, arrival_tol_m=0.3)
    p = tmp_path / "scn.json"
    save_scenario(s, p)
    back = load_scenario(p)
    assert back == s
    assert back.start_mode is StartMode.KNOWN
    assert back.start_yaw == pytest.approx(1.57)


def test_round_trip_kidnapped(tmp_path):
    s = Scenario(name="demo-kidnap", map_dir="m",
                 start=(-5.0, -5.0), goal=(5.0, -5.0),
                 start_mode=StartMode.KIDNAPPED)
    p = tmp_path / "scn.json"
    save_scenario(s, p)
    back = load_scenario(p)
    assert back == s
    assert back.start_mode is StartMode.KIDNAPPED
    assert back.start_yaw is None


def test_validate_ok_same_side_reachable():
    grid = _split_grid()
    s = Scenario(name="ok", map_dir="m", start=(-5.0, -5.0), goal=(5.0, -5.0))
    chk = validate(s, grid)
    assert chk.ok, chk.reasons
    assert chk.route_m is not None and chk.route_m > 0.0


def test_validate_rejects_out_of_map():
    grid = _split_grid()
    s = Scenario(name="oom", map_dir="m", start=(-5.0, -5.0), goal=(50.0, 50.0))
    chk = validate(s, grid)
    assert not chk.ok
    assert any("outside" in r for r in chk.reasons), chk.reasons


def test_validate_rejects_obstacle_start():
    grid = _split_grid()
    # (0,0) lands on the middle wall row → occupied.
    s = Scenario(name="wall", map_dir="m", start=(0.0, 0.0), goal=(5.0, -5.0))
    chk = validate(s, grid)
    assert not chk.ok
    assert any("obstacle" in r for r in chk.reasons), chk.reasons


def test_validate_rejects_unreachable_across_wall():
    grid = _split_grid()
    s = Scenario(name="split", map_dir="m", start=(-5.0, -5.0), goal=(5.0, 5.0))
    chk = validate(s, grid)
    assert not chk.ok
    assert any("route" in r for r in chk.reasons), chk.reasons
