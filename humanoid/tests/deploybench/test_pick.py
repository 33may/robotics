"""TDD for the picker's pure composition logic (logic/deploybench/pick.py::build_scenario).

The matplotlib event loop is untested GUI glue; all the decisions it makes route through
`build_scenario`, tested here: KNOWN defaults its heading to face the goal, KIDNAPPED carries
no hint, and an off-map / unroutable pick surfaces the FR2 reasons. Pure numpy/stdlib → brain.
"""

import numpy as np
import pytest

from humanoid.logic.deploybench.pick import build_scenario
from humanoid.logic.deploybench.scenario import StartMode
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain


def _split_grid(n=40, res=0.5, origin=(-10.0, -10.0)) -> OccupancyGrid:
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[n // 2, :] = True
    return OccupancyGrid(occ, res, origin)


def test_known_defaults_heading_to_goal():
    grid = _split_grid()
    scn, chk = build_scenario(name="n", map_dir="m", grid=grid,
                              start_xy=(-5.0, -5.0), goal_xy=(5.0, -5.0), start_mode="known")
    assert chk.ok, chk.reasons
    assert scn.start_mode is StartMode.KNOWN
    assert scn.start_yaw == pytest.approx(0.0, abs=1e-9)     # atan2(0, +10) = 0 (faces +x)


def test_known_explicit_yaw_wins():
    grid = _split_grid()
    scn, _ = build_scenario(name="n", map_dir="m", grid=grid, start_xy=(-5.0, -5.0),
                            goal_xy=(5.0, -5.0), start_mode="known", start_yaw=1.0)
    assert scn.start_yaw == pytest.approx(1.0)


def test_kidnapped_carries_no_hint():
    grid = _split_grid()
    scn, chk = build_scenario(name="n", map_dir="m", grid=grid,
                              start_xy=(-5.0, -5.0), goal_xy=(5.0, -5.0), start_mode="kidnapped")
    assert chk.ok
    assert scn.start_mode is StartMode.KIDNAPPED
    assert scn.start_yaw is None


def test_offmap_goal_surfaces_reason():
    grid = _split_grid()
    scn, chk = build_scenario(name="n", map_dir="m", grid=grid,
                              start_xy=(-5.0, -5.0), goal_xy=(50.0, 50.0), start_mode="known")
    assert not chk.ok
    assert any("outside" in r for r in chk.reasons), chk.reasons
