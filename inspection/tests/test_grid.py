#!/usr/bin/env python3
"""Grid vocabulary — index math, gloss, coverage ASCII.
Run: p inspection/tests/test_grid.py"""
import sys
from pathlib import Path

import numpy as np

from inspection.view.grid import (H_BINS, V_ELEVATIONS, cell_gloss,
                                  coverage_map, moves_from, neighbors,
                                  step_delta)


def test_step_delta_wraps():
    assert step_delta(5, 6) == 1
    assert step_delta(11, 0) == 1            # wraps forward past h=0
    assert step_delta(0, 11) == -1
    assert abs(step_delta(5, 11)) == H_BINS // 2      # opposite side


def test_cell_gloss_matches_decider_wording():
    assert cell_gloss((5, 1), (6, 1)) == "one step right, same height"
    assert cell_gloss((5, 1), (3, 1)) == "two steps left, same height"
    assert cell_gloss((11, 0), (0, 0)) == "one step right, same height"
    assert cell_gloss((5, 1), (11, 2)) == "opposite side, higher"
    assert cell_gloss((5, 1), (5, 0)) == "same side, lower"
    assert cell_gloss(None, (3, 2)) == "elevation 70 deg"


def test_neighbors_wrap_in_h_clamp_in_v():
    assert set(neighbors((0, 0))) == {(11, 0), (1, 0), (0, 1)}   # v floor
    assert len(neighbors((0, 1))) == 4
    assert set(neighbors((0, 2))) == {(11, 2), (1, 2), (0, 1)}   # v ceiling
    assert (0, 0) not in neighbors((0, 0))


def test_coverage_map_ascii():
    cov = np.zeros((3, 12), dtype=bool)
    cov[0, 2] = True; cov[1, 3] = True
    txt = coverage_map(cov, cur=(3, 1))
    assert txt.count("#") == 1 and txt.count("@") == 1   # cur overrides its mark
    assert txt.splitlines()[0].startswith("v2")          # highest elevation first


def test_moves_from_nearest_first():
    mv = moves_from((5, 1), {(6, 1), (4, 1), (5, 2), (11, 2)})
    assert mv[0].cell == (4, 1)                  # tie at distance 1, tuple order
    assert mv[0].gloss == "one step left, same height"
    assert mv[-1].cell == (11, 2)                # farthest last


def test_grid_is_light():
    import inspection.view.grid as g
    src = Path(g.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.cell" not in src
    assert "pinocchio" not in sys.modules        # importing grid must stay cheap


def main():
    test_step_delta_wraps(); test_cell_gloss_matches_decider_wording()
    test_neighbors_wrap_in_h_clamp_in_v(); test_coverage_map_ascii()
    test_moves_from_nearest_first(); test_grid_is_light()
    print("OK test_grid")


if __name__ == "__main__":
    main()
