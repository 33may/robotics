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


def test_radius_for_extent_is_the_pinhole_formula_between_its_clamps():
    """The fill formula Stage B stopped calling, tested DIRECTLY.

    `radius_for_extent` survives the ray-views rebuild as an importable
    (ray-views Stage B plan, Global Constraints) and its only remaining
    exercise was `test_faceview`'s indirect R_MIN-clamp assert through
    `face_radius` — which cannot tell a correct formula from a constant
    that happens to clamp. So: the similar-triangles value itself, and both
    clamps it is held between.
    """
    from inspection.view.grid import (CLOSENESS, FILL_TARGET, R_MAX, R_MIN,
                                      radius_for_extent)
    fy, height_px = 430.0, 480

    # r = fy*L / (fill*height) / closeness — pick an extent that lands
    # strictly INSIDE [R_MIN, R_MAX] so neither clamp can hide the maths.
    want = 0.30
    extent = want * FILL_TARGET * height_px * CLOSENESS / fy
    assert R_MIN < want < R_MAX, "fixture must exercise the unclamped band"
    assert np.isclose(radius_for_extent(extent, fy, height_px), want)

    # Both clamps, and the documented `r_min=0.0` escape from the floor.
    assert radius_for_extent(1e-6, fy, height_px) == R_MIN
    assert radius_for_extent(10.0, fy, height_px) == R_MAX
    assert radius_for_extent(1e-6, fy, height_px, r_min=0.0) < R_MIN
    # `closeness` divides: asking for 2x closer halves the distance — read
    # unfloored, because half of this fixture's want lands under R_MIN and
    # the clamp would otherwise answer for the slider.
    assert np.isclose(radius_for_extent(extent, fy, height_px, r_min=0.0,
                                        closeness=CLOSENESS * 2), want / 2)


def test_grid_is_light():
    import inspection.view.grid as g
    src = Path(g.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.cell" not in src
    # importing grid must stay cheap. Checked in a FRESH interpreter: in-suite
    # `"pinocchio" not in sys.modules` only held when this file ran before any
    # test that boots a rig — suite order, not grid's import graph.
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, inspection.view.grid; sys.exit('pinocchio' in sys.modules)"],
        cwd=Path(g.__file__).resolve().parents[2])
    assert proc.returncode == 0, "importing grid dragged pinocchio in"


def main():
    test_step_delta_wraps(); test_cell_gloss_matches_decider_wording()
    test_neighbors_wrap_in_h_clamp_in_v(); test_coverage_map_ascii()
    test_moves_from_nearest_first()
    test_radius_for_extent_is_the_pinhole_formula_between_its_clamps()
    test_grid_is_light()
    print("OK test_grid")


if __name__ == "__main__":
    main()
