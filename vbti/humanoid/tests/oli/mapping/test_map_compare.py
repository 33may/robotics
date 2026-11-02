"""Tests for map_compare — register a candidate occupancy grid to the GT map
(flip-aware) and score it for navigation-grade precision.

The map frame is REFLECTED vs world (registration.json in every bake), so the
2D fit must allow det(R) = -1. GT is the eval oracle only — none of this runs
in the deployment path.
"""

import numpy as np
import pytest

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid
from humanoid.logic.simulation.mapping.map_compare import (
    apply_transform2d,
    boundary_error,
    fit_traj_transform,
    refine_grid_alignment,
    score_grids,
)

pytestmark = pytest.mark.brain


def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ── fit_traj_transform ────────────────────────────────────────────────────────


class TestFitTrajTransform:
    def test_recovers_rigid_transform(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(-5, 5, (100, 2))
        R = rot2d(0.4)
        t = np.array([1.5, -2.0])
        dst = src @ R.T + t
        T = fit_traj_transform(src, dst)
        assert np.allclose(T["R"], R, atol=1e-9)
        assert np.allclose(T["t"], t, atol=1e-9)
        assert not T["mirrored"]

    def test_recovers_reflected_transform(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(-5, 5, (100, 2))
        M = rot2d(0.3) @ np.diag([1.0, -1.0])  # reflection
        t = np.array([-0.5, 3.0])
        dst = src @ M.T + t
        T = fit_traj_transform(src, dst)
        assert np.allclose(T["R"], M, atol=1e-9)
        assert T["mirrored"]
        # residual ~0
        assert T["residual_mean"] < 1e-9

    def test_apply_transform_roundtrip(self):
        rng = np.random.default_rng(2)
        src = rng.uniform(-5, 5, (50, 2))
        M = rot2d(-0.7) @ np.diag([1.0, -1.0])
        t = np.array([2.0, 1.0])
        dst = src @ M.T + t
        T = fit_traj_transform(src, dst)
        assert np.allclose(apply_transform2d(src, T), dst, atol=1e-9)


# ── grids for registration/scoring ────────────────────────────────────────────


def bar_grid(origin=(0.0, 0.0), shape=(100, 100), res=0.05, bars=((20, 25), (50, 55), (80, 85))):
    """Free grid with horizontal occupied bars (rack-row-like)."""
    g = np.zeros(shape, bool)
    for r0, r1 in bars:
        g[r0:r1, 10:90] = True
    return OccupancyGrid(g, res, origin)


class TestRefineGridAlignment:
    def test_recovers_small_offset(self):
        gt = bar_grid()
        # shifted origin moves the candidate's content +0.10 m in world y;
        # refine must find the -0.10 m correction (y is strongly constrained
        # by the bar thickness; x runs along the bars and is weak)
        cand = bar_grid(origin=(0.0, 0.10))
        seed = {"R": np.eye(2), "t": np.zeros(2), "mirrored": False}
        T = refine_grid_alignment(cand, gt, seed,
                                  search_t=0.3, step_t=0.05, search_deg=2.0, step_deg=0.5)
        assert abs(T["dy"] + 0.10) <= 0.051
        assert abs(T["dtheta_deg"]) <= 0.51

    def test_finds_known_translation_error(self):
        gt = bar_grid()
        cand = bar_grid()
        # corrupt the seed by 0.15 m in y: refine must recover it
        seed = {"R": np.eye(2), "t": np.array([0.0, 0.15]), "mirrored": False}
        T = refine_grid_alignment(cand, gt, seed,
                                  search_t=0.3, step_t=0.05, search_deg=1.0, step_deg=0.5)
        assert abs(T["dy"] + 0.15) <= 0.051  # correction cancels the corruption


# ── scoring ───────────────────────────────────────────────────────────────────


class TestScoreGrids:
    def test_perfect_candidate_scores_clean(self):
        gt = bar_grid()
        cand = bar_grid()
        T = {"R": np.eye(2), "t": np.zeros(2), "mirrored": False}
        s = score_grids(cand, gt, T)
        assert s["false_free_pct"] == 0.0
        assert s["false_blocked_observed_pct"] == 0.0

    def test_false_free_detected_and_split(self):
        gt = bar_grid()
        g = bar_grid().grid.copy()
        # carve a hole in a GT rack: candidate says free where GT occupied
        g[50:55, 40:44] = False
        cand = OccupancyGrid(g, 0.05, (0.0, 0.0))
        T = {"R": np.eye(2), "t": np.zeros(2), "mirrored": False}
        s = score_grids(cand, gt, T)
        assert s["false_free_pct"] > 0.0
        # 5-row bar: holes at rows 50-51 are within 0.15m of the bar edge
        # (smear), rows 52-54 interior... with 0.3m threshold all 4x5=20 cells
        # sit within 0.3m of GT free -> classified smear
        assert s["false_free_smear_pct"] > 0.0

    def test_boundary_error_zero_for_identical(self):
        gt = bar_grid()
        cand = bar_grid()
        T = {"R": np.eye(2), "t": np.zeros(2), "mirrored": False}
        be = boundary_error(cand, gt, T)
        assert be["p50_m"] == 0.0
        assert be["p95_m"] <= 0.05

    def test_interior_split_excludes_frontier_phantoms(self):
        gt = bar_grid()
        g = bar_grid().grid.copy()
        # phantom obstacle far from GT bars, at the coverage frontier:
        # observed ends at row 90 (beyond = unknown -> BLOCKED in the artifact);
        # phantom at rows 88-89 touches that unknown region
        g[88:90, 30:60] = True
        g[90:, :] = True  # unknown = blocked
        observed = np.zeros_like(g)
        observed[:90, :] = True
        cand = OccupancyGrid(g, 0.05, (0.0, 0.0))
        T = {"R": np.eye(2), "t": np.zeros(2), "mirrored": False}
        s = score_grids(cand, gt, T, observed_mask=observed)
        # interior precision clean (real bars only); overall polluted
        assert s["occupied_precision_interior_pct"] > s["occupied_precision_pct"]
        assert s["occupied_precision_interior_pct"] == 100.0

    def test_boundary_error_measures_shift(self):
        gt = bar_grid()
        cand = bar_grid()
        # transform with a 0.2 m y-offset: every candidate occupied cell lands
        # 0.2 m from its GT counterpart (bars are 0.25m thick, 1.5m apart)
        T = {"R": np.eye(2), "t": np.array([0.0, 0.2]), "mirrored": False}
        be = boundary_error(cand, gt, T)
        assert 0.1 <= be["p50_m"] <= 0.25
