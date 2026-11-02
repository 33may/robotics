"""Smoke TDD for locbench plots (logic/locbench/plots.py) — D12.

Plots are judged by eyes, not asserts — these tests pin the CONTRACT: every plot function
writes a nonempty PNG, survives LOST-heavy and empty-ish inputs, and the run sheet is wider
than tall (ultrawide-friendly). `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.locbench.episodes import Episode
from humanoid.logic.locbench.pairs import PosePair
from humanoid.logic.locbench.plots import (
    plot_episode_overlay,
    plot_error_distribution,
    plot_error_timeline,
    plot_run_sheet,
)
from humanoid.logic.oli.reason.localization import LocalizationStatus
from humanoid.logic.oli.reason.mapping import OccupancyGrid

pytestmark = pytest.mark.brain

S = 1_000_000_000


def _grid(n=40):
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return OccupancyGrid(occ, 0.5)


def _pairs(n=30, bias=0.05, lost_every=7):
    out = []
    for i in range(n):
        gt = (2.0 + i * 0.3, 5.0, 0.0)
        lost = lost_every and i % lost_every == 0
        out.append(PosePair(
            stamp_ns=(10 + i) * S,
            est=None if lost else (gt[0] + bias, gt[1] + bias, 0.01),
            gt=gt,
            status=LocalizationStatus.LOST if lost else LocalizationStatus.TRACKING,
            assoc_dt_ns=0))
    return out


_EP = Episode(id=0, spawn=(2.0, 5.0), goal=(12.0, 5.0), route_m=10.0)


def _nonempty(path):
    assert path.exists() and path.stat().st_size > 1000, f"{path} missing or empty"


def test_episode_overlay(tmp_path):
    _nonempty(plot_episode_overlay(_grid(), _pairs(), _EP, tmp_path / "ep.png",
                                   title="ep 0 — PASS"))


def test_overlay_survives_all_lost(tmp_path):
    _nonempty(plot_episode_overlay(_grid(), _pairs(lost_every=1), _EP, tmp_path / "l.png"))


def test_error_timeline(tmp_path):
    _nonempty(plot_error_timeline([_pairs(), _pairs(bias=0.2)], tmp_path / "t.png"))


def test_run_sheet_is_wide(tmp_path):
    pairs = [_pairs() for _ in range(6)]
    eps = [_EP] * 6
    p = plot_run_sheet(_grid(), eps, pairs, ["PASS"] * 5 + ["FAIL"], tmp_path / "sheet.png")
    _nonempty(p)
    from PIL import Image
    w, h = Image.open(p).size
    assert w > h    # ultrawide monitor: grids go wide, not tall


def test_error_distribution(tmp_path):
    _nonempty(plot_error_distribution([_pairs()], tmp_path / "d.png"))


def test_distribution_survives_no_answers(tmp_path):
    _nonempty(plot_error_distribution([_pairs(lost_every=1)], tmp_path / "e.png"))
