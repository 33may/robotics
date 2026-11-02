"""locbench/plots.py — the eyes on a run (design.md D12).

Four artifacts per run, all committed:

  - episode overlay: GT vs estimated trajectory on the occupancy map, LOST stretches marked
    on the GT track (that is where the candidate went silent), spawn/goal labeled;
  - error timeline: per-tick position error vs time with the PASS/DEPLOY gate lines;
  - run contact sheet: every episode overlay in one wide grid (ultrawide-friendly:
    more columns than rows) — the one-glance picture of a run;
  - error distribution: CDF of all per-tick errors across the run vs the gate lines.

matplotlib Agg (headless); consumed by `locbench run/score`. Anton reads these; the agent
reads report.json — same numbers, two audiences.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..oli.reason.mapping import OccupancyGrid  # noqa: E402
from .episodes import Episode  # noqa: E402
from .pairs import PosePair  # noqa: E402
from .verdict import DEPLOY, PASS  # noqa: E402


def _extent(grid: OccupancyGrid):
    x0, y0 = grid.origin
    return (x0, x0 + grid.ncols * grid.resolution, y0, y0 + grid.nrows * grid.resolution)


def _draw_overlay(ax, grid: OccupancyGrid, pairs: Sequence[PosePair],
                  episode: Optional[Episode] = None, title: str = "") -> None:
    ax.imshow(~grid.grid, cmap="gray", origin="lower", extent=_extent(grid),
              interpolation="nearest")
    gt_xy = np.asarray([p.gt[:2] for p in pairs]) if pairs else np.empty((0, 2))
    if len(gt_xy):
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="tab:green", linewidth=2.0,
                label="GT", zorder=4)
        lost = np.asarray([p.est is None for p in pairs])
        if lost.any():
            ax.plot(gt_xy[lost, 0], gt_xy[lost, 1], "x", color="black", markersize=5,
                    label="LOST", zorder=6)
        est_xy = np.asarray([p.est[:2] for p in pairs if p.est is not None])
        if len(est_xy):
            ax.plot(est_xy[:, 0], est_xy[:, 1], "--", color="tab:red", linewidth=1.6,
                    label="est", zorder=5)
    if episode is not None:
        ax.plot(*episode.spawn, "o", color="tab:blue", markersize=9, zorder=7)
        ax.plot(*episode.goal, "*", color="tab:blue", markersize=15, zorder=7)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def plot_episode_overlay(grid: OccupancyGrid, pairs: Sequence[PosePair],
                         episode: Optional[Episode], out_png: str | Path,
                         title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(10, 10 * grid.nrows / grid.ncols))
    _draw_overlay(ax, grid, pairs, episode, title)
    if pairs:
        ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out_png)


def plot_error_timeline(pairs_per_episode: Sequence[Sequence[PosePair]],
                        out_png: str | Path) -> Path:
    """Per-tick position error vs time-into-episode, one line per episode, gate lines."""
    fig, ax = plt.subplots(figsize=(14, 4))   # wide — ultrawide monitor, timelines read flat
    for i, pairs in enumerate(pairs_per_episode):
        answered = [p for p in pairs if p.est is not None]
        if not answered:
            continue
        t0 = answered[0].stamp_ns
        t = [(p.stamp_ns - t0) / 1e9 for p in answered]
        e = [math.dist(p.est[:2], p.gt[:2]) for p in answered]
        ax.plot(t, e, linewidth=1.0, alpha=0.8, label=f"ep {i}")
    ax.axhline(PASS.max_pos_m, color="tab:orange", linestyle="--", linewidth=1.0,
               label=f"PASS max ({PASS.max_pos_m} m)")
    ax.axhline(DEPLOY.max_pos_m, color="tab:green", linestyle="--", linewidth=1.0,
               label=f"DEPLOY max ({DEPLOY.max_pos_m} m)")
    ax.set_xlabel("time into episode [s]")
    ax.set_ylabel("position error [m]")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    return _save(fig, out_png)


def plot_run_sheet(grid: OccupancyGrid, episodes: Sequence[Optional[Episode]],
                   pairs_per_episode: Sequence[Sequence[PosePair]],
                   verdict_tiers: Sequence[str], out_png: str | Path) -> Path:
    """Every episode overlay in one wide grid — the one-glance picture of a run."""
    n = len(pairs_per_episode)
    if n == 0:
        raise ValueError("nothing to plot — no episodes")
    ncols = min(5, max(1, math.ceil(n / 2)))   # wide: more columns than rows
    nrows = math.ceil(n / ncols)
    cell_h = 3.2 * grid.nrows / grid.ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, cell_h * nrows),
                             squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        _draw_overlay(ax, grid, pairs_per_episode[i], episodes[i],
                      title=f"ep {i} — {verdict_tiers[i]}")
        color = {"DEPLOY": "green", "PASS": "olive", "FAIL": "red"}.get(verdict_tiers[i])
        if color:
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(2.5)
    return _save(fig, out_png)


def plot_error_distribution(pairs_per_episode: Sequence[Sequence[PosePair]],
                            out_png: str | Path) -> Path:
    """CDF of every per-tick position error in the run vs the gate lines."""
    errors: List[float] = []
    for pairs in pairs_per_episode:
        errors += [math.dist(p.est[:2], p.gt[:2]) for p in pairs if p.est is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    if errors:
        xs = np.sort(np.asarray(errors))
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), linewidth=1.8)
    ax.axvline(PASS.mean_pos_m, color="tab:orange", linestyle=":", label="PASS mean gate")
    ax.axvline(PASS.max_pos_m, color="tab:orange", linestyle="--", label="PASS max gate")
    ax.axvline(DEPLOY.max_pos_m, color="tab:green", linestyle="--", label="DEPLOY max gate")
    ax.set_xlabel("position error [m]")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_png)


def _save(fig, out_png: str | Path) -> Path:
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
