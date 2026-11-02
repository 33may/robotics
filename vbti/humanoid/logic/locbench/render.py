"""locbench/render.py — draw an episode set over the baked occupancy map (D2 approval, D12).

One figure, world coordinates: occupancy in grayscale, per-episode spawn (S#, green) → goal
(G#, red) with the planned A* route between them, the mapping-pass coverage tour (blue dashed,
visit order numbered), and the robot's boot origin (black ×). This is the picture Anton
approves before an episode set is frozen — and the base layer eval overlays reuse (§4.5).

matplotlib Agg only (headless-safe); imported lazily by the CLI so the pure sampling/scoring
paths never pay for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..oli.reason.localization import RobotPose  # noqa: E402
from ..oli.reason.mapping import OccupancyGrid, StaticMapping  # noqa: E402
from ..oli.reason.nav import GoalCoordinate, build_planner  # noqa: E402
from .episodes import EpisodeSet  # noqa: E402


def _extent(grid: OccupancyGrid):
    x0, y0 = grid.origin
    return (x0, x0 + grid.ncols * grid.resolution, y0, y0 + grid.nrows * grid.resolution)


def render_episode_set(
    grid: OccupancyGrid,
    es: EpisodeSet,
    out_png: str | Path,
) -> Path:
    """Render the approval picture; returns the written PNG path.

    Routes are planned with the DEPLOYMENT planner (`build_planner()` — clearance cost,
    weighted heuristic), so the picture Anton approves is the route Nav actually drives.
    (Sampling constraints deliberately use the knob-free shortest path instead — a frozen
    set must not change when planner knobs are retuned; see episodes.py.)"""
    world_map = StaticMapping.from_grid(grid).latest()
    planner = build_planner()
    fig, ax = plt.subplots(figsize=(16, 16 * grid.nrows / grid.ncols))
    ax.imshow(~grid.grid, cmap="gray", origin="lower", extent=_extent(grid),
              interpolation="nearest")

    origin_xy = _origin_from(es)
    if origin_xy is not None:
        ax.plot(*origin_xy, "kx", markersize=14, markeredgewidth=3, zorder=6)
        ax.annotate("boot", origin_xy, textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight="bold")

    # coverage tour (under the episodes)
    if es.coverage_goals:
        cx = [g[0] for g in es.coverage_goals]
        cy = [g[1] for g in es.coverage_goals]
        if origin_xy is not None:
            cx, cy = [origin_xy[0], *cx], [origin_xy[1], *cy]
        ax.plot(cx, cy, "--", color="tab:blue", linewidth=1.2, alpha=0.6, zorder=3)
        for i, g in enumerate(es.coverage_goals):
            ax.plot(*g, "o", color="tab:blue", markersize=7, alpha=0.7, zorder=4)
            ax.annotate(str(i + 1), g, textcoords="offset points", xytext=(6, 6),
                        fontsize=10, color="tab:blue")

    # episodes: spawn → route → goal (deployment-planner geometry)
    cmap = plt.get_cmap("tab10")
    for ep in es.episodes:
        c = cmap(ep.id % 10)
        planner.clear()  # each episode is a fresh FULL plan, like a real goal arrival
        route = planner.plan(RobotPose(0, ep.spawn[0], ep.spawn[1], 0.0),
                             GoalCoordinate(*ep.goal), world_map)
        if route:
            ax.plot([p[0] for p in route], [p[1] for p in route], "-",
                    color=c, linewidth=1.8, alpha=0.85, zorder=5)
        ax.plot(*ep.spawn, "o", color=c, markersize=11, markeredgecolor="white", zorder=6)
        ax.plot(*ep.goal, "*", color=c, markersize=17, markeredgecolor="white", zorder=6)
        ax.annotate(f"S{ep.id}", ep.spawn, textcoords="offset points", xytext=(7, 7),
                    fontsize=11, fontweight="bold", color=c)
        ax.annotate(f"G{ep.id}", ep.goal, textcoords="offset points", xytext=(7, 7),
                    fontsize=11, fontweight="bold", color=c)

    ax.set_title(f"locbench episodes — scene '{es.scene}', seed {es.seed}, "
                 f"{len(es.episodes)} episodes, {len(es.coverage_goals)} coverage goals")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _origin_from(es: EpisodeSet) -> Optional[tuple]:
    c = dict(es.constraints)
    if "origin_x" in c and "origin_y" in c:
        return (c["origin_x"], c["origin_y"])
    return None
