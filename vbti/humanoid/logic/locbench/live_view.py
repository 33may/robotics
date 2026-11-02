"""locbench/live_view.py — optional real-time run window (`locbench run --live-view`).

One matplotlib window over the occupancy map, updated as the evaluator collects streams:
the GT track (green, from the debug-pose-driven telemetry) vs the candidate's estimate
(red, from the shadow host), plus the unscored transit leg in gray and spawn/goal markers
per episode. Pure observability — the evaluator injects samples through `on_*` hooks and
every hook is exception-proof: a broken/closed window can NEVER affect a run's outcome or
its artifacts (the oracle stays the oracle).

Interactive backend note: `plots.py` pins Agg for headless artifact rendering; this module
switches the live figure to an interactive backend (TkAgg first, then Qt) at construction
and degrades to a no-op viewer (with a log line) when no GUI is available.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

from ..oli.reason.mapping import OccupancyGrid
from .episodes import Episode

_REDRAW_DT_S = 0.15         # ~7 Hz redraws — smooth enough, cheap enough
_BACKENDS = ("TkAgg", "QtAgg", "Qt5Agg")


class LiveView:
    """Live GT-vs-estimate window; construct via `try_open` (returns None headless)."""

    def __init__(self, grid: OccupancyGrid, plt, title: str) -> None:
        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=(7, 9))
        self._fig.canvas.manager.set_window_title(title)

        x0, y0 = grid.origin
        extent = (x0, x0 + grid.ncols * grid.resolution,
                  y0, y0 + grid.nrows * grid.resolution)
        self._ax.imshow(grid.grid, cmap="gray_r", origin="lower", extent=extent,
                        interpolation="nearest")
        self._ax.set_title(title)
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        (self._transit_line,) = self._ax.plot([], [], color="0.55", lw=1.0, ls=":",
                                              label="transit (unscored)")
        (self._gt_line,) = self._ax.plot([], [], color="tab:green", lw=1.8,
                                         label="ground truth")
        (self._est_line,) = self._ax.plot([], [], color="tab:red", lw=1.4,
                                          label="localization est")
        self._markers: List = []
        self._status = self._ax.text(0.02, 0.99, "", transform=self._ax.transAxes,
                                     va="top", ha="left", fontsize=9, family="monospace")
        self._ax.legend(loc="lower right", fontsize=8)

        self._transit: List[tuple] = []
        self._gt: List[tuple] = []
        self._est: List[tuple] = []
        self._episode: Optional[Episode] = None
        self._last_draw = 0.0
        self._dead = False

        plt.show(block=False)
        self._pump(force=True)

    # ── evaluator hooks (all exception-proof) ─────────────────────────────────────

    def on_episode(self, ep: Episode) -> None:
        try:
            self._episode = ep
            self._transit.clear()
            self._gt.clear()
            self._est.clear()
            for m in self._markers:
                m.remove()
            self._markers = [
                self._ax.plot(ep.spawn[0], ep.spawn[1], "o", color="tab:blue", ms=9)[0],
                self._ax.plot(ep.goal[0], ep.goal[1], "*", color="tab:blue", ms=15)[0],
            ]
            self._pump(force=True)
        except Exception:  # noqa: BLE001 — observability must never break the run
            self._dead = True

    def on_transit_gt(self, gt) -> None:
        try:
            self._transit.append((gt[1], gt[2]))
            self._pump()
        except Exception:  # noqa: BLE001
            self._dead = True

    def on_gt(self, gt) -> None:
        try:
            self._gt.append((gt[1], gt[2]))
            self._pump()
        except Exception:  # noqa: BLE001
            self._dead = True

    def on_est(self, est) -> None:
        try:
            pose = getattr(est, "pose", None)
            # LOST → nan gap: the red line visibly breaks where the candidate went silent
            self._est.append((pose.x, pose.y) if pose is not None else (math.nan, math.nan))
            self._pump()
        except Exception:  # noqa: BLE001
            self._dead = True

    def pump(self) -> None:
        """Keep the window responsive from any evaluator wait loop."""
        try:
            self._pump()
        except Exception:  # noqa: BLE001
            self._dead = True

    def close(self) -> None:
        try:
            self._plt.close(self._fig)
        except Exception:  # noqa: BLE001
            pass

    # ── internals ─────────────────────────────────────────────────────────────────

    def _pump(self, force: bool = False) -> None:
        if self._dead:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < _REDRAW_DT_S:
            return
        self._last_draw = now
        for line, pts in ((self._transit_line, self._transit),
                          (self._gt_line, self._gt), (self._est_line, self._est)):
            if pts:
                xs, ys = zip(*pts)
                line.set_data(xs, ys)
        ep = f"ep {self._episode.id}" if self._episode is not None else "—"
        err = ""
        if self._gt and self._est and not math.isnan(self._est[-1][0]):
            err = f"  pos err {math.dist(self._gt[-1], self._est[-1]):5.2f} m"
        self._status.set_text(f"{ep}  gt {len(self._gt):4d}  est {len(self._est):4d}{err}")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


def try_open(grid: OccupancyGrid, title: str,
             log: Callable[[str], None] = print) -> Optional[LiveView]:
    """Open the window on the first interactive backend that works; None when headless."""
    import matplotlib.pyplot as plt

    last = "no backends tried"
    for backend in _BACKENDS:
        try:
            plt.switch_backend(backend)
            return LiveView(grid, plt, title)
        except Exception as exc:  # noqa: BLE001 — try the next backend
            last = f"{backend}: {exc}"
    log(f"[locbench] --live-view: no interactive matplotlib backend ({last}) — "
        "running without the window")
    return None
