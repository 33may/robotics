"""deploybench/pick.py — the interactive "define start+goal on the map" picker (FR1).

Standalone matplotlib: render the baked occupancy map, click START then GOAL, toggle
KNOWN/KIDNAPPED, watch live FR2 validation + the planned route, and save a `Scenario` JSON.

`build_scenario` (the composition + validation logic) is pure and brain-tested; matplotlib is
imported lazily inside `ScenarioPicker.run()`, so importing this module does NOT pull a GUI
stack into the brain/robot boot (mirrors locbench's plots/runner split).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

from ..oli.reason.mapping import OccupancyGrid
from .scenario import (
    DEFAULT_ROBOT_RADIUS_M,
    Scenario,
    ScenarioCheck,
    StartMode,
    save_scenario,
    validate,
)

Point = Tuple[float, float]


def build_scenario(
    *,
    name: str,
    map_dir: str | Path,
    grid: OccupancyGrid,
    start_xy: Point,
    goal_xy: Point,
    start_mode: StartMode | str,
    start_yaw: Optional[float] = None,
    arrival_tol_m: float = 0.3,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
) -> Tuple[Scenario, ScenarioCheck]:
    """Compose picked points into a validated `Scenario`.

    KNOWN start defaults its heading to "face the goal" (the deterministic warm-start heading,
    matching locbench's transit convention) unless an explicit `start_yaw` is given. KIDNAPPED
    carries no hint. Returns the scenario AND its FR2 check so the caller can show why a pick
    is (not) runnable without re-validating.
    """
    mode = StartMode(start_mode)
    yaw: Optional[float] = None
    if mode is StartMode.KNOWN:
        yaw = (start_yaw if start_yaw is not None
               else math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]))
    scenario = Scenario(
        name=name, map_dir=str(map_dir),
        start=(float(start_xy[0]), float(start_xy[1])),
        goal=(float(goal_xy[0]), float(goal_xy[1])),
        start_mode=mode, start_yaw=yaw, arrival_tol_m=arrival_tol_m,
    )
    check = validate(scenario, grid, robot_radius_m=robot_radius_m)
    return scenario, check


class ScenarioPicker:
    """Interactive matplotlib picker. Not unit-tested (GUI event loop); its logic lives in the
    pure `build_scenario`. Usage: `ScenarioPicker(grid, map_dir).run(save_path)`."""

    def __init__(
        self,
        grid: OccupancyGrid,
        map_dir: str | Path,
        *,
        name: str = "scenario",
        robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
        arrival_tol_m: float = 0.3,
    ) -> None:
        self.grid = grid
        self.map_dir = str(map_dir)
        self.name = name
        self.robot_radius_m = robot_radius_m
        self.arrival_tol_m = arrival_tol_m
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.mode = StartMode.KNOWN
        self.scenario: Optional[Scenario] = None

    def run(self, save_path: Optional[str | Path] = None) -> Optional[Scenario]:
        import matplotlib.pyplot as plt  # lazy — keep GUI out of brain imports

        from ..oli.reason.nav import plan_path

        g = self.grid
        x0, y0 = g.origin
        x1, y1 = x0 + g.ncols * g.resolution, y0 + g.nrows * g.resolution

        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(g.grid, origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", alpha=0.9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                       fontsize=9, family="monospace",
                       bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        artists: list = []

        def clear_artists() -> None:
            while artists:
                artists.pop().remove()

        def redraw() -> None:
            clear_artists()
            reasons: Tuple[str, ...] = ()
            route_m = None
            if self.start is not None:
                artists.append(ax.plot(*self.start, "o", color="tab:green", ms=12,
                                       label="start")[0])
            if self.goal is not None:
                artists.append(ax.plot(*self.goal, "*", color="tab:red", ms=18,
                                       label="goal")[0])
            if self.start is not None and self.goal is not None:
                scn, chk = build_scenario(
                    name=self.name, map_dir=self.map_dir, grid=g,
                    start_xy=self.start, goal_xy=self.goal, start_mode=self.mode,
                    arrival_tol_m=self.arrival_tol_m, robot_radius_m=self.robot_radius_m)
                reasons, route_m = chk.reasons, chk.route_m
                self.scenario = scn if chk.ok else None
                if chk.ok:
                    route = plan_path(g.inflate(self.robot_radius_m), self.start, self.goal)
                    if route:
                        xs, ys = zip(*route)
                        artists.append(ax.plot(xs, ys, "-", color="tab:blue", lw=2, alpha=0.7)[0])
            status = ("READY — press ENTER to save"
                      if self.scenario is not None else
                      ("pick start, then goal" if self.goal is None else "INVALID"))
            lines = [
                f"scenario: {self.name}   mode: {self.mode.value.upper()}",
                f"start: {_fmt(self.start)}   goal: {_fmt(self.goal)}",
                f"route: {route_m if route_m is not None else '—'} m   {status}",
                *[f"  ✗ {r}" for r in reasons],
                "[click] start→goal   [k] known/kidnapped   [r] reset   [enter] save",
            ]
            info.set_text("\n".join(lines))
            ax.set_title(f"deploybench pick — {self.name}")
            fig.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not ax or event.xdata is None:
                return
            p = (float(event.xdata), float(event.ydata))
            if self.start is None or self.goal is not None:
                self.start, self.goal, self.scenario = p, None, None  # restart
            else:
                self.goal = p
            redraw()

        def on_key(event) -> None:
            if event.key == "k":
                self.mode = (StartMode.KIDNAPPED if self.mode is StartMode.KNOWN
                             else StartMode.KNOWN)
            elif event.key == "r":
                self.start = self.goal = self.scenario = None
            elif event.key in ("enter", "return") and self.scenario is not None:
                if save_path is not None:
                    save_scenario(self.scenario, save_path)
                    print(f"[deploybench] saved scenario → {save_path}")
                plt.close(fig)
                return
            redraw()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
        return self.scenario


def _fmt(p: Optional[Point]) -> str:
    return "—" if p is None else f"({p[0]:.2f}, {p[1]:.2f})"
