"""deploybench/scenario.py — an interactive, per-run evaluation course.

Unlike locbench's frozen, seeded `EpisodeSet` (one committed oracle course for every
candidate, forever), a deploybench `Scenario` is defined ad hoc against whatever map a demo
baked: the operator picks a start and a goal on the map (`deploybench pick`), chooses KNOWN vs
KIDNAPPED start, and saves it. A scenario is reproducible (saved JSON) but is NOT a frozen
oracle — demos cover only part of a scene, so the eval course is chosen per run (FR1).

`validate` is the FR2 guard: start and goal must lie in the mapped free space (not off the
baked map, not inside an obstacle) with a route Nav can actually drive between them — checked
with the same `plan_path` A* the planner uses on the footprint-inflated grid. Pure
numpy/stdlib on the emitted `OccupancyGrid`; never imported by `logic/oli/`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from ..oli.reason.mapping import OccupancyGrid
from ..oli.reason.nav import plan_path

FORMAT_VERSION = 1

Point = Tuple[float, float]

# Robot footprint used to inflate the map before routing — matches the nav planner default
# (nav/AGENTS.md: footprint is a robot/policy knob, never baked into the world map).
DEFAULT_ROBOT_RADIUS_M = 0.30


class StartMode(str, Enum):
    """How the localizer is seeded at spawn (FR6 — GT is never an algorithm input except the
    operator's KNOWN hint)."""

    KNOWN = "known"          # operator "you are here": localizer gets an initial_pose hint
    KIDNAPPED = "kidnapped"  # no hint — the algorithm must self-localize (global reloc)


@dataclass(frozen=True)
class Scenario:
    """One operator-defined deploy course on a baked map.

    `start_yaw` is a KNOWN-start heading hint (rad, map frame); when None the run reads the
    live spawn heading. It is meaningless for KIDNAPPED and must stay None there.
    """

    name: str
    map_dir: str
    start: Point                              # (x, y) map frame
    goal: Point                               # (x, y) map frame
    start_mode: StartMode = StartMode.KNOWN
    start_yaw: Optional[float] = None         # KNOWN heading hint; None => read live at spawn
    arrival_tol_m: float = 0.3                # GT distance to the goal that counts as arrived
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", (float(self.start[0]), float(self.start[1])))
        object.__setattr__(self, "goal", (float(self.goal[0]), float(self.goal[1])))
        object.__setattr__(self, "start_mode", StartMode(self.start_mode))
        object.__setattr__(self, "arrival_tol_m", float(self.arrival_tol_m))
        object.__setattr__(self, "version", int(self.version))
        if self.start_yaw is not None:
            object.__setattr__(self, "start_yaw", float(self.start_yaw))
        if self.start_mode is StartMode.KIDNAPPED and self.start_yaw is not None:
            raise ValueError("KIDNAPPED start carries no pose hint — start_yaw must be None")


@dataclass(frozen=True)
class ScenarioCheck:
    """The FR2 verdict: is this a runnable course on this map, and how long is the route."""

    ok: bool
    route_m: Optional[float]           # planned A* length [m], or None if unroutable
    reasons: Tuple[str, ...]           # human-readable failures (empty when ok)


def validate(
    scenario: Scenario,
    grid: OccupancyGrid,
    *,
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M,
) -> ScenarioCheck:
    """Check start/goal are on the map, in free space, and joined by a drivable route.

    Routing runs on the footprint-inflated grid (what Nav drives), so a goal wedged against a
    rack fails as "no drivable route" even though its own cell is free.
    """
    reasons: List[str] = []
    for label, p in (("start", scenario.start), ("goal", scenario.goal)):
        row, col = grid.world_to_cell(*p)
        if not grid.in_bounds(row, col):
            reasons.append(f"{label} {p} is outside the mapped area")
        elif grid.is_occupied_cell(row, col):
            reasons.append(f"{label} {p} is inside an obstacle")

    route_m: Optional[float] = None
    if not reasons:  # only route once both endpoints are on-map and free
        reach = grid.inflate(robot_radius_m)
        route = plan_path(reach, scenario.start, scenario.goal)
        if route is None:
            reasons.append("no drivable route from start to goal on the inflated map")
        else:
            route_m = round(_route_length(route), 3)

    return ScenarioCheck(ok=not reasons, route_m=route_m, reasons=tuple(reasons))


def _route_length(path: List[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


# ── persistence ──────────────────────────────────────────────────────────────


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    doc = {
        "version": scenario.version,
        "name": scenario.name,
        "map_dir": scenario.map_dir,
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "start_mode": scenario.start_mode.value,
        "start_yaw": scenario.start_yaw,
        "arrival_tol_m": scenario.arrival_tol_m,
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def load_scenario(path: str | Path) -> Scenario:
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"scenario {path}: version {doc.get('version')!r} != {FORMAT_VERSION} — "
            "re-save it with this code"
        )
    return Scenario(
        name=doc["name"],
        map_dir=doc["map_dir"],
        start=tuple(doc["start"]),
        goal=tuple(doc["goal"]),
        start_mode=StartMode(doc["start_mode"]),
        start_yaw=doc.get("start_yaw"),
        arrival_tol_m=float(doc.get("arrival_tol_m", 0.3)),
        version=int(doc["version"]),
    )
