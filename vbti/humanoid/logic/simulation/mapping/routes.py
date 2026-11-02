"""routes.py — committed coverage routes + clearance validation (MAY-173 locdev T2).

A route is a YAML artifact (name, clearance_m, speed, loop, waypoints) committed
next to this module in `routes/`. `validate_route` walks every segment against
the inflated occupancy grid and reports blocked points — the render-approve-
freeze discipline the locbench episodes use: a committed route re-validates
after any scene/map change instead of silently driving into a new rack.

Pure numpy/stdlib (+PyYAML lazily, like occupancy_io's build-time bridge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from humanoid.logic.oli.reason.mapping.costmap import OccupancyGrid

Violation = Tuple[int, Tuple[float, float]]  # (segment index, blocked world point)


@dataclass(frozen=True)
class Route:
    name: str
    clearance_m: float
    speed: float
    loop: bool
    waypoints: List[Tuple[float, float]]


def load_route(path: Path | str) -> Route:
    """Load a committed route YAML."""
    import yaml  # lazy: keeps the brain's import path yaml-free

    raw = yaml.safe_load(Path(path).read_text())
    return Route(
        name=str(raw["name"]),
        clearance_m=float(raw["clearance_m"]),
        speed=float(raw.get("speed", 0.8)),
        loop=bool(raw.get("loop", False)),
        waypoints=[(float(x), float(y)) for x, y in raw["waypoints"]],
    )


def _segments(route: Route) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    pts = route.waypoints
    segs = list(zip(pts[:-1], pts[1:]))
    if route.loop and len(pts) > 1:
        segs.append((pts[-1], pts[0]))
    return segs


def plan_route(
    route: Route, grid: OccupancyGrid, spacing_m: float = 1.0
) -> List[Tuple[float, float]]:
    """Expand the route's sparse TARGETS into a dense drivable path with the
    DEPLOYMENT planner (`build_planner()` — clearance cost, weighted heuristic),
    then downsample to `spacing_m`. The drive then covers exactly the trajectory
    geometry Nav produces at eval time (factory.py's own preview rule).
    Raises ValueError when a target is unreachable."""
    from humanoid.logic.oli.reason.localization import RobotPose
    from humanoid.logic.oli.reason.mapping import StaticMapping
    from humanoid.logic.oli.reason.nav import GoalCoordinate, build_planner

    world_map = StaticMapping.from_grid(grid).latest()
    assert world_map is not None  # StaticMapping always has its one baked map
    planner = build_planner()
    targets = list(route.waypoints[1:])
    if route.loop:
        targets.append(route.waypoints[0])
    dense: List[Tuple[float, float]] = [route.waypoints[0]]
    cur = route.waypoints[0]
    for t in targets:
        planner.clear()  # each leg is a fresh FULL plan, like a real goal arrival
        leg = planner.plan(RobotPose(0, cur[0], cur[1], 0.0), GoalCoordinate(*t), world_map)
        if not leg:
            raise ValueError(f"route '{route.name}': target {t} unreachable from {cur}")
        dense.extend((float(px), float(py)) for px, py in leg)
        cur = t
    return _downsample(dense, spacing_m)


def _downsample(
    path: List[Tuple[float, float]], spacing_m: float
) -> List[Tuple[float, float]]:
    """Keep points ~spacing_m apart along the path; endpoints always survive and
    the final point replaces a too-close last keeper (no sub-spacing tail gap)."""
    kept = [path[0]]
    acc = 0.0
    for prev, pt in zip(path[:-1], path[1:]):
        acc += math.dist(prev, pt)
        if acc >= spacing_m:
            kept.append(pt)
            acc = 0.0
    if kept[-1] != path[-1]:
        if math.dist(kept[-1], path[-1]) < spacing_m / 2 and len(kept) > 1:
            kept[-1] = path[-1]
        else:
            kept.append(path[-1])
    return kept


def validate_route(
    route: Route, grid: OccupancyGrid, step_m: float = 0.1
) -> List[Violation]:
    """First blocked point per segment against the clearance-inflated grid;
    empty list = route is drivable."""
    inflated = grid.inflate(route.clearance_m)
    violations: List[Violation] = []
    for i, ((x0, y0), (x1, y1)) in enumerate(_segments(route)):
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(length / step_m)))
        for k in range(n + 1):
            t = k / n
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if inflated.is_occupied(x, y):
                violations.append((i, (x, y)))
                break
    return violations
