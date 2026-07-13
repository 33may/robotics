"""nav/planner.py — the planning module: pure grid A* + the stateful `Planner` over it.

`plan_path` is the pure global search: given a grid and two world points, return collision-free
world waypoints (8-connected A*, optional soft clearance cost, optional weighted heuristic).

`Planner` is the module the orchestrator calls (design.md D8, D9). It consumes the EMITTED
`Map` value — `plan(pose, goal, world_map)` — never a mapping-module ref (modules consume
contracts, not each other; bus/OOP-ready). It privately owns its derivations:

  - the ROBOT layer (hard footprint inflate + soft clearance gradient — robot/policy knobs,
    never baked into the world map), cached keyed on `world_map.version`;
  - the path cache + goal-change detection: new goal → full A*; same goal → cheap LOCAL
    re-plan of the near horizon with the far tail spliced on; local failure → full fallback.

A `version` bump rebuilds the robot layer AND drops the cached path (a changed world
invalidates the spliced tail). With the static map the version never bumps, so derivations
happen exactly once — the pre-split behavior and cost, preserved by construction.

Pure: numpy/stdlib only (the `brain` invariant).
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..localization import RobotPose
from ..mapping import Map, OccupancyGrid
from .types import GoalCoordinate

if TYPE_CHECKING:
    import numpy as np

Cell = Tuple[int, int]
Point = Tuple[float, float]

# 8-connected steps with Euclidean move cost.
_STEPS: List[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),                          (0, 1, 1.0),
    (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2)),
]


def plan_path(
    grid: OccupancyGrid,
    start: Point,
    goal: Point,
    cost: Optional["np.ndarray"] = None,
    weight: float = 1.0,
) -> Optional[List[Point]]:
    """A* from `start` to `goal` (world coords); returns cell-center waypoints, or None.

    8-connected, Euclidean step cost + admissible Euclidean heuristic. Occupied cells are
    impassable — except the start cell, so the robot can always leave where it stands. Returns
    None if the goal is blocked, either endpoint is out of bounds, or the goal is unreachable.

    `cost` (optional, grid-shaped ≥0) is a soft per-cell penalty added when *entering* a cell —
    e.g. `OccupancyGrid.clearance_cost` to prefer clearance from walls. Being ≥0 it keeps the
    Euclidean heuristic admissible (true cost ≥ geometric distance), so A* stays optimal.

    `weight` >1 = weighted A*: inflate the heuristic so the search expands far fewer nodes (big
    speedup when `cost` flattens the heuristic), at the price of ε-suboptimality — path cost ≤
    `weight` × optimal. 1.0 = exact. Small values (≈1.2) are near-lossless in practice.
    """
    sr, sc = grid.world_to_cell(*start)
    gr, gc = grid.world_to_cell(*goal)
    start_cell: Cell = (sr, sc)
    goal_cell: Cell = (gr, gc)
    if not grid.in_bounds(sr, sc) or not grid.in_bounds(gr, gc):
        return None
    if grid.is_occupied_cell(gr, gc):
        return None

    def h(r: int, c: int) -> float:
        return math.hypot(r - gr, c - gc)

    open_heap: List[Tuple[float, Cell]] = [(weight * h(sr, sc), start_cell)]
    came: Dict[Cell, Cell] = {}
    gscore: Dict[Cell, float] = {start_cell: 0.0}
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal_cell:
            return _reconstruct(grid, came, cur)
        if cur in closed:
            continue
        closed.add(cur)
        r, c = cur
        for dr, dc, step in _STEPS:
            nb: Cell = (r + dr, c + dc)
            if not grid.in_bounds(*nb):
                continue
            if nb != start_cell and grid.is_occupied_cell(*nb):
                continue
            tentative = gscore[cur] + step
            if cost is not None:
                tentative += float(cost[nb[0], nb[1]])   # soft penalty for entering nb
            if tentative < gscore.get(nb, math.inf):
                came[nb] = cur
                gscore[nb] = tentative
                heapq.heappush(open_heap, (tentative + weight * h(*nb), nb))
    return None


def _reconstruct(grid: OccupancyGrid, came: Dict[Cell, Cell], cur: Cell) -> List[Point]:
    cells: List[Cell] = [cur]
    while cur in came:
        cur = came[cur]
        cells.append(cur)
    cells.reverse()
    return [grid.cell_to_world(r, c) for r, c in cells]


def _split_ahead(pose: RobotPose, path: List[Point], horizon_m: float) -> Tuple[Point, List[Point]]:
    """Walk `path` from the waypoint nearest the robot until `horizon_m` accumulated; return
    (that horizon waypoint, the remaining tail after it). The local re-plan targets the horizon;
    the tail is spliced on unchanged."""
    i0 = min(range(len(path)), key=lambda i: (path[i][0] - pose.x) ** 2 + (path[i][1] - pose.y) ** 2)
    acc, ih = 0.0, i0
    for i in range(i0, len(path) - 1):
        acc += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        ih = i + 1
        if acc >= horizon_m:
            break
    return path[ih], path[ih + 1:]


class Planner:
    """Stateful planning module over the pure `plan_path` (see module docstring).

    Construction takes the robot/policy knobs only; the world arrives per call as the emitted
    `Map` value. Defaults (0.0 radii/weights) are a no-op point planner, `heuristic_weight`>1 =
    weighted A* (path ≤ weight×optimal), `horizon_m` bounds each local re-plan so compute stays
    flat regardless of goal distance.
    """

    def __init__(
        self,
        *,
        robot_radius_m: float = 0.0,
        inflation_radius_m: float = 0.0,
        clearance_weight: float = 0.0,
        heuristic_weight: float = 1.0,
        horizon_m: float = 2.0,
    ) -> None:
        self._robot_radius_m = robot_radius_m
        self._inflation_radius_m = inflation_radius_m
        self._clearance_weight = clearance_weight
        self._weight = heuristic_weight
        self._horizon_m = horizon_m
        # derived robot layer, cached keyed on Map.version (D9)
        self._layer_version: Optional[int] = None
        self._plan_grid: Optional[OccupancyGrid] = None
        self._cost: Optional["np.ndarray"] = None
        # path cache + goal-change detection (D9)
        self._goal: Optional[GoalCoordinate] = None
        self._path: Optional[List[Point]] = None

    @property
    def path(self) -> Optional[List[Point]]:
        """The most recently planned path (world waypoints), or None — for observers to render."""
        return self._path

    def clear(self) -> None:
        """Forget the cached goal + path (the goal changed or was cleared) — next plan is FULL.
        The derived robot layer survives: it depends on the map version, not the goal."""
        self._goal = None
        self._path = None

    def plan(
        self, pose: RobotPose, goal: Optional[GoalCoordinate], world_map: Map
    ) -> Optional[List[Point]]:
        """Plan from `pose` to `goal` on the given world snapshot; cache + return the path.

        New goal → FULL plan. Same goal → cheap LOCAL re-plan (near `horizon_m` re-solved, far
        tail spliced), full fallback if the local solve fails (robot drifted off the path). A
        `world_map.version` bump rebuilds the robot layer and forces a full plan. None when
        there is no goal or it is blocked/unreachable.
        """
        if goal is None:
            self._goal = None
            self._path = None
            return None
        self._refresh_robot_layer(world_map)
        if goal != self._goal:                    # new goal → next plan is full
            self._goal = goal
            self._path = None
        assert self._plan_grid is not None
        if not self._path:
            self._path = self._full_plan(pose)
            return self._path
        horizon, tail = _split_ahead(pose, self._path, self._horizon_m)
        local = plan_path(
            self._plan_grid, (pose.x, pose.y), horizon, cost=self._cost, weight=self._weight
        )
        self._path = (local + tail) if local else self._full_plan(pose)
        return self._path

    def _refresh_robot_layer(self, world_map: Map) -> None:
        """(Re)derive the robot layer iff the world changed — keyed on `Map.version` (D9).

        A bump also DROPS the cached path: the spliced tail was planned against the old world
        and cannot be trusted."""
        if world_map.version == self._layer_version:
            return
        self._plan_grid = world_map.grid.inflate(self._robot_radius_m)
        self._cost = world_map.grid.clearance_cost(
            self._inflation_radius_m, self._clearance_weight
        )
        self._layer_version = world_map.version
        self._path = None

    def _full_plan(self, pose: RobotPose) -> Optional[List[Point]]:
        assert self._plan_grid is not None and self._goal is not None
        return plan_path(
            self._plan_grid, (pose.x, pose.y), (self._goal.x, self._goal.y),
            cost=self._cost, weight=self._weight,
        )
