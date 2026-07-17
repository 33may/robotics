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
from collections import deque
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


def _goal_component(plan_grid: OccupancyGrid, goal_cell: Cell) -> set:
    """All plan-free cells 8-connected to the goal (flood fill)."""
    comp = {goal_cell}
    q: deque = deque([goal_cell])
    while q:
        r, c = q.popleft()
        for dr, dc, _ in _STEPS:
            nb: Cell = (r + dr, c + dc)
            if nb in comp or not plan_grid.in_bounds(*nb) or plan_grid.is_occupied_cell(*nb):
                continue
            comp.add(nb)
            q.append(nb)
    return comp


def _escape_to_free(
    world: OccupancyGrid, plan_grid: OccupancyGrid, start: Point, goal: Point
) -> Optional[List[Point]]:
    """Escape corridor for a start pocket sealed off by the robot layer
    (2026-07-17, Anton).

    A baked map can leave the spawn inside an inflation-locked pocket (the
    start/end camera blind spot). BFS from `start` over RAW-free world cells —
    physically observed/swept floor, never a real obstacle — until touching the
    goal's plan-free connected component. Returns the escape waypoints
    ([] when the start is already in the goal component), or None when no
    raw-free route to the goal component exists.
    """
    sr, sc = plan_grid.world_to_cell(*start)
    gr, gc = plan_grid.world_to_cell(*goal)
    if not plan_grid.in_bounds(sr, sc) or not plan_grid.in_bounds(gr, gc):
        return None
    if plan_grid.is_occupied_cell(gr, gc):
        return None
    comp = _goal_component(plan_grid, (gr, gc))
    if (sr, sc) in comp:
        return []
    if world.is_occupied_cell(sr, sc):
        return None
    came: Dict[Cell, Cell] = {}
    seen = {(sr, sc)}
    q: deque = deque([(sr, sc)])
    while q:
        cur = q.popleft()
        if cur in comp:
            cells = [cur]
            while cells[-1] in came:
                cells.append(came[cells[-1]])
            cells.reverse()
            return [plan_grid.cell_to_world(r, c) for r, c in cells]
        r, c = cur
        for dr, dc, _ in _STEPS:
            nb: Cell = (r + dr, c + dc)
            if nb in seen or not world.in_bounds(*nb) or world.is_occupied_cell(*nb):
                continue
            seen.add(nb)
            came[nb] = cur
            q.append(nb)
    return None


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
        # derived robot layer, cached keyed on (grid identity, version) (D9) — identity guards
        # against two mapping sources reusing the same version number for different grids
        self._layer_key: Optional[Tuple[int, int]] = None
        self._world_grid: Optional[OccupancyGrid] = None
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
        """(Re)derive the robot layer iff the world changed — keyed on (grid identity, version)
        (D9). A change also DROPS the cached path: the spliced tail was planned against the old
        world and cannot be trusted."""
        key = (id(world_map.grid), world_map.version)
        if key == self._layer_key:
            return
        self._world_grid = world_map.grid  # the emitted Map VALUE's grid (not a module ref)
        self._plan_grid = world_map.grid.inflate(self._robot_radius_m)
        self._cost = world_map.grid.clearance_cost(
            self._inflation_radius_m, self._clearance_weight
        )
        self._layer_key = key
        self._path = None

    def _full_plan(self, pose: RobotPose) -> Optional[List[Point]]:
        assert self._plan_grid is not None and self._goal is not None
        assert self._world_grid is not None
        start: Point = (pose.x, pose.y)
        goal: Point = (self._goal.x, self._goal.y)
        escape = _escape_to_free(self._world_grid, self._plan_grid, start, goal)
        if escape is None:
            return None  # no raw-free route to the goal's component
        if escape:
            start = escape[-1]  # plan from the escape exit, prepend the corridor
        path = plan_path(self._plan_grid, start, goal, cost=self._cost, weight=self._weight)
        if path is None:
            return None
        return escape[:-1] + path if escape else path
