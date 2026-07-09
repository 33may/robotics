"""nav/planner.py — grid A* path planning on the inflated costmap (2D, 8-connected).

The global planner: given where the robot is and where the goal is (both map-frame world
coords), search the occupancy grid for a collision-free cell path and return it as world
waypoints. Run it on an **inflated** grid (`OccupancyGrid.inflate(robot_radius)`) so the
circle-footprint clearance is already baked into the free space and the search can treat the
robot as a point. This is the small, ownable slice of "Nav2" we hand-roll for the no-ROS PoC;
the pursuit controller (`controller.py`) turns the path into a velocity command.

Pure: numpy/stdlib only (the `brain` invariant).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from .costmap import OccupancyGrid

Cell = Tuple[int, int]
Point = Tuple[float, float]

# 8-connected steps with Euclidean move cost.
_STEPS: List[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),                          (0, 1, 1.0),
    (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2)),
]


def plan_path(grid: OccupancyGrid, start: Point, goal: Point) -> Optional[List[Point]]:
    """A* from `start` to `goal` (world coords); returns cell-center waypoints, or None.

    8-connected, Euclidean step cost + admissible Euclidean heuristic. Occupied cells are
    impassable — except the start cell, so the robot can always leave where it stands. Returns
    None if the goal is blocked, either endpoint is out of bounds, or the goal is unreachable.
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

    open_heap: List[Tuple[float, Cell]] = [(h(sr, sc), start_cell)]
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
            if tentative < gscore.get(nb, math.inf):
                came[nb] = cur
                gscore[nb] = tentative
                heapq.heappush(open_heap, (tentative + h(*nb), nb))
    return None


def _reconstruct(grid: OccupancyGrid, came: Dict[Cell, Cell], cur: Cell) -> List[Point]:
    cells: List[Cell] = [cur]
    while cur in came:
        cur = came[cur]
        cells.append(cur)
    cells.reverse()
    return [grid.cell_to_world(r, c) for r, c in cells]
