"""mapping/costmap.py — the 2D occupancy grid the planner searches (SE(2) footprint nav).

The map is baked once from the scene geometry (`--scene` USD) into a boolean occupancy grid;
obstacles include everything from shelves to a box on the floor — anything within the robot's
height column projects to an occupied cell (2.5D-to-classify, 2D-to-plan). The robot is treated
as a **circle footprint**: rather than test the polygon each step, obstacles are dilated by the
robot radius (`inflate`), so the planner can search for a single free *point* through the
inflated free space. A future oriented-polygon footprint replaces `inflate` without touching the
planner.

Pure: numpy + stdlib only (the `brain` invariant).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class OccupancyGrid:
    """A 2D occupancy grid in the map frame: True = occupied/blocked, False = free.

    `resolution` is meters per cell; `origin` is the world (x, y) of the grid's (row=0, col=0)
    cell corner. Indexed `[row, col]` with col along +x and row along +y (right-handed, ROS-style
    map). Out-of-bounds queries are treated as occupied — the robot cannot plan outside the known
    map.
    """

    def __init__(
        self,
        occupied: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        grid = np.asarray(occupied, dtype=bool)
        if grid.ndim != 2:
            raise ValueError(f"occupancy grid must be 2D, got shape {grid.shape}")
        self.grid = grid
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

    @property
    def nrows(self) -> int:
        return self.grid.shape[0]

    @property
    def ncols(self) -> int:
        return self.grid.shape[1]

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        return (row, col)

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """World coords of the *center* of cell (row, col)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return (x, y)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.nrows and 0 <= col < self.ncols

    def is_occupied_cell(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True  # outside the known map = blocked
        return bool(self.grid[row, col])

    def is_occupied(self, x: float, y: float) -> bool:
        row, col = self.world_to_cell(x, y)
        return self.is_occupied_cell(row, col)

    def inflate(self, radius_m: float) -> "OccupancyGrid":
        """Return a NEW grid with obstacles dilated by `radius_m` (circle-footprint inflation).

        Non-destructive: the original grid is unchanged. Any cell whose center lies within
        `radius_m` of an occupied cell center becomes occupied, so the planner can treat the robot
        as a point moving through the inflated free space.
        """
        out = self.grid.copy()
        r_cells = int(math.ceil(radius_m / self.resolution))
        if r_cells <= 0:
            return OccupancyGrid(out, self.resolution, self.origin)

        # Disk structuring element: offsets whose metric distance is within radius.
        res2, rad2 = self.resolution ** 2, radius_m ** 2
        offsets = [
            (dr, dc)
            for dr in range(-r_cells, r_cells + 1)
            for dc in range(-r_cells, r_cells + 1)
            if (dr * dr + dc * dc) * res2 <= rad2
        ]
        occ_rows, occ_cols = np.nonzero(self.grid)
        for r0, c0 in zip(occ_rows.tolist(), occ_cols.tolist()):
            for dr, dc in offsets:
                r, c = r0 + dr, c0 + dc
                if 0 <= r < self.nrows and 0 <= c < self.ncols:
                    out[r, c] = True
        return OccupancyGrid(out, self.resolution, self.origin)

    def clearance_cost(self, inflation_radius_m: float, weight: float) -> np.ndarray:
        """Soft per-cell penalty that decays with distance to the nearest obstacle.

        Returns a float array (grid shape) added to A* step cost (`plan_path(..., cost=)`) so the
        planner prefers open space but still hugs a wall when that's the only route. The penalty
        ramps linearly from `weight` at the obstacle edge to 0 at `inflation_radius_m` (and stays 0
        beyond) — the soft companion to the hard `inflate` (impassable) boundary. Distance is a
        Euclidean distance transform on the RAW obstacles, so set `inflation_radius_m` larger than
        the hard footprint radius for the gradient to bite across the passable band.

        This is the robot/policy layer (footprint clearance), kept out of the map bake so one baked
        occupancy artifact serves any footprint/tuning and the knobs stay live-tunable.
        """
        if inflation_radius_m <= 0.0 or weight == 0.0 or not self.grid.any():
            return np.zeros(self.grid.shape, dtype=float)
        from scipy import ndimage  # lazy: keep costmap import light, scipy only when planning
        # EDT gives, for each FREE cell, the distance (in cells) to the nearest occupied cell.
        dist_m = ndimage.distance_transform_edt(~self.grid) * self.resolution
        ramp = np.clip(1.0 - dist_m / inflation_radius_m, 0.0, 1.0)
        return (weight * ramp).astype(float)
