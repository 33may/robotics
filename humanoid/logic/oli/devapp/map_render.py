"""devapp/map_render.py — pure rasterization of the nav map for the MapPanel.

Composites the `OccupancyGrid` + live pose / path / goal into a NORTH-UP RGB image (max-y at top,
occupied dark, free light) for display. Pure numpy — no GL/ImGui — so the geometry (world→pixel,
overlays) is unit-testable without a window; `map_panel.py` just blits the result via ImmVision.
"""

from __future__ import annotations

import math

import numpy as np

_OCC = np.array([40, 40, 48], np.uint8)     # occupied → dark slate
_FREE = np.array([235, 235, 235], np.uint8)  # free → light grey
_ROBOT = np.array([230, 30, 30], np.uint8)   # robot → red
_PATH = np.array([70, 130, 255], np.uint8)   # planned path → blue
_GOAL = np.array([0, 175, 0], np.uint8)      # goal → green


def base_rgb(grid) -> np.ndarray:
    """`OccupancyGrid` → (H, W, 3) uint8, NORTH-UP: occupied dark, free light.

    Grid row 0 is min-y (origin corner); flip vertically so max-y renders at the image top.
    """
    occ = np.flipud(grid.grid)
    img = np.where(occ[..., None], _OCC, _FREE)
    return np.ascontiguousarray(img.astype(np.uint8))


def world_to_pixel(grid, x: float, y: float):
    """World (x, y) → (row, col) in the NORTH-UP image (max-y at top)."""
    r, c = grid.world_to_cell(x, y)
    return grid.nrows - 1 - r, c


def pixel_to_world(grid, row: float, col: float):
    """Inverse of `world_to_pixel`: NORTH-UP image pixel (row, col) → world (x, y) at that
    cell's centre. `row`/`col` may be floats (a click inside a pixel); they floor to a cell."""
    r = grid.nrows - 1 - int(math.floor(row))
    c = int(math.floor(col))
    return grid.cell_to_world(r, c)


def _marker(img: np.ndarray, row: int, col: int, color: np.ndarray, size: int = 4) -> None:
    h, w = img.shape[:2]
    r0, r1 = max(0, row - size), min(h, row + size + 1)
    c0, c1 = max(0, col - size), min(w, col + size + 1)
    if r1 > r0 and c1 > c0:
        img[r0:r1, c0:c1] = color


def compose(grid, base: np.ndarray, pose=None, path=None, goal=None) -> np.ndarray:
    """Copy `base` and overlay path (blue), goal (green), robot (red + heading). NORTH-UP RGB.

    Non-destructive: `base` (the static map raster) is never mutated.
    """
    img = base.copy()
    if path:
        for x, y in path:
            r, c = world_to_pixel(grid, x, y)
            _marker(img, r, c, _PATH, size=1)
    if goal is not None:
        gr, gc = world_to_pixel(grid, goal[0], goal[1])
        _marker(img, gr, gc, _GOAL, size=5)
    if pose is not None:
        r, c = world_to_pixel(grid, pose.x, pose.y)
        dr, dc = -math.sin(pose.yaw), math.cos(pose.yaw)  # +x→+col, +y→−row (north-up)
        for t in range(4, 20):
            _marker(img, int(round(r + dr * t)), int(round(c + dc * t)), _ROBOT, size=1)
        _marker(img, r, c, _ROBOT, size=5)
    return img
