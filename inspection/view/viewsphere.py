#!/usr/bin/env python3
"""Viewsphere around the object centroid — D1 cells + reachability (D2).

v1 addressing is {h, v}: 12 azimuth bins of 30 deg (h=0 faces the robot
base, i.e. the direction from centroid back toward the base), 3 elevation
bins at ~20/45/70 deg. One shell, radius r. Roll and depth come later as
{h, v, r, d} coordinates.

Reachability: enumerate cells -> look-at IK -> reachable | blocked.
Blocked cells are invisible as actions, visible as facts (D2 asymmetry).
"""

import numpy as np

from inspection.ik import solve_lookat

H_BINS = 12                     # 30 deg each, h=0 toward robot base
V_ELEVATIONS = [20.0, 45.0, 70.0]   # deg above the table plane
DEFAULT_R_M = 0.20


def cell_direction(h: int, v: int) -> np.ndarray:
    """Unit vector from centroid toward the camera position of cell {h,v}.

    h=0 points from the centroid back toward the robot base (-x direction
    of the base frame, since the object sits at +x from the base).
    """
    az = np.radians(180.0 + h * (360.0 / H_BINS))   # base sits at -x from object
    el = np.radians(V_ELEVATIONS[v])
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])


def cell_cam_pose(centroid: np.ndarray, h: int, v: int,
                  r: float = DEFAULT_R_M) -> tuple[np.ndarray, np.ndarray]:
    """Cell -> (camera position, look direction toward centroid)."""
    d = cell_direction(h, v)
    pos = centroid + r * d
    return pos, -d


def enumerate_cells() -> list[dict]:
    return [{"h": h, "v": v} for v in range(len(V_ELEVATIONS))
            for h in range(H_BINS)]


def reachability(centroid: np.ndarray, r: float = DEFAULT_R_M,
                 q0_deg: dict | None = None) -> dict:
    """IK-filter every cell. Returns {(h,v): joints_deg | None}."""
    out = {}
    for c in enumerate_cells():
        pos, look = cell_cam_pose(centroid, c["h"], c["v"], r)
        out[(c["h"], c["v"])] = solve_lookat(pos, look, q0_deg=q0_deg)
    return out


def reachability_map_str(reach: dict) -> str:
    """ASCII map: rows = v (top row highest), cols = h. #=reachable .=blocked"""
    lines = []
    for v in reversed(range(len(V_ELEVATIONS))):
        row = "".join("#" if reach[(h, v)] is not None else "."
                      for h in range(H_BINS))
        lines.append(f"v{v} ({V_ELEVATIONS[v]:.0f}deg)  {row}")
    lines.append(f"            h={''.join(str(h % 10) for h in range(H_BINS))}"
                 "  (h0 faces base)")
    return "\n".join(lines)
