#!/usr/bin/env python3
"""Synthetic overhead D405 scene: flat table + one box. Shared by the
geometry seed test and the loop smoke test — no camera, no robot."""
import numpy as np

INTR = {"fx": 400.0, "fy": 400.0, "ppx": 424.0, "ppy": 240.0}
DEPTH_SCALE = 0.001            # 1 mm per unit, like the D405
# camera 0.5 m above the table at x=0.30, looking straight down (cv frame)
T_BASE_CAM = np.array([[1.0, 0, 0, 0.30],
                       [0, -1.0, 0, 0.00],
                       [0, 0, -1.0, 0.50],
                       [0, 0, 0, 1.0]])


def synth_capture():
    """Returns (depth_u16, intr, depth_scale, T_base_cam, expected_centroid).

    Table plane fills the frame at 0.5 m; an 80x80 px box sits 5 cm proud
    at the principal point -> object centroid ~ (0.30, 0.00, ~0.025-0.05)
    in the base frame (top face at z=0.05).
    """
    depth = np.full((480, 848), 500, dtype=np.uint16)      # 0.5 m table
    depth[200:280, 384:464] = 450                          # box top at 0.45 m
    return depth, INTR, DEPTH_SCALE, T_BASE_CAM, np.array([0.30, 0.0, 0.05])
