#!/usr/bin/env python3
"""Synthetic overhead D405 scene: flat table + one box. Shared by the
geometry seed test and the loop smoke test — no camera, no robot."""
import numpy as np

# width/height are part of the real intrinsics dict (`camera._intrinsics_dict`)
# and the shell-radius derivation reads them — a fake that omitted them made
# every synthetic capture fail deep inside the settle leg.
INTR = {"fx": 400.0, "fy": 400.0, "ppx": 424.0, "ppy": 240.0,
        "width": 848, "height": 480}
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


def synth_capture_touching_clutter():
    """Subject box with a bar lying ~9 mm alongside it, plus the subject mask.

    Models run 2408-cup2 view 003, the failure no geometric guard could catch:
    a cable passed behind the cup close enough to be single-linkage connected
    to it INSIDE a single view, so seeded growth had no grounds to refuse it
    and the 8 cm jump gate saw a point 20 mm from the cloud. The bar here sits
    inside GROW_EPS_M for exactly that reason — a test where the contaminant
    is far away would pass without testing anything.

    Returns (depth_u16, intr, depth_scale, T_base_cam, subject_mask).
    """
    depth = np.full((480, 848), 500, dtype=np.uint16)
    depth[200:280, 384:464] = 450          # subject, 50 mm proud, base x ~.255-.345
    depth[200:280, 472:600] = 455          # bar, 8 px gap ~= 9 mm, runs to x ~.50
    mask = np.zeros(depth.shape, dtype=bool)
    mask[200:280, 384:464] = True          # the subject alone
    return depth, INTR, DEPTH_SCALE, T_BASE_CAM, mask


def synth_capture_two_objects():
    """Same scene plus a BIGGER second box ~7 cm away in y.

    Models run 2408-geomtest: the inspected object stops being the biggest
    thing in frame (there, the mug went dark and a cable out-pointed it), so
    a largest-cluster rule elects the wrong one. The decoy is 120x120 px
    against the subject's 80x80, and the gap between them (~67 mm) is well
    over the growth threshold.

    Returns (depth_u16, intr, depth_scale, T_base_cam, subject_xy, decoy_xy).
    """
    depth = np.full((480, 848), 500, dtype=np.uint16)
    depth[200:280, 384:464] = 450                          # subject (smaller)
    depth[340:460, 384:504] = 450                          # decoy (bigger)
    return (depth, INTR, DEPTH_SCALE, T_BASE_CAM,
            np.array([0.30, 0.0]), np.array([0.32, -0.18]))
