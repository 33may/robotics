#!/usr/bin/env python3
"""v1 collision safety for the exploration loop.

The wrist camera looks along its +z; the gripper fingers occupy roughly
70-130 mm ahead of the camera (measured from our own depth captures —
they are the permanent close-range blob). We approximate the arm's
business end with probe points in the camera frame and require them to
clear the table plane and the object's inflated AABB — at the target
configuration AND sampled along the joint-space path.

This is a stand-in for proper mesh collision (pinocchio+coal) — good
enough to stop the arm punching the table; not a guarantee.
"""

import numpy as np

from vbti.logic.servos.fk_so101 import fk
from vbti.logic.inspection.geometry import load_T_flange_cam

# Probe points, camera frame (m): finger volume + camera housing itself
PROBES_CAM = np.array([
    [0.0, 0.0, 0.0],      # camera housing
    [0.0, 0.0, 0.07],     # finger root
    [0.0, 0.03, 0.10],    # finger mid (fingers sit low in the image -> +y)
    [0.0, 0.03, 0.13],    # fingertip
    [0.0, -0.02, 0.07],   # upper jaw
])

TABLE_MARGIN_M = 0.02
OBJECT_MARGIN_M = 0.012   # true distance-to-AABB margin. NOT box inflation:
                          # the fingers must point at the object from ~r-13cm,
                          # a fat inflated-box test forbids all close viewing
PATH_SAMPLES = 12


def probe_points(joints_deg: dict) -> np.ndarray:
    """World positions of the probe points at a configuration."""
    T = fk(joints_deg) @ load_T_flange_cam()
    return PROBES_CAM @ T[:3, :3].T + T[:3, 3]


def config_is_safe(joints_deg: dict,
                   obj_aabb: tuple[np.ndarray, np.ndarray] | None) -> bool:
    """Probes must clear the table and the object's inflated AABB."""
    p = probe_points(joints_deg)
    if (p[:, 2] < TABLE_MARGIN_M).any():
        return False
    if obj_aabb is not None:
        # Euclidean distance from each probe to the AABB surface
        closest = np.clip(p, obj_aabb[0], obj_aabb[1])
        dist = np.linalg.norm(p - closest, axis=1)
        if (dist < OBJECT_MARGIN_M).any():
            return False
    return True


def path_is_safe(j_from: dict, j_to: dict,
                 obj_aabb: tuple[np.ndarray, np.ndarray] | None) -> bool:
    """Check joint-space interpolation samples between two configurations."""
    keys = [k for k in j_to if k in j_from]
    for t in np.linspace(0.0, 1.0, PATH_SAMPLES):
        j = dict(j_from)
        for k in keys:
            j[k] = j_from[k] + t * (j_to[k] - j_from[k])
        if not config_is_safe(j, obj_aabb):
            return False
    return True
