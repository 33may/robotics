"""Cell frame around the inspected object — {h, v} directions, camera poses.

Keeps the addressing scheme (Anton, 2026-08-18): cells are {h, v} — 12
azimuth bins of 30 deg with h=0 facing the robot base, elevations from
`grid.V_ELEVATIONS`, azimuth computed from the actual center.

A cell's DISTANCE is no longer a property of this helper. A view is a ray
and its depth is the closest provably-safe sample on it (`view/rays.py`,
docs/blueprints/ray-views.md) — the reachability map, the roll
search and the shell-radius derivation died with Stage D. `r` here is only
the radius the cell frame is DRAWN at: the UI's cell markers and the
nominal `cam_pose` behind them. `run/machine.py:_recenter` feeds it the
mean depth the rays actually resolved, falling back to `grid.DEFAULT_R`
before anything has resolved.

`T_FLANGE_CAM` and `ROLLS` live here and are imported by `view/rays.py`,
`view/faceview.py`, `run/collect.py` and `run/machine.py`.
"""

import numpy as np

from inspection.cell.geometry import load_T_flange_cam
# Cell addressing lives in the light half (grid.py) so the AI tiers can speak
# about cells without importing IK/motion.
from inspection.view.grid import H_BINS, V_ELEVATIONS, DEFAULT_R

# CALIBRATED hand-eye (2026-08-19, calib/T_flange_cam_*.npy): left eye at
# flange [+142.4, -8.3, -47.5] mm, boresight 1.07 deg off the tool axis.
# Loaded in this module's camera convention (+X boresight, +Z image-up).
T_FLANGE_CAM = load_T_flange_cam("xfwd")

# Roll preference: upright first, then a half turn — and NOTHING else
# (Anton 2026-08-24). Both are LOSSLESS to undo in image space and both keep
# the frame landscape, so every capture can be presented to the VLM in one
# orientation. Measured on the demo cup at r=0.244: {0, 180} reaches 26/36
# cells — exactly what the old twelve-roll ladder reached. The nine
# intermediate rolls bought no reachability at all, they only produced
# tilted images that no de-rotation can fix without interpolating.
ROLLS = np.deg2rad([0, 180])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class ViewSphere:
    """The {h, v} cell frame around a center point (base frame).

    `r` is a DRAWING radius, not a standoff: cell depth is resolved per ray
    by `view/rays.py`.
    """

    def __init__(self, center, r=DEFAULT_R, elevations=V_ELEVATIONS):
        self.center = np.asarray(center, dtype=float)
        self.r = float(r)
        self.elevations = tuple(elevations)   # deg ABOVE THE TABLE PLANE
        # h=0 faces the base: azimuth of the center->base direction
        d = -self.center[:2]
        self._az0 = np.arctan2(d[1], d[0])

    def cells(self):
        return [(h, v) for v in range(len(self.elevations)) for h in range(H_BINS)]

    def cell_dir(self, h, v):
        """Unit vector center -> camera position of cell {h, v}."""
        az = self._az0 + h * (2 * np.pi / H_BINS)
        el = np.deg2rad(self.elevations[v])
        return np.array([np.cos(el) * np.cos(az),
                         np.cos(el) * np.sin(az),
                         np.sin(el)])

    def cam_pose(self, h, v, roll=0.0):
        """Camera pose 4x4: +X boresight at the center, +Z image-up (world-up
        preferred, radial fallback near the pole), rolled about the boresight."""
        out = self.cell_dir(h, v)
        pos = self.center + self.r * out
        x = -out                                        # boresight
        ref = np.array([0.0, 0.0, 1.0])
        if abs(x @ ref) > 0.95:                         # top-down: up degenerate
            ref = np.array([np.cos(self._az0), np.sin(self._az0), 0.0])
        y = np.cross(ref, x); y /= np.linalg.norm(y)    # image-right
        z = np.cross(x, y)                              # image-up
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
        T[:3, :3] = T[:3, :3] @ _rot_x(roll)
        return T
