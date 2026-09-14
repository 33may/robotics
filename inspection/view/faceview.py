"""Boxguard face-evidence views (design: docs/2026-09-14-face-evidence-design.md).

The boxguard cuboid already carries orthogonality: a face normal IS the
orthogonal view direction. This module turns a box Primitive into camera
targets — and (Task 2) resolves the closest ACCESSIBLE view when the exact
orthogonal pose cannot be reached.
"""
import numpy as np

from inspection.view.grid import radius_for_extent

#: Box-frame face handles, stable under yaw. Bottom is excluded — it faces
#: the table.
FACES = ("x+", "x-", "y+", "y-", "top")

#: Stability gate tolerance: max face-normal delta between consecutive
#: boxguard refits that still counts as "unchanged" (design §gate).
FACE_STABLE_TOL_DEG_DEFAULT = 5.0

#: (axis column, sign, indices of the two extents that span the face)
_FACE_DEF = {"x+": (0, +1.0, (1, 2)), "x-": (0, -1.0, (1, 2)),
             "y+": (1, +1.0, (0, 2)), "y-": (1, -1.0, (0, 2)),
             "top": (2, +1.0, (0, 1))}


def face_frame(prim, face):
    """Face center, outward unit normal, and (a, b) extents of that wall."""
    axis, sign, (i, j) = _FACE_DEF[face]
    ext = [float(v) for v in prim.params["extents"]]
    n = sign * prim.axes[:, axis]
    c = prim.center + n * (ext[axis] / 2.0)
    return c, n, (ext[i], ext[j])


def face_cam_pose(center, normal, r):
    """Camera 4x4 on the normal ray: +X boresight at the face, +Z image-up
    world-preferred — the same up-vector rule as ViewSphere.cam_pose."""
    normal = np.asarray(normal, dtype=float)
    pos = np.asarray(center, dtype=float) + r * normal
    x = -normal
    ref = np.array([0.0, 0.0, 1.0])
    if abs(x @ ref) > 0.95:                      # top face: up degenerate
        ref = np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


def face_radius(extent2d, fy, height_px):
    """Per-face framing standoff; R_MIN clamp lives in radius_for_extent."""
    return radius_for_extent(max(extent2d), fy, height_px)


def normals_delta_deg(prim_a, prim_b):
    """Max angular change of any face normal between two box fits [deg]."""
    worst = 0.0
    for face in FACES:
        _, na, _ = face_frame(prim_a, face)
        _, nb, _ = face_frame(prim_b, face)
        dot = float(np.clip(na @ nb, -1.0, 1.0))
        worst = max(worst, float(np.degrees(np.arccos(dot))))
    return worst
