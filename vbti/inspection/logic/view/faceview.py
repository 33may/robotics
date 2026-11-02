"""Boxguard face geometry (design: docs/blueprints/face-evidence.md).

The boxguard cuboid already carries orthogonality: a face normal IS the
orthogonal view direction. This module turns a box Primitive into the frames
that geometry needs — face centers, normals, wall extents, corners, the
camera pose on a normal ray, and the top view's yaw-aligned roll.

It no longer RESOLVES a view: the exact/z-lift/cone ladder died with Stage D.
`view/rays.py` builds the face RAY FAMILY (shift, then tilt) out of these
frames and picks the closest provably-safe sample on it
(docs/blueprints/ray-views.md).
"""
import numpy as np


def _rot_x(a):
    """Rotation about the camera's own +X (boresight) axis — replicated
    from `viewsphere._rot_x` rather than imported: that helper is
    underscore-private to its module."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


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


def face_cam_pose(center, normal, r, roll=0.0):
    """Camera 4x4 on the normal ray: +X boresight at the face, +Z image-up
    world-preferred — the same up-vector rule as ViewSphere.cam_pose,
    optionally rolled about the boresight (same convention/helper)."""
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
    T[:3, :3] = T[:3, :3] @ _rot_x(roll)
    return T


def face_corners(prim, face):
    """The four corners of `face`'s wall, base frame — the geometry a
    containment check projects, corner by corner, instead of bounding by
    an extent formula."""
    axis, sign, (i, j) = _FACE_DEF[face]
    c, _n, (ea, eb) = face_frame(prim, face)
    a1, a2 = prim.axes[:, i], prim.axes[:, j]
    return np.array([c + sa * (ea / 2.0) * a1 + sb * (eb / 2.0) * a2
                     for sa in (-1.0, 1.0) for sb in (-1.0, 1.0)])


def top_align_roll(prim):
    """Roll about the top view's boresight that lays the box footprint's
    WIDER side along the image WIDTH axis (Anton's ruling, 2026-09-21,
    demo7: the top capture framed the yawed box diagonally — the camera
    must rotate with the box, not with the world axes).

    Alignment is mod pi, so the 180-deg collision partner (the ROLLS walk)
    is equally aligned: walk (r, r + pi) wherever ROLLS would be walked.
    Only the top face gets this; a side view's roll is pinned by the
    horizon (image-up must stay world-up)."""
    center, normal, (ea, eb) = face_frame(prim, "top")
    wide = prim.axes[:, 0] if ea >= eb else prim.axes[:, 1]
    T0 = face_cam_pose(center, normal, 1.0, 0.0)
    x, y0, z0 = T0[:3, 0], T0[:3, 1], T0[:3, 2]
    w = wide - (wide @ x) * x              # project into the image plane
    return float(np.arctan2(w @ z0, w @ y0))


def normals_delta_deg(prim_a, prim_b):
    """Max angular change of any face normal between two box fits [deg]."""
    worst = 0.0
    for face in FACES:
        _, na, _ = face_frame(prim_a, face)
        _, nb, _ = face_frame(prim_b, face)
        dot = float(np.clip(na @ nb, -1.0, 1.0))
        worst = max(worst, float(np.degrees(np.arccos(dot))))
    return worst


#: Widest tilt the face ray family is allowed to reach, degrees — the
#: classic measurability cutoff. The ladder it used to cap is gone; this
#: survives because `run/app.py:face_method()` stamps it into the run's
#: recorded `view_methods` params, and `rays.CONE_MAX_DEG` (the live tilt
#: rung's own cap) holds the same value.
CONE_MAX_DEG = 60.0
