"""Boxguard face-evidence views (design: docs/2026-09-14-face-evidence-design.md).

The boxguard cuboid already carries orthogonality: a face normal IS the
orthogonal view direction. This module turns a box Primitive into camera
targets — and (Task 2) resolves the closest ACCESSIBLE view when the exact
orthogonal pose cannot be reached.
"""
from collections import namedtuple

import numpy as np

from inspection.view.grid import radius_for_extent
# ROLLS is mandated by ruling (2026-09-14): every ladder candidate must walk
# camera rolls before being declared invalid — the tool hull is a cross with
# an off-axis camera slab, and the 180 deg flip clears collisions the
# upright roll cannot (ViewSphere.cell_reachable/plan_to_cell do the same
# walk). This does pull the heavy IK/motion/pinocchio chain in through
# viewsphere.py at import time — an acknowledged cost of the ruling, not an
# oversight (see task-2-report.md fix-report for the note to the controller).
from inspection.view.viewsphere import ROLLS


def _rot_x(a):
    """Rotation about the camera's own +X (boresight) axis — replicated
    from viewsphere.py's `_rot_x` (viewsphere.py:54) rather than imported:
    that helper is underscore-private to its module, and ROLLS is already
    the one symbol this file borrows from viewsphere."""
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


#: Resolution ladder defaults (design §Request-time resolution ladder).
ZLIFT_STEP_M = 0.02          # rung 2 raise increment
ZLIFT_MAX_DEG = 45.0         # rung 2 hands over to the cone past this
CONE_STEP_DEG = 5.0          # rung 3 ring spacing
CONE_MAX_DEG = 60.0          # classic measurability cutoff
CONE_AZIMUTHS = 12

FaceResolution = namedtuple("FaceResolution",
                            ["T_cam", "T_flange", "theta_deg", "rung", "roll"])


def _endpoint_ok(world, ik, center, direction, r, t_fc_inv, q_near):
    """First (roll-order) free endpoint at this candidate direction/radius.

    Walks ROLLS = [0, 180 deg] before declaring the candidate invalid — the
    tool hull is a cross with an off-axis camera slab, so a 180 deg flip
    often clears a collision the upright roll cannot (the same walk
    ViewSphere.cell_reachable does). Returns (T_cam, T_flange, q, roll) or
    None.
    """
    for roll in ROLLS:
        T_cam = face_cam_pose(center, direction, r, roll)
        T_fl = T_cam @ t_fc_inv
        free = [q for q in ik.branches(T_fl, q_near=q_near)
                if not world.is_colliding(q)]
        if free:
            return T_cam, T_fl, free[0], roll
    return None


def resolve_face_view(world, ik, t_flange_cam, prim, face, r, q_near=None):
    """Closest accessible view of `face`, most-orthogonal-first.

    Rungs (each the cheapest fix for its own failure mode):
      exact  — the orthogonal pose itself.
      zlift  — slim/short objects put a side camera at table height; rise
               along +Z looking at the face center until valid or 45° off.
      cone   — tilt rings around the normal, 5°..60°, 12 azimuths; the
               first valid endpoint is the most orthogonal accessible.
    Every candidate endpoint is tried at each camera roll (see
    `_endpoint_ok`) before being rejected. Returns None when nothing within
    60° has a collision-free IK branch at any roll — the face is honestly
    unreachable.
    """
    t_fc_inv = np.linalg.inv(np.asarray(t_flange_cam, dtype=float))
    center, normal, _ = face_frame(prim, face)

    hit = _endpoint_ok(world, ik, center, normal, r, t_fc_inv, q_near)  # rung 1
    if hit:
        T_cam, T_fl, q, roll = hit
        return FaceResolution(T_cam, T_fl, 0.0, "exact", roll)

    if abs(normal[2]) < 0.5:                                   # rung 2
        base = center + r * normal
        z_cap = r * np.tan(np.deg2rad(ZLIFT_MAX_DEG))
        dz = ZLIFT_STEP_M
        while dz <= z_cap:
            pos = base + np.array([0.0, 0.0, dz])
            d = pos - center
            dn = d / np.linalg.norm(d)
            theta = float(np.degrees(np.arccos(
                np.clip(dn @ normal, -1.0, 1.0))))
            hit = _endpoint_ok(world, ik, center, dn,
                               float(np.linalg.norm(d)), t_fc_inv, q_near)
            if hit:
                T_cam, T_fl, q, roll = hit
                return FaceResolution(T_cam, T_fl, theta, "zlift", roll)
            dz += ZLIFT_STEP_M

    a = np.array([0.0, 0.0, 1.0])                              # rung 3
    if abs(normal @ a) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = CONE_STEP_DEG
    while theta <= CONE_MAX_DEG:
        th = np.deg2rad(theta)
        for k in range(CONE_AZIMUTHS):
            phi = 2 * np.pi * k / CONE_AZIMUTHS
            d = (np.cos(th) * normal
                 + np.sin(th) * (np.cos(phi) * u + np.sin(phi) * v))
            if d[2] < 0.05:               # never approach from under the table
                continue
            hit = _endpoint_ok(world, ik, center, d, r, t_fc_inv, q_near)
            if hit:
                T_cam, T_fl, q, roll = hit
                return FaceResolution(T_cam, T_fl, float(theta), "cone", roll)
        theta += CONE_STEP_DEG
    return None
