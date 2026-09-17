"""Ray views (design: docs/2026-09-17-ray-views-design.md).

A view is a ray, not a point: distance is never computed by a formula —
it is the closest sample on the ray that is provably safe (sensor range,
IK, collision), found offline in the kinematic sandbox.
"""
from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from inspection.view.faceview import face_cam_pose, face_frame
from inspection.view.grid import H_BINS, R_MAX, V_ELEVATIONS
from inspection.view.viewsphere import ROLLS

#: One knob for every ray discretisation: depth grid AND face shift grid.
RAY_STEP_M = 0.02
#: D405 minimum depth range at 720p (datasheet + camera.py note).
MIN_RANGE_M = 0.105


@dataclass(frozen=True)
class Ray:
    """Anchor point and unit direction, base frame."""
    anchor: tuple
    direction: tuple

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError("ray direction must be a nonzero vector")
        object.__setattr__(self, "direction", tuple(d / n))
        object.__setattr__(self, "anchor",
                           tuple(float(v) for v in self.anchor))


Sample = namedtuple("Sample", ["depth", "roll", "q", "T_cam", "T_flange"])


def sample_ray(world, ik, t_flange_cam, ray, clear_m=0.0, q_near=None):
    """All provably-safe samples on `ray`, ascending depth.

    `clear_m` = camera-to-object-surface distance already consumed at depth
    0 (0 for a face ray — its anchor sits ON the wall; the anchor→surface
    run for a centroid ray). The sensor floor is surface distance, so a
    sample is admissible from depth clear_m + MIN_RANGE_M.
    """
    t_fc_inv = np.linalg.inv(np.asarray(t_flange_cam, dtype=float))
    lo = clear_m + MIN_RANGE_M
    d0 = float(np.ceil(lo / RAY_STEP_M)) * RAY_STEP_M
    out = []
    for d in np.arange(d0, R_MAX + 1e-9, RAY_STEP_M):
        for roll in ROLLS:
            T_cam = face_cam_pose(ray.anchor, ray.direction, float(d), roll)
            T_fl = T_cam @ t_fc_inv
            free = [q for q in ik.branches(T_fl, q_near=q_near)
                    if not world.is_colliding(q)]
            if free:
                out.append(Sample(float(d), float(roll), free[0],
                                  T_cam, T_fl))
                break                     # first roll wins at this depth
    return out


CONE_STEP_DEG, CONE_MAX_DEG, CONE_AZIMUTHS = 5.0, 60.0, 12


def cell_ray(center, h, v):
    """Ray of viewsphere cell {h, v}: centroid anchor, grid direction —
    the same az0 convention as ViewSphere (h=0 faces the robot base)."""
    center = np.asarray(center, dtype=float)
    az0 = np.arctan2(-center[1], -center[0])
    az = az0 + h * (2 * np.pi / H_BINS)
    el = np.deg2rad(V_ELEVATIONS[v])
    return Ray(tuple(center),
               (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                np.sin(el)))


def cell_clear_m(aabb, ray):
    """Anchor→surface run along the ray (slab method) — feeds sample_ray's
    clear_m so the sensor floor measures SURFACE distance, not anchor
    distance."""
    mn, mx = (np.asarray(v, dtype=float) for v in aabb)
    o, d = np.asarray(ray.anchor), np.asarray(ray.direction)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (mn - o) / d, (mx - o) / d
    t_exit = float(np.nanmin(np.maximum(t1, t2)))
    return max(t_exit, 0.0)


def _d_contain(fy, height_px, half_span):
    """Closest depth at which a vertical half-span [m] stays inside the
    frame — the inverse of shift_max(d) = d*height_px/(2*fy) - half_span.
    FOV containment is a per-depth constraint (controller ruling,
    task-2-report.md): it cannot cap a shift at a fixed MIN_RANGE_M depth,
    it becomes each family member's own MINIMUM admissible depth instead."""
    return (2.0 * fy / height_px) * half_span


def face_family(prim, face, fy, height_px):
    """Ordered (ray, shift, theta_deg, d_min) candidates for one boxguard
    face: parallel +z shifts ascending (fronto-parallel — a plane parallel
    to the sensor projects as a pure similarity, zero keystone), then the
    classic cone. `d_min` is the closest depth this member may be sampled
    from — MIN_RANGE_M for cone entries (tilted views never promised
    containment, they are already the degraded rung), and
    max(MIN_RANGE_M, containment depth) for shift/top entries, so a taller
    shift only pushes its own floor out rather than being rejected outright.
    First family member with samples (at its own d_min floor) wins; no
    member -> face inaccessible, honestly."""
    center, normal, (ext_a, ext_b) = face_frame(prim, face)
    out = []
    if abs(normal[2]) < 0.5:                      # side faces shift up
        # d_min <= R_MAX is the real cap now; equivalently
        # s <= R_MAX*height_px/(2*fy) - ext_b/2 (same shape as the old cap,
        # R_MAX in place of MIN_RANGE_M — enumerate while a depth exists).
        cap = R_MAX * height_px / (2.0 * fy) - ext_b / 2.0
        s = 0.0
        while s <= cap + 1e-9:
            d_min = max(MIN_RANGE_M, _d_contain(fy, height_px, s + ext_b / 2.0))
            out.append((Ray(tuple(center + np.array([0, 0, s])),
                            tuple(normal)), float(s), 0.0, float(d_min)))
            s += RAY_STEP_M
    else:                                         # top face: exact only
        d_min = max(MIN_RANGE_M,
                    _d_contain(fy, height_px, max(ext_a, ext_b) / 2.0))
        out.append((Ray(tuple(center), tuple(normal)), 0.0, 0.0,
                    float(d_min)))
    a = np.array([0.0, 0.0, 1.0])
    if abs(normal @ a) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    th = CONE_STEP_DEG
    while th <= CONE_MAX_DEG:
        t = np.deg2rad(th)
        for k in range(CONE_AZIMUTHS):
            phi = 2 * np.pi * k / CONE_AZIMUTHS
            d = (np.cos(t) * normal
                 + np.sin(t) * (np.cos(phi) * u + np.sin(phi) * v))
            if d[2] < 0.05:               # never approach from under table
                continue
            out.append((Ray(tuple(center), tuple(d)), 0.0, float(th),
                        MIN_RANGE_M))
        th += CONE_STEP_DEG
    return out
