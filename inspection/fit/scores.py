"""Uniform scoring — computed OUTSIDE the fitters, so methods compare.

A fitter reports what it believes; it does not get to grade itself. Every
`Decomposition` that leaves the bank has had `coverage`, `residual_mm`, and
each primitive's `inliers`/`support`/`residual_mm` recomputed here from one
function of `(points, kind, pose, params)`. A planar decomposition and a
superquadric one are then two numbers on the same axis.

Point-to-primitive distance, per kind — all unsigned, all in metres:

    box           surface of the solid box (outside: to the box; inside: to
                  the nearest face)
    cylinder      the LATERAL surface only, clipped to the axial extent. A
                  cylinder RANSAC fits a tube; capping it would silently
                  claim a cup's open top and inflate coverage.
    plane_patch   the bounded rectangle (not the infinite plane), so a patch
                  cannot claim points a metre away along its own plane
    superquadric  the radial approximation |p|.|F(p)^(-eps1/2) - 1|, the same
                  one EMS itself minimises — using a different metric to
                  score EMS than EMS optimises would be scoring a different
                  problem
    mesh          distance to the nearest vertex (a proxy; no mesh fitter is
                  registered yet, and it is documented as a proxy)

**What these numbers are not.** They are residual-family scores: they
measure how well a shape hugs the points it was fitted to. They are blind
to coherent bias — the ~4 cm fused-vs-live offset this project has already
been bitten by moves the whole cloud and every primitive with it, and
coverage stays 1.0. Never sell coverage or residual as accuracy.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from inspection.fit.contract import Decomposition, Primitive

#: A point counts as claimed within this distance of a primitive's surface.
#: 5 mm = 2x the 2.5 mm point spacing = ~3 sigma of D405 depth noise; it is a
#: statement about the sensor, not about any one object.
COVER_TOL_M = 0.005


def _local(P: np.ndarray, prim: Primitive) -> np.ndarray:
    T = prim.T
    return (P - T[:3, 3]) @ T[:3, :3]


def _box_dist(L: np.ndarray, extents) -> np.ndarray:
    h = np.asarray(extents, float) / 2.0
    q = np.abs(L) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return np.abs(outside + inside)


def _cyl_dist(L: np.ndarray, radius: float, length: float) -> np.ndarray:
    r = np.hypot(L[:, 0], L[:, 1])
    dz = np.maximum(np.abs(L[:, 2]) - length / 2.0, 0.0)
    return np.hypot(r - radius, dz)


def _plane_dist(L: np.ndarray, extents) -> np.ndarray:
    h = np.asarray(extents, float) / 2.0
    dx = np.maximum(np.abs(L[:, 0]) - h[0], 0.0)
    dy = np.maximum(np.abs(L[:, 1]) - h[1], 0.0)
    return np.sqrt(dx ** 2 + dy ** 2 + L[:, 2] ** 2)


def _sq_dist(L: np.ndarray, scale, eps) -> np.ndarray:
    a = np.maximum(np.asarray(scale, float), 1e-9)
    e1, e2 = float(max(eps[0], 1e-3)), float(max(eps[1], 1e-3))
    r = np.linalg.norm(L, axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = ((((L[:, 0] / a[0]) ** 2) ** (1 / e2)
              + ((L[:, 1] / a[1]) ** 2) ** (1 / e2)) ** (e2 / e1)
             + ((L[:, 2] / a[2]) ** 2) ** (1 / e1))
        d = r * np.abs(f ** (-e1 / 2) - 1.0)
    return np.nan_to_num(d, nan=np.inf, posinf=np.inf)


def _mesh_dist(P: np.ndarray, params) -> np.ndarray:
    from scipy.spatial import cKDTree
    V = np.asarray(params["vertices"], float)
    if len(V) == 0:
        return np.full(len(P), np.inf)
    return cKDTree(V).query(P)[0]


def surface_distance(P: np.ndarray, prim: Primitive) -> np.ndarray:
    """|distance| from every point to `prim`'s surface, in metres."""
    P = np.asarray(P, float)
    if prim.kind == "mesh":
        return _mesh_dist(P, prim.params)
    L = _local(P, prim)
    if prim.kind == "box":
        return _box_dist(L, prim.params["extents"])
    if prim.kind == "cylinder":
        return _cyl_dist(L, prim.params["radius"], prim.params["length"])
    if prim.kind == "plane_patch":
        return _plane_dist(L, prim.params["extents"])
    if prim.kind == "superquadric":
        return _sq_dist(L, prim.params["scale"], prim.params["eps"])
    raise ValueError(f"unscorable kind {prim.kind!r}")


def score(P: np.ndarray, dec: Decomposition,
          tol: float = COVER_TOL_M) -> Decomposition:
    """Recompute every score on `dec` in place, and return it.

    Each point goes to its single nearest primitive and only counts if that
    primitive is within `tol` — so overlapping primitives cannot both bank
    the same point and inflate the decomposition's coverage.
    """
    P = np.asarray(P, float)
    if dec.abstain or not dec.primitives or len(P) == 0:
        dec.coverage = 0.0
        dec.residual_mm = 0.0
        dec.n_primitives = len(dec.primitives)
        return dec

    D = np.stack([surface_distance(P, p) for p in dec.primitives], axis=1)
    D = np.nan_to_num(D, nan=np.inf, posinf=np.inf)
    owner = np.argmin(D, axis=1)
    best = D[np.arange(len(P)), owner]
    claimed = best <= tol

    for j, prim in enumerate(dec.primitives):
        m = claimed & (owner == j)
        prim.inliers = np.flatnonzero(m).tolist()
        prim.support = int(m.sum())
        prim.residual_mm = float(np.sqrt((best[m] ** 2).mean()) * 1000) if m.any() else 0.0

    dec.coverage = float(claimed.mean())
    dec.residual_mm = (float(np.sqrt((best[claimed] ** 2).mean()) * 1000)
                       if claimed.any() else 0.0)
    dec.n_primitives = len(dec.primitives)
    return dec


def score_row(P: np.ndarray, dec: Decomposition) -> Dict:
    """One flat record per (step, method) for the score log / X5 curves."""
    return {"method": dec.method, "version": dec.version,
            "n_points": int(len(P)), "abstain": bool(dec.abstain),
            "reason": dec.reason, "n_primitives": int(dec.n_primitives),
            "coverage": round(float(dec.coverage), 4),
            "residual_mm": round(float(dec.residual_mm), 3),
            "kinds": sorted({p.kind for p in dec.primitives}),
            "labels": [p.label for p in dec.primitives],
            "support": [int(p.support) for p in dec.primitives]}
