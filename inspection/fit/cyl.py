"""`cyl` — cylinder RANSAC on estimated normals, refined by least squares.

**Why this is written here and not imported.** The library survey on
2026-09-03 found nothing installable on py3.11 that fits a cylinder well:

- `pyransac3d` 0.7.0 installs fine, but its own documentation says the
  cylinder model "does NOT present good results on real data on the current
  version". Measured on our clouds it agrees: it returns a *horizontal* axis
  on 2408-cup5 (axis [0.994, 0.105, -0.013] on an upright mug) because its
  hypothesis uses three sampled points and no normals at all.
- `python-pcl` / `pclpy`: no py3.11 distribution on PyPI at all. PCL's
  `SACMODEL_CYLINDER` is the algorithm we want; the bindings are dead.
- CGAL's SWIG bindings do ship `CGAL_Shape_detection.efficient_RANSAC(...,
  cylinders=True)`, but the API returns only a per-point shape index — no
  radius, no axis — so the parameters would have to be refitted here anyway,
  in exchange for a ~100 MB binary dependency in a shared env.
- `cylinder_fitting` on PyPI is plain least squares with no outlier model;
  it fits the cup, the handle and the rim as one cylinder.

So the hypothesis step is PCL's `SACMODEL_CYLINDER` formulation, implemented
directly: sample two points with normals, the axis is `n_i x n_j`, and the
axis position is where the two normals' projections intersect in the plane
perpendicular to it. That is ~40 lines, uses Open3D only for normals and
scipy for the refine, and is fully seeded.

**Determinism.** Every hypothesis comes from `np.random.default_rng(seed)`;
the same points and the same seed give the same cylinder, bit for bit.

**Sequential extraction.** After a cylinder is accepted its inliers are
removed and the search runs again, up to `max_cylinders`, stopping when the
best remaining support falls under `min_support_frac` of the original cloud.
Primitives are named for their geometry afterwards, never for the order:
the upright one with the most support is `body`, the rest are `cyl_<axis>`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from inspection.fit import common
from inspection.fit.contract import (Decomposition, Primitive, abstained,
                                     frame_from_axis)

NAME = "cyl"
SEMVER = "1.0"

PARAMS = dict(
    n_hypotheses=3000,
    # 4 mm inlier band: 2.5 mm spacing + 1.5 mm depth noise. Wide enough to
    # hold a 3 mm-thick cup wall (outer and inner surface both), narrow
    # enough that the handle does not join the body.
    inlier_band_m=0.004,
    # The point's normal must actually point radially. Without this test a
    # cylinder can be "supported" by a flat face that happens to graze it.
    normal_tol_deg=35.0,
    # Radii we will entertain, for a ~0.1 m table-top object.
    min_radius_m=0.008,
    max_radius_m=0.090,
    min_support=120,
    min_support_frac=0.06,
    max_cylinders=2,
    refine_rounds=3,
    # An accepted cylinder must be seen from enough directions to be a
    # cylinder rather than a curved strip: inliers must span >= 120 deg of
    # azimuth about the axis.
    min_azimuth_deg=120.0,
    upright_tol_deg=25.0,
    extent_pct=1.0,            # axial extent from the 1st..99th percentile
    # A second cylinder must be a *different* cylinder. Without this the cup
    # body gets found twice: the first pass' normal test leaves the noisier
    # half of the wall unclaimed, and it re-forms into a near-coaxial twin
    # (observed on 2408-cup3: r=34.6 mm then r=39.5 mm, same axis).
    dup_axis_deg=20.0,
    dup_axis_offset_m=0.015,
    dup_radius_m=0.012,
)
VERSION = common.version_string(NAME, SEMVER, PARAMS)


def _hypotheses(P, N, rng, prm):
    """PCL SACMODEL_CYLINDER hypothesis: two points and their normals."""
    n = len(P)
    pairs = rng.integers(0, n, size=(prm["n_hypotheses"], 2))
    for i, j in pairs:
        if i == j:
            continue
        ni, nj = N[i], N[j]
        ax = np.cross(ni, nj)
        na = np.linalg.norm(ax)
        if na < 1e-3:                       # near-parallel normals: no axis
            continue
        ax = ax / na
        # 2D frame perpendicular to the axis
        u = ni - (ni @ ax) * ax
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            continue
        u = u / nu
        v = np.cross(ax, u)
        p0 = np.array([P[i] @ u, P[i] @ v])
        p1 = np.array([P[j] @ u, P[j] @ v])
        d0 = np.array([ni @ u, ni @ v])
        d1 = np.array([nj @ u, nj @ v])
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        d0, d1 = d0 / l0, d1 / l1
        det = d0[0] * (-d1[1]) - (-d1[0]) * d0[1]
        if abs(det) < 1e-3:
            continue
        rhs = p1 - p0
        t = (rhs[0] * (-d1[1]) - (-d1[0]) * rhs[1]) / det
        c2 = p0 + t * d0
        r = float(np.linalg.norm(c2 - p0))
        if not (prm["min_radius_m"] <= r <= prm["max_radius_m"]):
            continue
        yield ax, c2[0] * u + c2[1] * v, r


def _inliers(P, N, ax, c, r, prm):
    d = P - c
    d = d - np.outer(d @ ax, ax)
    dist = np.linalg.norm(d, axis=1)
    ok = np.abs(dist - r) <= prm["inlier_band_m"]
    radial = d / np.maximum(dist, 1e-12)[:, None]
    cos = np.abs(np.einsum("ij,ij->i", radial, N))
    ok &= cos > np.cos(np.deg2rad(prm["normal_tol_deg"]))
    return ok


def _refine(Q, ax, c, r):
    """Least-squares cylinder on the inliers: axis (2 angles), point, radius."""
    th = float(np.arccos(np.clip(ax[2], -1, 1)))
    ph = float(np.arctan2(ax[1], ax[0]))
    x0 = np.array([th, ph, c[0], c[1], c[2], r], float)

    def unpack(x):
        a = np.array([np.sin(x[0]) * np.cos(x[1]),
                      np.sin(x[0]) * np.sin(x[1]), np.cos(x[0])])
        return a, x[2:5], x[5]

    def res(x):
        a, cc, rr = unpack(x)
        d = Q - cc
        d = d - np.outer(d @ a, a)
        return np.linalg.norm(d, axis=1) - rr

    try:
        out = least_squares(res, x0, method="lm", max_nfev=300)
        a, cc, rr = unpack(out.x)
    except Exception:
        return ax, c, r
    if not np.isfinite(rr) or rr <= 0:
        return ax, c, r
    return a, cc, float(rr)


def _azimuth_span(Q, ax, c):
    """Degrees of azimuth about the axis that the inliers actually cover."""
    d = Q - c
    d = d - np.outer(d @ ax, ax)
    ref = np.eye(3)[int(np.argmin(np.abs(ax)))]
    u = np.cross(ref, ax)
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    a = np.sort(np.degrees(np.arctan2(d @ v, d @ u)) % 360.0)
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    return float(360.0 - gaps.max())


def _duplicate(c, accepted, prm) -> bool:
    """Is this the same cylinder as one already accepted?"""
    for a in accepted:
        if abs(float(c["axis"] @ a["axis"])) < np.cos(np.deg2rad(prm["dup_axis_deg"])):
            continue
        d = c["center"] - a["center"]
        if float(np.linalg.norm(d - (d @ a["axis"]) * a["axis"])) > prm["dup_axis_offset_m"]:
            continue
        if abs(c["radius"] - a["radius"]) < prm["dup_radius_m"]:
            return True
    return False


def _shell(P, ax, c, r, band) -> np.ndarray:
    """Every point on the cylinder's shell, normals ignored.

    Used to retire points after a cylinder is accepted: they are explained,
    whether or not their (noisy) normal passed the radial test.
    """
    d = P - c
    d = d - np.outer(d @ ax, ax)
    return np.abs(np.linalg.norm(d, axis=1) - r) <= band


def _one(P, N, live, rng, prm):
    """Best cylinder among the still-unclaimed points, or None."""
    idx = np.flatnonzero(live)
    if len(idx) < prm["min_support"]:
        return None
    Ps, Ns = P[idx], N[idx]
    best = (0, None)
    for ax, c, r in _hypotheses(Ps, Ns, rng, prm):
        ok = _inliers(Ps, Ns, ax, c, r, prm)
        s = int(ok.sum())
        if s > best[0]:
            best = (s, (ax, c, r))
    if best[1] is None or best[0] < prm["min_support"]:
        return None
    ax, c, r = best[1]
    ok = _inliers(Ps, Ns, ax, c, r, prm)
    for _ in range(prm["refine_rounds"]):
        if ok.sum() < 8:
            break
        ax, c, r = _refine(Ps[ok], ax, c, r)
        if not (prm["min_radius_m"] <= r <= prm["max_radius_m"]):
            return None
        ok = _inliers(Ps, Ns, ax, c, r, prm)
    if ok.sum() < prm["min_support"]:
        return None
    if ax[2] < 0:                      # pin the axis head: +Z-ish, always
        ax = -ax
    Q = Ps[ok]
    if _azimuth_span(Q, ax, c) < prm["min_azimuth_deg"]:
        return None
    # Extent and centre from the inliers projected on the axis. Percentiles,
    # not max-min: the extreme of a noisy sample sits ~3 sigma past the true
    # end, which on 6000 points reads a 90 mm cylinder as 96.5 mm. The 1st
    # and 99th percentile of a uniformly sampled segment span 98% of it, so
    # the span is divided back by 0.98.
    t = (Q - c) @ ax
    lo, hi = np.percentile(t, [prm["extent_pct"], 100 - prm["extent_pct"]])
    length = float((hi - lo) / (1.0 - 2 * prm["extent_pct"] / 100.0))
    center = c + 0.5 * (lo + hi) * ax
    d = Q - center
    d = d - np.outer(d @ ax, ax)
    resid = np.abs(np.linalg.norm(d, axis=1) - r)
    return dict(axis=ax, center=center, radius=float(r), length=length,
                inliers=idx[ok], rms_mm=float(np.sqrt((resid ** 2).mean()) * 1000))


def fit(points: np.ndarray, *, seed: int = 0, table_normal=None,
        warm_start: Optional[Decomposition] = None, **kw) -> Decomposition:
    prm = dict(PARAMS)
    prm.update({k: v for k, v in kw.items() if k in PARAMS})
    version = (VERSION if prm == PARAMS
               else common.version_string(NAME, SEMVER, prm))

    P = np.asarray(points, float)
    why = common.maturity_reason(P)
    if why:
        return abstained(NAME, version, why)

    N = common.estimate_normals(P)
    rng = np.random.default_rng(seed)
    live = np.ones(len(P), bool)
    found = []
    floor = max(prm["min_support"], int(prm["min_support_frac"] * len(P)))
    for _ in range(prm["max_cylinders"]):
        c = _one(P, N, live, rng, prm)
        if c is None or len(c["inliers"]) < floor:
            break
        if _duplicate(c, found, prm):
            break
        found.append(c)
        # retire the whole shell, not just the normal-passing inliers, so the
        # same wall cannot be discovered twice.
        t = (P - c["center"]) @ c["axis"]
        shell = _shell(P, c["axis"], c["center"], c["radius"],
                       prm["inlier_band_m"])
        live[shell & (np.abs(t) <= c["length"] / 2 + prm["inlier_band_m"])] = False

    if not found:
        return abstained(NAME, version,
                         f"no cylinder with >= {floor} supporting points and "
                         f">= {prm['min_azimuth_deg']:.0f} deg of azimuth")

    up = np.array([0, 0, 1.0]) if table_normal is None else \
        np.asarray(table_normal, float) / np.linalg.norm(table_normal)
    upright = [abs(float(c["axis"] @ up)) > np.cos(np.deg2rad(prm["upright_tol_deg"]))
               for c in found]
    body = None
    if any(upright):
        cands = [i for i, u in enumerate(upright) if u]
        body = max(cands, key=lambda i: len(found[i]["inliers"]))

    labels = [("body" if i == body else f"cyl_{common.axis_name(c['axis'])}")
              for i, c in enumerate(found)]
    labels = common.disambiguate(labels, [c["radius"] for c in found])

    prims = [Primitive(kind="cylinder",
                       pose=frame_from_axis(c["axis"], c["center"]),
                       params={"radius": c["radius"], "length": c["length"]},
                       label=lb, inliers=c["inliers"].tolist(),
                       residual_mm=c["rms_mm"], support=len(c["inliers"]))
             for c, lb in zip(found, labels)]
    return Decomposition(method=NAME, version=version, primitives=prims,
                         reason=f"{len(prims)} cylinder(s) extracted")
