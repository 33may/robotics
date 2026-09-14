"""`planar` — Open3D planar-patch detection behind a curvature guard.

The detector is Open3D 0.19's `detect_planar_patches` (Araujo & Oliveira's
robust-statistics planar patch detection). Left alone on a cup it is a
menace: on 2408-cup3's final cloud it returns eleven "planar patches", four
of them 80 x 75 mm slabs cut straight through a 70 mm-wide curved body. That
is the failure mode this module exists to stop — a fitter that tiles a
cylinder with fake planes is worse than one that says nothing, because the
fake planes look like structure to everything downstream.

So the raw patches are candidates, nothing more. Each one is cleaned and
then has to survive a guard:

1. **Clean.** Keep points inside the patch's in-plane footprint AND within
   4 mm of its plane AND whose normal agrees in sign with the patch normal
   AND that belong to the largest connected blob. Open3D's boxes are thick
   enough to scoop the far wall of a cup; two opposing surfaces average out
   to something flat, and the guard would be fooled by its own input.
2. **Guard.** Reject unless all hold:
   - support >= 40 points and both in-plane extents >= 15 mm (a face smaller
     than that, on a 0.1 m object, is noise wearing a hat);
   - RMS distance to the plane <= 3.0 mm (~2x depth noise);
   - **the patch is wide enough for the question to be decidable.** An arc of
     radius R spanning width w bows away from its chord by w^2/(8R). If that
     is under 2x the patch's own plane-fit rms, this cloud cannot tell a
     plane from an arc of the object's own curvature (R = 60 mm), and the
     honest answer is to reject rather than to call it flat. At 1.2 mm rms
     that floor is 34 mm — which is what rejects the 22 mm-wide strip
     Open3D reports down the front of 2408-cup4, whose points in fact lie on
     a 50 mm-radius arc;
   - median angle between point normals and the patch normal <= 13 deg;
   - **the normal field does not rotate coherently.** Regress the in-plane
     component of the point normals against the in-plane coordinate: the
     slope is 1/R for a surface of curvature radius R, and r^2 says whether
     the rotation is systematic or is just noise. Reject if r^2 > 0.5 with
     R < 300 mm.

The last is the curvature guard proper, and its thresholds are not guesses.
On a synthetic 7200-point box of 60 x 50 x 90 mm (sigma = 1.5 mm) the six
true faces score r^2 0.12-0.26 with inferred R 153-224 mm and fan 8.5-10.9
deg. On a synthetic cylinder of radius 35 mm the patches score r^2 0.77-0.91
with R 35-37 mm and fan 14-20 deg. r^2 is the discriminator with the margin;
R only bounds *which* curvatures count as curvature, which is why a
large-radius cylinder is caught too — its normals still rotate coherently.

A patch's PCA surface variation lam3/sum(lam) was tried as a guard and
dropped: it is extent-dependent (lam3 is the noise floor, lam1+lam2 the
face's own size), so it rejects the 60 x 50 mm faces of a small box while
passing the 93 x 51 mm ones. `max_rms_m` is the scale-free form of the same
test, since rms == sqrt(lam3).

Survivors are merged when they describe the same plane (normals within
10 deg, offsets within 4 mm), refit, and emitted as `plane_patch`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from inspection.fit import common
from inspection.fit.contract import Decomposition, Primitive, abstained, pose_from

NAME = "planar"
SEMVER = "1.0"

PARAMS = dict(
    # --- Open3D detector knobs -------------------------------------------
    # min_num_points sets the octree depth, i.e. the *scale* of candidates.
    # 10 is Open3D's own default and the only setting that still yields
    # candidates on our clouds: 40 gives one patch, 50 gives none. We keep
    # it permissive on purpose and let the guard do the rejecting.
    min_num_points=10,
    normal_variance_threshold_deg=60.0,
    coplanarity_deg=75.0,
    outlier_ratio=0.75,
    min_plane_edge_length=0.015,
    # --- cleaning ---------------------------------------------------------
    band_m=0.004,              # +/- half-thickness of the plane's slab
    normal_side_deg=40.0,      # keep one face of a thin wall, not both
    link_factor=2.5,           # connected-component linkage, x spacing
    # --- guard ------------------------------------------------------------
    min_support=40,
    min_edge_m=0.015,
    max_rms_m=0.0030,
    max_fan_deg=13.0,
    min_curv_radius_m=0.30,
    curv_r2=0.50,
    # Resolvability. An arc of radius R spanning width w departs from its
    # chord by a sagitta of w^2/(8R). If that is not at least
    # `sagitta_factor` x the patch's own plane-fit rms, the data cannot tell
    # a plane from an arc of the object's own curvature, and the honest
    # answer is to reject rather than to call it flat. R is fixed at
    # `resolve_radius_m` = 0.06 m, the curvature scale of a 0.1 m object.
    resolve_radius_m=0.06,
    sagitta_factor=2.0,
    # --- merge ------------------------------------------------------------
    merge_normal_deg=10.0,
    merge_offset_m=0.004,
)
VERSION = common.version_string(NAME, SEMVER, PARAMS)


def _clean(P, N, box, spacing, prm):
    """Points that genuinely belong to this patch's surface."""
    R = np.asarray(box.R, float)
    c = np.asarray(box.center, float)
    e = np.asarray(box.extent, float)
    L = (P - c) @ R
    half = e / 2.0
    m = ((np.abs(L[:, 0]) <= half[0]) & (np.abs(L[:, 1]) <= half[1])
         & (np.abs(L[:, 2]) <= prm["band_m"]))
    idx = np.flatnonzero(m)
    if len(idx) < 4:
        return idx
    n = R[:, 2]
    d = N[idx] @ n
    sgn = 1.0 if (d > 0).sum() >= (d < 0).sum() else -1.0
    idx = idx[(sgn * d) > np.cos(np.deg2rad(prm["normal_side_deg"]))]
    if len(idx) < 4:
        return idx
    return common.largest_component(P, idx, spacing, prm["link_factor"])


def _measure(P, N, idx):
    """Plane fit + the four guard statistics for one cleaned patch."""
    Q = P[idx]
    c, axes, lam = common.pca_frame(Q)
    n = axes[:, 2]
    if float(np.mean(N[idx] @ n)) < 0:
        n = -n
        axes = np.column_stack([axes[:, 0], -axes[:, 1], n])
    d = (Q - c) @ n
    rms = float(np.sqrt((d ** 2).mean()))
    sv = float(lam[2] / max(lam.sum(), 1e-18))
    nn = N[idx] * np.sign(N[idx] @ n)[:, None]
    fan = float(np.median(np.degrees(np.arccos(np.clip(nn @ n, -1, 1)))))
    # curvature: the in-plane tilt of the normal against the in-plane
    # coordinate. Slope is 1/R for a cylinder; ~0 with low r^2 for a plane.
    k, r2 = 0.0, 0.0
    for e in (axes[:, 0], axes[:, 1]):
        x = (Q - c) @ e
        y = nn @ e
        if x.std() < 1e-9:
            continue
        A = np.polyfit(x, y, 1)
        ss = float(((y - y.mean()) ** 2).sum())
        rr = 1.0 - float(((y - np.polyval(A, x)) ** 2).sum()) / max(ss, 1e-18)
        if abs(A[0]) > abs(k):
            k, r2 = float(A[0]), rr
    R_curv = 1.0 / abs(k) if abs(k) > 1e-6 else np.inf
    ext = (float(4 * np.sqrt(max(lam[0], 0))), float(4 * np.sqrt(max(lam[1], 0))))
    return dict(center=c, axes=axes, normal=n, rms=rms, sv=sv, fan=fan,
                R_curv=R_curv, r2=r2, extents=ext, n=len(idx))


def _verdict(m, prm) -> str:
    """'' if the patch is a real plane, else the reason it is not."""
    if m["n"] < prm["min_support"]:
        return f"support {m['n']} < {prm['min_support']}"
    if min(m["extents"]) < prm["min_edge_m"]:
        return f"short edge {min(m['extents']) * 1000:.0f} mm"
    if m["rms"] > prm["max_rms_m"]:
        return f"rms {m['rms'] * 1000:.1f} mm off-plane"
    w_min = np.sqrt(8 * prm["resolve_radius_m"]
                    * prm["sagitta_factor"] * max(m["rms"], 1e-4))
    if min(m["extents"]) < w_min:
        return (f"undecidable: {min(m['extents']) * 1000:.0f} mm wide needs "
                f"{w_min * 1000:.0f} mm to separate plane from "
                f"{prm['resolve_radius_m'] * 1000:.0f} mm arc at "
                f"{m['rms'] * 1000:.1f} mm noise")
    if m["fan"] > prm["max_fan_deg"]:
        return f"normal fan {m['fan']:.1f} deg"
    if m["R_curv"] < prm["min_curv_radius_m"] and m["r2"] > prm["curv_r2"]:
        return f"curved: R={m['R_curv'] * 1000:.0f} mm (r2={m['r2']:.2f})"
    return ""


def _merge(kept, prm):
    """Fold patches that describe the same plane into one, greedily by size.

    Greedy by support, not by extraction order — two runs that find the same
    patches in a different order still merge them the same way.
    """
    order = sorted(range(len(kept)), key=lambda i: -kept[i][1]["n"])
    groups: List[List[int]] = []
    for i in order:
        mi = kept[i][1]
        for g in groups:
            mj = kept[g[0]][1]
            if (abs(float(mi["normal"] @ mj["normal"]))
                    > np.cos(np.deg2rad(prm["merge_normal_deg"]))
                    and abs(float((mi["center"] - mj["center"]) @ mj["normal"]))
                    < prm["merge_offset_m"]):
                g.append(i)
                break
        else:
            groups.append([i])
    return [np.unique(np.concatenate([kept[i][0] for i in g])) for g in groups]


def fit(points: np.ndarray, *, seed: int = 0, table_normal=None,
        warm_start: Optional[Decomposition] = None, **kw) -> Decomposition:
    """Fit bounded planar patches. `seed` unused — the detector is exact."""
    prm = dict(PARAMS)
    prm.update({k: v for k, v in kw.items() if k in PARAMS})
    version = (VERSION if prm == PARAMS
               else common.version_string(NAME, SEMVER, prm))

    P = np.asarray(points, float)
    why = common.maturity_reason(P)
    if why:
        return abstained(NAME, version, why)

    import open3d as o3d
    N = common.estimate_normals(P)
    spacing = common.cloud_spacing(P)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    boxes = pc.detect_planar_patches(
        normal_variance_threshold_deg=prm["normal_variance_threshold_deg"],
        coplanarity_deg=prm["coplanarity_deg"],
        outlier_ratio=prm["outlier_ratio"],
        min_plane_edge_length=prm["min_plane_edge_length"],
        min_num_points=prm["min_num_points"],
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=common.NORMAL_KNN))

    kept, rejected = [], []
    for b in boxes:
        idx = _clean(P, N, b, spacing, prm)
        if len(idx) < 4:
            rejected.append("empty after cleaning")
            continue
        m = _measure(P, N, idx)
        why_not = _verdict(m, prm)
        (rejected.append(why_not) if why_not else kept.append((idx, m)))

    if not kept:
        return abstained(
            NAME, version,
            f"no planar patch survived the curvature guard "
            f"({len(boxes)} candidates, rejected: "
            f"{'; '.join(sorted(set(rejected))[:4]) or 'none proposed'})")

    prims = []
    for idx in _merge(kept, prm):
        m = _measure(P, N, idx)
        if _verdict(m, prm):          # a merge can only make a patch worse
            continue
        Q = P[idx]
        c, axes = m["center"], m["axes"]
        u = (Q - c) @ axes[:, 0]
        v = (Q - c) @ axes[:, 1]
        cen = c + 0.5 * (u.min() + u.max()) * axes[:, 0] \
                + 0.5 * (v.min() + v.max()) * axes[:, 1]
        ext = [float(u.max() - u.min()), float(v.max() - v.min())]
        prims.append((Primitive(
            kind="plane_patch",
            pose=pose_from(np.column_stack([axes[:, 0], axes[:, 1], m["normal"]]), cen),
            params={"extents": ext}, label="",
            inliers=idx.tolist(),
            residual_mm=m["rms"] * 1000, support=len(idx)),
            m["normal"], float(m["center"] @ m["normal"])))

    if not prims:
        return abstained(NAME, version,
                         "patches survived individually but not after merging")

    labels = common.disambiguate([f"face_{common.direction_name(n)}" for _, n, _ in prims],
                                 [off for _, _, off in prims])
    out = []
    for (p, _, _), lb in zip(prims, labels):
        p.label = lb
        out.append(p)
    return Decomposition(method=NAME, version=version, primitives=out,
                         reason=f"{len(boxes)} candidates, {len(out)} kept")


def diagnostics(points: np.ndarray, **kw) -> dict:
    """Guard bookkeeping for the score log — never part of the contract."""
    prm = dict(PARAMS)
    prm.update({k: v for k, v in kw.items() if k in PARAMS})
    P = np.asarray(points, float)
    if common.maturity_reason(P):
        return {"candidates": 0, "rejected": []}
    import open3d as o3d
    N = common.estimate_normals(P)
    spacing = common.cloud_spacing(P)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    boxes = pc.detect_planar_patches(
        normal_variance_threshold_deg=prm["normal_variance_threshold_deg"],
        coplanarity_deg=prm["coplanarity_deg"],
        outlier_ratio=prm["outlier_ratio"],
        min_plane_edge_length=prm["min_plane_edge_length"],
        min_num_points=prm["min_num_points"],
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=common.NORMAL_KNN))
    rej = []
    for b in boxes:
        idx = _clean(P, N, b, spacing, prm)
        if len(idx) < 4:
            rej.append("empty after cleaning")
            continue
        rej.append(_verdict(_measure(P, N, idx), prm) or "kept")
    return {"candidates": len(boxes), "rejected": rej,
            "n_kept": sum(r == "kept" for r in rej)}
