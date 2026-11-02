"""Cloud -> triangle mesh, deterministically, with a fallback that never crashes.

The mesh is not decoration: it is the coverage *denominator*, the source of
every surface normal, and the occluder in the raycast. A mesh that
hallucinates area makes the plan look worse than it is, and a mesh that
closes an unseen back side makes it look better. So the recipe here is the
one that was measured on the real step clouds, not the library default:

1. **Normals** — Open3D estimation (r=12 mm, k=30, matching `fit/common.py`),
   then **orient toward the nearest recorded camera position**. Measured on
   2408-cup3: the `fit/common.py` centroid rule agrees with the recorded
   cameras on only 74 % of points at step 005 (the cup cavity and handle are
   where it fails) and 89 % at step 029;
   `orient_normals_consistent_tangent_plane` is worse (77 % / 69 %) and
   graph-dependent. Camera positions are data we already have at that step.
   With no cameras supplied we fall back to the centroid rule and say so in
   `info["normals"]`.
2. **Poisson depth 8**, `n_threads=1` (determinism), `linear_fit=False`.
   Depth 9 produced a byte-identical mesh at 2.5 mm spacing, so 8 is the
   saturation point, not a guess.
3. **Trim by distance to the input cloud, 6 mm.** Poisson balloons a closed
   skin over the unseen side: raw step-005 extent was [95, 132, 132] mm for
   a cloud measuring [93, 120, 90]. The density-quantile trim everyone
   reaches for does NOT fix this (q0.05 still leaves [93, 132, 114]); the
   distance trim does ([90, 129, 98]). 4 mm halved the area, 10 mm let the
   balloon back in.
4. **Fallback ladder** — alpha shape (12 mm), then convex hull. On these
   runs it never fired: Poisson d8 + trim works even on the 930-point first
   steps.

The meshes are **not watertight** and that is correct — the underside was
never seen, and `RaycastingScene` does not care. Winding is meaningless on
an open sheet (an early cup4 mesh has negative signed volume), so nothing
downstream may read orientation off the triangles: sample normals are
re-signed from the oriented cloud in `planner.sample_surface`.

    from inspection.viewplan import mesh
    m, info = mesh.reconstruct(P, cams=cam_positions)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from inspection.fit.common import (NORMAL_KNN, NORMAL_RADIUS_M, cloud_spacing,
                                   maturity_reason)

__all__ = ["maturity_reason", "reconstruct", "oriented_normals", "save",
           "load", "table_plane", "mesh_area", "POISSON_DEPTH", "TRIM_M",
           "ALPHA_M", "TABLE_MARGIN_M"]

#: Octree depth. Measured saturation point at 2.5 mm point spacing — depth 9
#: returned a byte-identical mesh, depth 7 lost the handle's inner edge.
POISSON_DEPTH = 8
#: Distance-to-cloud trim. 2.4x the 2.5 mm point spacing, ~= the 3-6 mm
#: registration noise: far enough not to eat real surface, near enough to
#: cut the Poisson balloon over the unseen side.
TRIM_M = 0.006
#: Alpha for the fallback mesher (never needed on cup3/4/5, kept honest).
ALPHA_M = 0.012
#: The recorder kept points >= 8 mm above the fitted table
#: (`cell/geometry.py ABOVE_TABLE_M`); undoing that margin recovers the plane.
TABLE_MARGIN_M = 0.008
#: Sanity window for mesh area against a crude cloud-area estimate.
AREA_RATIO_WINDOW = (0.3, 3.0)
MIN_TRIANGLES = 100


def _o3d():
    import open3d as o3d
    return o3d


def oriented_normals(P: np.ndarray, cams: Optional[np.ndarray] = None,
                     radius: float = NORMAL_RADIUS_M,
                     knn: int = NORMAL_KNN) -> Tuple[np.ndarray, str]:
    """Unit normals with a pinned sign. Returns (N, rule-name).

    With `cams` (recorded camera positions, (C,3), base frame) each normal is
    flipped to face the *nearest* camera. Nearest, not best-aligned: once the
    run has surrounded the object there is a camera on both sides of nearly
    every tangent plane, and the best-aligned rule degenerates (it disagrees
    with the nearest rule on 55 % of points at step 029). Without cameras we
    use the centroid-outward rule from `fit/common.py`.
    """
    o3d = _o3d()
    P = np.asarray(P, dtype=float)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                             max_nn=knn))
    N = np.asarray(pc.normals).copy()
    if cams is not None and len(cams):
        C = np.asarray(cams, dtype=float).reshape(-1, 3)
        near = C[np.argmin(np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2),
                           axis=1)]
        flip = np.einsum("ij,ij->i", N, near - P) < 0
        rule = "nearest_camera"
    else:
        flip = np.einsum("ij,ij->i", N, P - P.mean(0)) < 0
        rule = "centroid_outward"
    N[flip] *= -1
    n = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.where(n < 1e-12, 1.0, n)
    return N, rule


def mesh_area(m) -> float:
    return float(m.get_surface_area()) if len(m.triangles) else 0.0


def _clean(m, keep_largest: bool = True) -> Tuple[Any, int]:
    """Degenerate/duplicate scrub + largest connected component."""
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    n_comp = 1
    if keep_largest and len(m.triangles):
        labels, counts, _ = m.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        n_comp = int(len(counts))
        if n_comp > 1:
            m.remove_triangles_by_mask(labels != int(np.argmax(counts)))
            m.remove_unreferenced_vertices()
    return m, n_comp


def _poisson(pc, P: np.ndarray, depth: int, trim_m: float):
    o3d = _o3d()
    m, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, width=0, scale=1.1, linear_fit=False, n_threads=1)
    V = np.asarray(m.vertices)
    if len(V):
        d, _ = cKDTree(P).query(V)
        m.remove_vertices_by_mask(d > trim_m)
    return m


def _alpha(pc, alpha: float):
    o3d = _o3d()
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha)


def _hull(pc):
    m, _ = pc.compute_convex_hull()
    return m


def reconstruct(P: np.ndarray, *, seed: int = 0,
                cams: Optional[np.ndarray] = None,
                method: str = "poisson",
                depth: int = POISSON_DEPTH,
                trim_m: float = TRIM_M,
                alpha_m: float = ALPHA_M) -> Tuple[Any, Dict[str, Any]]:
    """(TriangleMesh, info) or `ValueError` with a one-sentence reason.

    `method` picks the primary mesher — "poisson" (default) or "alpha" (the
    cross-check mesher: mechanistically different, so a plan that agrees
    across both is a statement about the points rather than about Poisson).
    Either way the ladder ends at the convex hull rather than at an
    exception, and `info["method"]` records which rung answered.

    `seed` is accepted for signature symmetry with the rest of the package;
    every step here is deterministic (`n_threads=1`, no sampler), so it is
    unused — that is deliberate, not an oversight.
    """
    o3d = _o3d()
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 4:
        raise ValueError(f"cannot mesh a {P.shape} array")
    N, rule = oriented_normals(P, cams)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)

    spacing = cloud_spacing(P)
    crude_area = len(P) * spacing ** 2
    lo, hi = AREA_RATIO_WINDOW
    ladder = ([("poisson", lambda: _poisson(pc, P, depth, trim_m)),
               ("alpha", lambda: _alpha(pc, alpha_m))]
              if method == "poisson" else
              [("alpha", lambda: _alpha(pc, alpha_m)),
               ("poisson", lambda: _poisson(pc, P, depth, trim_m))])
    ladder.append(("hull", lambda: _hull(pc)))

    tried = []
    for name, build in ladder:
        try:
            m = build()
        except Exception as exc:                              # noqa: BLE001
            tried.append(f"{name}:{type(exc).__name__}")
            continue
        m, n_comp = _clean(m)
        area = mesh_area(m)
        nt = len(m.triangles)
        if name != "hull":
            if nt < MIN_TRIANGLES:
                tried.append(f"{name}:{nt}tri")
                continue
            if not (lo * crude_area <= area <= hi * crude_area):
                tried.append(f"{name}:area{area / max(crude_area, 1e-12):.2f}x")
                continue
        if nt < 4:
            tried.append(f"{name}:{nt}tri")
            continue
        m.compute_vertex_normals()
        info = {
            "method": name,
            "depth": int(depth) if name == "poisson" else 0,
            "trim_m": float(trim_m) if name == "poisson" else 0.0,
            "alpha_m": float(alpha_m) if name == "alpha" else 0.0,
            "normals": rule,
            "n_vertices": int(len(m.vertices)),
            "n_triangles": int(nt),
            "area_m2": float(area),
            "cloud_area_m2": float(crude_area),
            "area_ratio_cloud": float(area / max(crude_area, 1e-12)),
            "n_components": int(n_comp),
            "watertight": bool(m.is_watertight()),
            "fallback_from": ",".join(tried),
        }
        return m, info
    raise ValueError(f"no mesher produced a usable surface ({'; '.join(tried)})")


def save(m, path) -> None:
    o3d = _o3d()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), m, write_ascii=False):
        raise IOError(f"could not write mesh to {path}")


def load(path):
    o3d = _o3d()
    m = o3d.io.read_triangle_mesh(str(path))
    if not len(m.triangles):
        raise IOError(f"no triangles in {path}")
    return m


def table_plane(P: np.ndarray, margin_m: float = TABLE_MARGIN_M) -> list:
    """`[0, 0, 1, -table_z]` — the table under an object-only step cloud.

    No RANSAC. `cell/geometry.py:fit_table` is unseeded, so calling it would
    make two runs of the planner on the same cloud disagree; and it is not
    needed. The recorder already dropped the table and kept only points
    >= 8 mm above it, so the cloud's own low tail IS the table plus that
    margin. The 1st percentile (robust to a few low outliers) minus 8 mm
    measured 0.0080 / 0.0076 / 0.0067 m on cup3 / cup4 / cup5 at the final
    step and moved < 4 mm across all steps of a run.

    The plane is expressed so `[a,b,c] . p + d` is the signed height above
    the table in metres, positive up.
    """
    P = np.asarray(P, dtype=float)
    z = float(np.percentile(P[:, 2], 1.0)) - float(margin_m)
    return [0.0, 0.0, 1.0, -z]


def height_above(plane, points: np.ndarray) -> np.ndarray:
    """Signed metres above `plane` for each point — the table guard's test."""
    a, b, c, d = [float(v) for v in plane]
    Q = np.asarray(points, dtype=float).reshape(-1, 3)
    return Q @ np.array([a, b, c]) + d
