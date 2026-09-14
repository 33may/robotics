"""Shared machinery for the fitters: scale, normals, labels, maturity gate.

Everything here is written for **table-top objects of roughly 0.1 m** seen
by a D405 at ~2.5 mm point spacing. Absolute thresholds are stated once, in
millimetres, with the reasoning attached — the pass-1 trap was copying
library defaults tuned for building scans onto a coffee cup.

Two conventions the whole package leans on:

- **Content-derived labels.** A primitive is named for what it *is* — its
  axis direction, its role — never for the order RANSAC happened to find it
  in. Two runs of the same fitter on the same cloud, or the same fitter on
  step k and step k+1, must agree on the name of the cup's body.
- **Version fingerprints.** A fitter's `version` is `name/semver+p:<hash>`
  where the hash covers every parameter that can change the answer. A fit
  JSON therefore says what produced it, and a parameter tweak is visible on
  disk without a diff.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# --- scale constants -------------------------------------------------------
#: Objects we fit. Used to scale every "is this big enough / small enough"
#: test, so nothing is a magic number tuned to one cup.
OBJECT_SCALE_M = 0.10
#: Depth noise of the D405 on a matte surface at 0.2-0.3 m, one sigma.
DEPTH_NOISE_M = 0.0015
#: Voxel size the step clouds were downsampled at.
VOXEL_M = 0.003

#: Maturity gate. Below this the cloud is a stub: any primitive fitted to it
#: is a statement about the first two viewpoints, not about the object.
MIN_POINTS = 250
MIN_EXTENT_M = 0.02           # bbox diagonal; a 2 cm blob is not an object
MIN_BBOX_RANK = 2             # needs extent in at least 2 axes (not a line)

#: Normal estimation neighbourhood — 12 mm ~ 5 voxels, wide enough to be
#: stable, narrow enough that a 35 mm-radius cup wall still reads as curved.
NORMAL_RADIUS_M = 0.012
NORMAL_KNN = 30


def cloud_spacing(P: np.ndarray) -> float:
    """Median nearest-neighbour distance — the cloud's own length unit."""
    if len(P) < 2:
        return VOXEL_M
    d, _ = cKDTree(P).query(P, k=2)
    return float(max(np.median(d[:, 1]), 1e-6))


def estimate_normals(P: np.ndarray, radius: float = NORMAL_RADIUS_M,
                     knn: int = NORMAL_KNN) -> np.ndarray:
    """Per-point unit normals, oriented outward from the cloud centroid.

    Open3D's estimator is deterministic; the sign it returns is not, so we
    pin it with a rule (outward from the centroid) rather than with
    `orient_normals_consistent_tangent_plane`, whose result depends on the
    graph and would drift between steps. Sign only matters here for the
    signed tests in the planar guard, which re-derive it per patch anyway.
    """
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                             max_nn=knn))
    N = np.asarray(pc.normals).copy()
    out = P - P.mean(0)
    flip = np.einsum("ij,ij->i", N, out) < 0
    N[flip] *= -1
    return N


def maturity_reason(P: np.ndarray) -> str:
    """Empty string if the cloud is worth fitting, else why it is not.

    Abstaining is the point: a fitter that always answers is a fitter whose
    answer carries no information about whether the scan is done.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        return "input is not an (N,3) point array"
    if len(P) < MIN_POINTS:
        return f"cloud not ripe: {len(P)} points < {MIN_POINTS}"
    ext = P.max(0) - P.min(0)
    if float(np.linalg.norm(ext)) < MIN_EXTENT_M:
        return (f"cloud not ripe: bbox diagonal "
                f"{np.linalg.norm(ext) * 1000:.0f} mm < {MIN_EXTENT_M * 1000:.0f} mm")
    if int((ext > 2 * VOXEL_M).sum()) < MIN_BBOX_RANK:
        return "cloud not ripe: extent in fewer than 2 axes (a line, not a surface)"
    return ""


def fingerprint(params: Dict) -> str:
    """8 hex chars over every parameter that can change the answer."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def version_string(name: str, semver: str, params: Dict) -> str:
    return f"{name}/{semver}+p:{fingerprint(params)}"


# --- content-derived naming -----------------------------------------------
_CARDINALS = (("px", (1, 0, 0)), ("nx", (-1, 0, 0)), ("py", (0, 1, 0)),
              ("ny", (0, -1, 0)), ("pz", (0, 0, 1)), ("nz", (0, 0, -1)))
_CARDINAL_TOL_DEG = 25.0
_OBLIQUE_BIN_DEG = 30.0


def direction_name(v: Sequence[float]) -> str:
    """A stable name for a unit direction.

    Cardinal if it is within 25 deg of a base-frame axis, else a coarse
    (30 deg) elevation/azimuth bin. Coarse on purpose: it must survive the
    few degrees a fit wobbles between steps.
    """
    u = np.asarray(v, float)
    u = u / max(np.linalg.norm(u), 1e-12)
    best, bestc = None, -2.0
    for name, c in _CARDINALS:
        d = float(u @ np.asarray(c, float))
        if d > bestc:
            best, bestc = name, d
    if bestc > np.cos(np.deg2rad(_CARDINAL_TOL_DEG)):
        return best
    elev = np.degrees(np.arcsin(np.clip(u[2], -1, 1)))
    azim = np.degrees(np.arctan2(u[1], u[0])) % 360.0
    eb = int(round(elev / _OBLIQUE_BIN_DEG)) * int(_OBLIQUE_BIN_DEG)
    ab = int(round(azim / _OBLIQUE_BIN_DEG) * _OBLIQUE_BIN_DEG) % 360
    return f"e{eb:+03d}a{ab:03d}"


def axis_name(v: Sequence[float]) -> str:
    """Direction name for an *unsigned* axis (a cylinder axis has no head)."""
    u = np.asarray(v, float)
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return direction_name(u)


def disambiguate(labels: Iterable[str], keys: Iterable[float]) -> list:
    """Make labels unique without ever using extraction order.

    Colliding labels get a suffix built from a geometric key (a plane's
    offset, a cylinder's radius) rounded to 5 mm — still content-derived, so
    it is reproducible from the primitive alone.
    """
    labels, keys = list(labels), list(keys)
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    out = []
    for lb, k in zip(labels, keys):
        out.append(lb if counts[lb] == 1 else f"{lb}_{int(round(k * 200)) * 5:d}")
    # a pathological tie on the geometric key too: fall back to a stable
    # sort-free suffix on the key's full precision, still not an index.
    seen: Dict[str, int] = {}
    final = []
    for lb, k in zip(out, keys):
        if out.count(lb) > 1:
            lb = f"{lb}x{fingerprint({'k': round(float(k), 6)})[:4]}"
        while lb in seen:                     # last resort, still deterministic
            seen[lb] += 1
            lb = f"{lb}b"
        seen[lb] = 0
        final.append(lb)
    return final


def largest_component(P: np.ndarray, idx: np.ndarray, spacing: float,
                      link: float = 2.5) -> np.ndarray:
    """Keep only the biggest contiguous blob of `idx` — one surface, not two.

    Open3D's planar-patch boxes are fat enough to scoop points off the far
    wall of a cup; those points sit near the same plane and would otherwise
    make a curved patch look flat by cancellation.
    """
    from scipy.sparse.csgraph import connected_components
    if len(idx) < 3:
        return idx
    Q = P[idx]
    t = cKDTree(Q)
    n, lab = connected_components(t.sparse_distance_matrix(t, link * spacing),
                                  directed=False)
    if n <= 1:
        return idx
    return idx[lab == np.bincount(lab).argmax()]


def pca_frame(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, 3x3 axes as columns descending by variance, eigenvalues)."""
    c = Q.mean(0)
    _, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    lam = s ** 2 / max(len(Q), 1)
    return c, vt.T, lam
