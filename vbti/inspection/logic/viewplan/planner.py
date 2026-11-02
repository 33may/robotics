"""Mesh -> samples -> candidates -> measurability -> greedy -> planned views.

The hypothesis this package tests, as code. Every function is pure and
deterministic: the only randomness is `np.random.default_rng(seed)` in the
surface sampler, every `argmax` breaks ties on the lowest index, and Open3D's
own samplers are never called (they take no seed in 0.19 and
`o3d.utility.random.seed` is process-global state we refuse to depend on).
Running `plan_step` twice with the same seed produces byte-identical JSON —
there is a test that says so.

**The objective is orthogonality, not coverage.** The gates (standoff window,
incidence cutoff, field of view, self-occlusion, table) only decide which
(view, patch) pairs are *admissible*; the greedy then maximises

    gain(v) = SUM_s  area_s * max(0, q(v,s) - best_s),   q = cos^k(incidence)

so a view that sees a patch square-on outranks one that sees it at 55 deg
even though both "cover" it. That subordination — set cover as the pruning
applied to a ranked field, not as the ranking — is the answer to the pass-1
critique that a boolean gate makes orthogonality a floor.

Variants (roadmap section 1). `base` is the roadmap's primary `cos`; the name
`base` is kept because it is the CLI default:

    base       cos-weighted, Poisson, 60 deg gate   — the hypothesis
    binary     plain coverage, everything else equal — the ablation control
    cos2       cos^2 weighting                       — sharpness probe
    cos75      cos-weighted, 75 deg gate             — is the gate load-bearing
    altmesh    cos-weighted on an alpha-shape mesh   — is instability meshing
    continuity base + warm start from the previous step's plan — is the churn
               real disagreement or greedy tie-breaking (see `greedy`)

    from inspection.viewplan import planner
    plan, mesh, arrays = planner.plan_step(P, "2408-cup3", 10, "base", seed=0)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from inspection.fit.common import cloud_spacing as _spacing
from inspection.fit.common import maturity_reason
from inspection.viewplan import mesh as meshmod
from inspection.viewplan.contract import (PlannedView, StepPlan, abstained,
                                          fingerprint, look_at_cv)

DATA = Path(__file__).resolve().parents[2] / "data"

#: D405 colour intrinsics as recorded in every `runs/<run>/session.json`.
#: Distortion is ignored on purpose: the FOV test feeds a margin, not a
#: measurement, and the coefficients are < 6e-2.
DEFAULT_INTR = {"fx": 431.64349365234375, "fy": 431.2937927246094,
                "ppx": 417.2223205566406, "ppy": 237.80137634277344,
                "width": 848, "height": 480}

#: Physical constants of the sensor, not knobs. Sweeping them tests the
#: datasheet rather than the planner.
STANDOFF_WINDOW_M = (0.07, 0.20)
SWEET_SPOT_M = 0.10

#: The table guard, and the only guard (settled requirement 2): a candidate
#: camera must sit at least 10 mm above the table plane.
#:
#: The obvious companion test — "and it may not look up from under the
#: table" — is **provably redundant**, which is why it is measured rather
#: than enforced. The half-space above a plane is convex; every surface
#: sample is above the table (the recorder kept points >= 8 mm above it and
#: the 6 mm mesh trim cannot reach through); so a camera above the table
#: aimed at a sample above the table has its whole sight line above the
#: table, whatever its boresight does. Enforcing it anyway would cull every
#: legitimate view of an overhang. `MAX_BORESIGHT_UP` therefore only feeds
#: the `n_up_after_table` metric, which is the evidence for this paragraph.
TABLE_CLEARANCE_M = 0.01
MAX_BORESIGHT_UP = 0.26

#: The second guard, and the reason it is not a violation of settled
#: requirement 2: a camera is a body, not a point. A candidate offset one
#: standoff along a cavity-wall normal can land *inside* the cup — measured on
#: the shipped `base` grid, 6 of 1090 mid-run views had the camera closer than
#: 30 mm to the accumulated cloud and one (2408-cup3 step 011 view 9) sat
#: 0.6 mm from it, inside the cloud's own bounding box, claiming 237 samples
#: through the open side of the Poisson sheet. Those poses are not "views the
#: robot cannot reach" (which we deliberately do not model) — they are poses
#: no rigid body can occupy, and they inflate mid-run coverage and churn the
#: stability metric. 30 mm is the D405's half-body (42 x 42 x 23 mm), i.e. the
#: distance from the optical centre to the far corner of its own case.
CAMERA_CLEARANCE_M = 0.03

#: Gate codes, ordered along the test chain, so "how far did this pair get"
#: is `max(code)` and the achievability breakdown needs no extra bookkeeping.
GATE_PASS, GATE_STANDOFF, GATE_INCIDENCE, GATE_FOV, GATE_OCCLUSION = 0, 1, 2, 3, 4
GATE_NAMES = {GATE_STANDOFF: "standoff", GATE_INCIDENCE: "incidence",
              GATE_FOV: "fov", GATE_OCCLUSION: "occlusion"}

_BASE = {
    "mesher": "poisson",
    "standoff_m": SWEET_SPOT_M,
    "standoff_window_m": list(STANDOFF_WINDOW_M),
    "incidence_cutoff_deg": 60.0,
    "n_samples": 3000,
    "n_candidates": 300,
    "occlusion_tol_m": 0.004,
    "weight": "cosine",
    "max_views": 12,
    "min_gain_frac": 0.01,
    "fov_margin_px": 20,
    "clearance_m": CAMERA_CLEARANCE_M,
    "continuity_bonus": 0.0,
    "continuity_radius_m": 0.05,
}


def _variant(**over) -> Dict[str, Any]:
    p = dict(_BASE)
    p.update(over)
    return p


VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": _variant(),
    "binary": _variant(weight="binary"),
    "cos2": _variant(weight="cosine2"),
    "cos75": _variant(incidence_cutoff_deg=75.0),
    "altmesh": _variant(mesher="alpha"),
    # 0.30 measured, not guessed. Sweeping the bonus over all three runs
    # (mean M5 churn of the last 10 steps / share of those steps at or under
    # the 0.02 stability threshold / final coverage@60):
    #
    #   bonus  cup3            cup4            cup5            cov60 cost
    #   0.00   0.071  0%       0.066  0%       0.074  0%       —  (the shipped
    #   0.05   0.032 20%       0.031 20%       0.046  0%          behaviour)
    #   0.15   0.025 30%       0.020 60%       0.021 60%       ~0.3 pp
    #   0.30   0.013 80%       0.007 100%      0.015 90%       ~1.0 pp
    #   0.60   0.003 100%      0.003 100%      0.005 100%      ~1.5 pp
    #
    # 0.60 freezes the plan outright, which would make the convergence
    # measurement circular — a plan that cannot move always looks converged.
    # 0.30 crosses the threshold on every run while still moving when the
    # geometry does (worst single step in the last 10: 0.012-0.024 m).
    "continuity": _variant(continuity_bonus=0.30),
}


def variants() -> List[str]:
    return sorted(VARIANTS)


def params_for(variant: str) -> Dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; have {variants()}")
    return json.loads(json.dumps(VARIANTS[variant]))       # deep, plain copy


def version_of(variant: str) -> str:
    return fingerprint(variant, params_for(variant))


# --- run-local data the planner is allowed to read -------------------------
def run_dir(run: str, data: Path = DATA) -> Optional[Path]:
    """Resolve a run id across BOTH archives — runs/ first, datasets/ second.

    The box sweeps are collection runs under data/datasets/ in the NATIVE
    record layout; reading data/runs/ only made the planner silently degrade
    on them (default intrinsics + centroid-rule normals, 2026-09-10). Same
    resolution order as `record/__main__._run_dir`.
    """
    for root in (Path(data) / "runs", Path(data) / "datasets"):
        if (root / run / "run.json").exists():
            return root / run
    return None


def _intrinsics_with_source(run: str, data: Path = DATA
                            ) -> Tuple[Dict[str, float], str]:
    """(colour intrinsics, source) — read through the record layer's `Run`.

    Fallback ladder (2026-09-10, after the box sweep silently planned 9 real
    runs on fixture fallbacks for three days):
    - run not on disk (synthetic/test id)   -> D405 default, source declared
    - run on disk, rig=fake, no session     -> D405 default, source declared
    - run on disk, rig=real, unreadable     -> RuntimeError. A named REAL run
      whose lookup misses must scream, never quietly default — the same
      principle as a survey refusal aborting the sweep loudly.
    """
    d = run_dir(run, data)
    if d is None:
        return dict(DEFAULT_INTR), "default(run-not-on-disk)"
    from inspection.record.run import Run
    try:
        r = Run.load(d)
        s = r.session
        intr = None if s is None else s.intrinsics.get("color")
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(f"{run}: run exists at {d} but its record is "
                           f"unreadable ({type(exc).__name__}: {exc})") from exc
    if intr is None:
        if r.record.rig == "real":
            raise RuntimeError(
                f"{run}: real-rig run without readable colour intrinsics in "
                f"session.json — refusing the D405 default silently")
        return dict(DEFAULT_INTR), "default(fake-rig)"
    return ({k: float(getattr(intr, k)) for k in ("fx", "fy", "ppx", "ppy")}
            | {"width": int(intr.width), "height": int(intr.height)},
            "session")


def session_intrinsics(run: str, data: Path = DATA) -> Dict[str, float]:
    """Colour intrinsics of a recorded run — `_intrinsics_with_source` sans
    the source. Same loud ladder: raises on a broken real-rig run."""
    return _intrinsics_with_source(run, data)[0]


def recorded_cams(run: str, upto_step: Optional[int] = None,
                  data: Path = DATA) -> np.ndarray:
    """Camera positions of steps 0..k, for the normal-orientation rule.

    Through `Run`, indexed by step_id, never by `range()`: 2408-cup5 has no
    `029/` at all while its derived step_030.ply exists, and any code that
    counts entries gets that run wrong. Rejected steps carry no pose and are
    skipped. Returns an empty (0,3) array when the run is unavailable — the
    caller then falls back to the centroid rule rather than failing.
    """
    d = run_dir(run, data)
    if d is None:
        return np.empty((0, 3))
    from inspection.record.run import Run
    try:
        steps = Run.load(d).steps
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(f"{run}: run exists at {d} but its record is "
                           f"unreadable ({type(exc).__name__}: {exc})") from exc
    out = [s.T_base_cam[:3, 3] for s in steps
           if s.T_base_cam is not None
           and (upto_step is None or s.record.step_id <= int(upto_step))]
    return np.asarray(out, dtype=float).reshape(-1, 3)


# --- surface sampling ------------------------------------------------------
def sample_surface(m, n: int, seed: int, cloud: Optional[np.ndarray] = None,
                   cloud_normals: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """(points, normals, area_per_sample) — area-uniform, seeded, ours.

    Area-uniform means every sample carries the same area share, which is
    what makes `area_m2 = area_per_sample * n_covered` true and lets the npz
    store one scalar instead of a per-sample weight.

    **The normal is the nearest oriented cloud normal, not the triangle's.**
    That is not a shortcut, it is the more accurate estimate, and it was
    measured both ways. A Poisson triangle spans ~1 mm on a cloud carrying
    1.5-3 mm of noise, so its normal inherits that noise wholesale; the cloud
    normal is a PCA fit over a 12 mm neighbourhood. On a synthetic sphere
    (truth known) the mean angular error is 15.5 deg for triangle normals,
    10.8 deg for area-averaged vertex normals and **6.6 deg** for the
    transferred cloud normal. On 2408-cup3 step 029 the same swap moves
    coverage@60 from 0.728 to 0.764 and mean achieved incidence from 30.0 to
    28.1 deg. Since these normals define both the candidate boresights and
    the incidence the objective ranks, the noisy version is not merely
    inelegant — it plans genuinely worse views.

    The mesh's own normal (vertex-interpolated, sign-transferred) is the
    fallback where the nearest cloud point is further than `3 x spacing`,
    which on the real runs never happens (max sample-to-cloud distance 7.0 mm
    at 2.5 mm spacing) but does on a convex-hull fallback mesh, whose facets
    bridge whatever gap they like. Winding is never trusted: an early-step
    Poisson sheet can come out with negative signed volume and the alpha
    fallback is arbitrary, so every sign comes from the cloud.
    """
    V = np.asarray(m.vertices, dtype=float)
    F = np.asarray(m.triangles, dtype=np.int64)
    if len(F) == 0:
        raise ValueError("cannot sample an empty mesh")
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(tri_area.sum())
    if total <= 0:
        raise ValueError("mesh has zero area")

    rng = np.random.default_rng(int(seed))
    cum = np.cumsum(tri_area)
    idx = np.searchsorted(cum, rng.random(int(n)) * total, side="right")
    idx = np.clip(idx, 0, len(F) - 1)
    u = rng.random(int(n))
    v = rng.random(int(n))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    P = (a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx]))

    # mesh-side normal: area-averaged vertex normals, barycentrically blended
    # (smoother than the flat triangle, and the only option where the cloud
    # is too far away to speak)
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    VN = np.asarray(m.vertex_normals, dtype=float)
    w = np.stack([1.0 - u - v, u, v], axis=1)
    N = (w[:, 0:1] * VN[F[idx, 0]] + w[:, 1:2] * VN[F[idx, 1]]
         + w[:, 2:3] * VN[F[idx, 2]])
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    bad = (ln < 1e-12).ravel()
    N = N / np.where(ln < 1e-12, 1.0, ln)
    if bad.any():                                   # degenerate vertex normals
        fb = cross[idx][bad]
        N[bad] = fb / np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-15)

    if cloud is not None and cloud_normals is not None and len(cloud):
        C = np.asarray(cloud, float)
        CN = np.asarray(cloud_normals, float)
        d, nn = cKDTree(C).query(P)
        ref = CN[nn]
        N[np.einsum("ij,ij->i", N, ref) < 0] *= -1  # sign always from the cloud
        near = d <= 3.0 * _spacing(C)
        N[near] = ref[near]                         # direction too, where local
    else:
        ref = P - V.mean(0)
        N[np.einsum("ij,ij->i", N, ref) < 0] *= -1
    return P, N, total / float(n)


# --- candidate cameras -----------------------------------------------------
def farthest_points(P: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point subsample; seed = point nearest centroid."""
    k = int(min(k, len(P)))
    d = np.linalg.norm(P - P.mean(0), axis=1)
    first = int(np.argmin(d))
    picked = [first]
    dist = np.linalg.norm(P - P[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        picked.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(P - P[nxt], axis=1))
    return np.asarray(picked, dtype=int)


def clearance_to(cloud: Optional[np.ndarray], pos: np.ndarray) -> np.ndarray:
    """Distance from each camera position to the nearest cloud point.

    `inf` where there is no cloud to measure against, so a caller that never
    got one keeps the old behaviour instead of silently culling everything.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    if cloud is None or not len(cloud):
        return np.full(len(pos), np.inf)
    d, _ = cKDTree(np.asarray(cloud, dtype=float).reshape(-1, 3)).query(pos)
    return np.asarray(d, dtype=float)


def candidate_views(P: np.ndarray, N: np.ndarray, params: Dict[str, Any],
                    table_plane: Sequence[float], *,
                    cloud: Optional[np.ndarray] = None,
                    extra_T: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(T (M,4,4), anchor_ids (M,), info) — one orthogonal view per anchor.

    A candidate is the *ideal* view of its anchor patch: the camera sits one
    standoff out along the surface normal and looks straight back down it, so
    the incidence angle at that patch is 0 by construction. Everything the
    greedy later does is choosing among these ideals; the measurability
    matrix is what says which other patches each one happens to also serve.

    Two guards, and only two (settled requirement 2 still holds: no IK, no
    reachability, no joint limits). A candidate is dropped if its camera
    falls on or below the table plane, or if the camera *body* would be
    inside the object — `CAMERA_CLEARANCE_M` from the input cloud. The
    "looking up from under the table" half of the table requirement needs no
    test of its own; see `TABLE_CLEARANCE_M`.

    `extra_T` are poses to consider alongside the freshly generated ones —
    the previous step's plan, for the `continuity` variant. They face the
    same two guards (a pose that was legal against 1.1k points can be inside
    the object against 4k of them) and carry anchor id -1, since they belong
    to no sample of *this* step's surface.
    """
    d = float(params["standoff_m"])
    pos = P + d * N
    bore = -N
    h = meshmod.height_above(table_plane, pos)
    under = h <= TABLE_CLEARANCE_M
    clear = float(params.get("clearance_m", CAMERA_CLEARANCE_M))
    inside = (clearance_to(cloud, pos) < clear) & ~under
    keep = ~under & ~inside

    anchors = np.flatnonzero(keep)
    if len(anchors) > int(params["n_candidates"]):
        anchors = anchors[farthest_points(P[anchors], int(params["n_candidates"]))]
        anchors = np.sort(anchors)
    T = np.stack([look_at_cv(pos[i], bore[i]) for i in anchors]) if len(anchors) \
        else np.empty((0, 4, 4))

    n_extra = 0
    if extra_T is not None and len(extra_T):
        E = np.asarray(extra_T, dtype=float).reshape(-1, 4, 4)
        epos = E[:, :3, 3]
        ok = (meshmod.height_above(table_plane, epos) > TABLE_CLEARANCE_M) & \
             (clearance_to(cloud, epos) >= clear)
        E = E[ok]
        n_extra = int(len(E))
        if n_extra:
            T = np.concatenate([T, E]) if len(T) else E
            anchors = np.concatenate([anchors, np.full(n_extra, -1, dtype=int)])

    info = {"n_anchors_legal": int(keep.sum()),
            "n_culled_table": int(under.sum()),
            "n_culled_clearance": int(inside.sum()),
            "n_up_after_table": int((bore[keep, 2] > MAX_BORESIGHT_UP).sum()),
            "n_warm_start": n_extra,
            "n_candidates": int(len(T))}
    return T, anchors, info


# --- measurability ---------------------------------------------------------
def _project(T: np.ndarray, P: np.ndarray, intr: Dict[str, float]):
    """(u, v, z_cam) for every (view, sample) pair — pinhole, OpenCV axes.

    `cell/geometry.py:project_to_pixels` is the same four lines but it drops
    points behind the lens, which destroys the row/column correspondence a
    matrix needs. Same convention, same intrinsics, index-preserving.
    """
    R = T[:, :3, :3]
    D = P[None, :, :] - T[:, None, :3, 3]
    pc = np.einsum("mij,msi->msj", R, D)
    z = pc[:, :, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = intr["ppx"] + intr["fx"] * pc[:, :, 0] / safe
    v = intr["ppy"] + intr["fy"] * pc[:, :, 1] / safe
    return u, v, z


def visibility(m, T: np.ndarray, P: np.ndarray, N: np.ndarray,
               intr: Dict[str, float], params: Dict[str, Any],
               *, cutoff_deg: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cover (M,S) bool, cosv (M,S) f64, gate (M,S) int8).

    A view measures a sample iff it passes, in order: the standoff window,
    the incidence cutoff, the field of view with a margin, and a
    self-occlusion raycast. `gate` records how far each failing pair got
    (`GATE_*`), which is the whole achievability decomposition — the cup's
    table-contact ring and its cavity are then *measured* rather than
    argued about.

    `cosv` is the incidence cosine where covered and 0 elsewhere, so a
    selector can weight by orthogonality without re-deriving angles.
    """
    import open3d as o3d

    M, S = len(T), len(P)
    if M == 0 or S == 0:
        return (np.zeros((M, S), bool), np.zeros((M, S)), np.zeros((M, S), np.int8))
    lo, hi = [float(x) for x in params["standoff_window_m"]]
    cut = float(params["incidence_cutoff_deg"] if cutoff_deg is None else cutoff_deg)
    cos_cut = float(np.cos(np.deg2rad(cut)))
    margin = float(params["fov_margin_px"])
    W, H = float(intr["width"]), float(intr["height"])

    cam = T[:, :3, 3]
    vec = cam[:, None, :] - P[None, :, :]                    # sample -> camera
    dist = np.linalg.norm(vec, axis=2)
    safe = np.where(dist < 1e-12, 1e-12, dist)
    cosv = np.einsum("msi,si->ms", vec, N) / safe

    gate = np.zeros((M, S), dtype=np.int8)
    ok = (dist >= lo) & (dist <= hi)
    gate[~ok] = GATE_STANDOFF
    ok &= cosv >= cos_cut
    gate[(gate == GATE_PASS) & ~ok] = GATE_INCIDENCE
    u, v, z = _project(T, P, intr)
    infov = (z > 1e-3) & (u >= margin) & (u <= W - margin) & \
            (v >= margin) & (v <= H - margin)
    ok &= infov
    gate[(gate == GATE_PASS) & ~ok] = GATE_FOV

    rows, cols = np.nonzero(ok)
    if len(rows):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        tol = float(params["occlusion_tol_m"])
        seen = np.empty(len(rows), dtype=bool)
        for s0 in range(0, len(rows), 500_000):
            sl = slice(s0, s0 + 500_000)
            r, c = rows[sl], cols[sl]
            d = safe[r, c]
            # camera -> sample, never sample -> camera: the sample lies ON
            # the mesh, so a ray leaving it self-hits at t=0 and every pair
            # reads as occluded.
            rays = np.concatenate([cam[r], -vec[r, c] / d[:, None]], axis=1)
            t_hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32))
                                    )["t_hit"].numpy()
            seen[sl] = t_hit >= d - tol
        ok[rows, cols] = seen
        blocked = np.zeros((M, S), dtype=bool)
        blocked[rows[~seen], cols[~seen]] = True
        gate[blocked] = GATE_OCCLUSION

    cover = ok
    cosv = np.where(cover, np.clip(cosv, -1.0, 1.0), 0.0)
    return cover, cosv, gate


# --- selection -------------------------------------------------------------
def quality(cover: np.ndarray, cosv: np.ndarray, weight: str) -> np.ndarray:
    """The field the greedy ranks. Coverage is the pruning, this is the rank."""
    if weight == "binary":
        return cover.astype(float)
    if weight == "cosine":
        return cosv
    if weight == "cosine2":
        return cosv ** 2
    raise ValueError(f"unknown weight {weight!r}")


def pose_affinity(T: np.ndarray, prev_T: Optional[np.ndarray], radius_m: float,
                  lambda_rot_m_per_rad: float = 0.1) -> np.ndarray:
    """(M,) in [0,1]: how close each candidate is to *some* previous pick.

    The distance is the same one the M5 stability metric uses — metres of
    translation plus `lambda` metres per radian of boresight rotation — so
    the thing the greedy is nudged toward is exactly the thing the stability
    curve measures. A hat kernel (`1 - d/r`, clipped) rather than a Gaussian:
    it reaches zero at the radius, so a candidate on the far side of the
    object provably gets no bonus at all.
    """
    M = len(T)
    if prev_T is None or not len(prev_T) or M == 0 or radius_m <= 0:
        return np.zeros(M)
    A = np.asarray(T, dtype=float).reshape(-1, 4, 4)
    B = np.asarray(prev_T, dtype=float).reshape(-1, 4, 4)
    d_pos = np.linalg.norm(A[:, None, :3, 3] - B[None, :, :3, 3], axis=2)
    cos = np.clip(np.einsum("ik,jk->ij", A[:, :3, 2], B[:, :3, 2]), -1.0, 1.0)
    d = d_pos + float(lambda_rot_m_per_rad) * np.arccos(cos)
    return np.clip(1.0 - d.min(1) / float(radius_m), 0.0, 1.0)


def greedy(cover: np.ndarray, cosv: np.ndarray, area: float,
           params: Dict[str, Any], *, affinity: Optional[np.ndarray] = None
           ) -> Tuple[List[Dict[str, Any]], str]:
    """Orthogonality-weighted greedy set cover. Returns (picks, terminator).

    Stops on the 12-view cap or when the best remaining marginal gain falls
    below `min_gain_frac` of the *coverable* area — coverable, not total,
    because on a cup 17-27 % of the mesh is unviewable from any legal pose
    and a threshold against the total would never fire. Which rule fired is
    returned, because "hit the cap" and "ran out of gain" are different
    answers to the question the POC asks.

    `affinity` (the `continuity` variant) breaks near-ties toward the
    previous step's plan: the *ranking* score is
    `gain * (1 + continuity_bonus * affinity)` while the *acceptance* test
    (`> 0`, `>= floor`) stays on the true gain, so continuity can reorder
    equals but can never buy a view that earns nothing. The measured problem
    it attacks: with hundreds of candidates on a smooth incidence field the
    top gains sit within a few per cent of each other, so the argmax swaps
    between steps long after coverage and incidence have converged, and the
    M5 churn plateaus at 3-5x the 20 mm stability threshold.
    """
    M, S = cover.shape
    Q = quality(cover, cosv, str(params["weight"]))
    best = np.zeros(S)
    coverable = float(cover.any(0).sum()) * area
    floor = float(params["min_gain_frac"]) * coverable
    bonus = float(params.get("continuity_bonus", 0.0))
    if affinity is None or bonus <= 0:
        rank_w = np.ones(M)
    else:
        rank_w = 1.0 + bonus * np.asarray(affinity, dtype=float).reshape(M)
    picks: List[Dict[str, Any]] = []
    used = np.zeros(M, dtype=bool)
    term = "gain"
    while len(picks) < int(params["max_views"]):
        gains = (np.maximum(Q - best[None, :], 0.0).sum(1)) * area
        ranked = gains * rank_w
        ranked[used] = -1.0
        v = int(np.argmax(ranked))                      # ties -> lowest index
        if gains[v] <= 0 or gains[v] < floor:
            break
        used[v] = True
        best = np.maximum(best, Q[v])
        picks.append({"view_idx": v, "gain": float(gains[v]),
                      "covered_ids": np.flatnonzero(cover[v]),
                      "best_after": best.copy()})
    else:
        term = "max_views"
    return picks, term


# --- composition -----------------------------------------------------------
def _angle_stats(cos: np.ndarray) -> Dict[str, float]:
    if not len(cos):
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    th = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {"mean": float(th.mean()), "median": float(np.median(th)),
            "p95": float(np.percentile(th, 95)), "max": float(th.max())}


def plan_step(P_cloud: np.ndarray, run: str, step: int, variant: str = "base",
              seed: int = 0, *, intr: Optional[Dict[str, float]] = None,
              cams: Optional[np.ndarray] = None, data: Path = DATA,
              table_plane: Optional[Sequence[float]] = None,
              prev_T: Optional[np.ndarray] = None
              ) -> Tuple[StepPlan, Any, Dict[str, np.ndarray]]:
    """The whole pipeline for one (run, step, variant). Never raises.

    Anything that goes wrong inside becomes an abstain naming the exception,
    exactly like `fit/bank.py:fit` — a truthful row in the score log beats a
    hole in it, and a step that cannot be planned is itself the answer to
    "after which step is the geometry good enough".

    Returns `(StepPlan, mesh, arrays)`; `arrays` is the npz sidecar
    (points / normals / area / best_cos / covered_by) and is empty on abstain.

    `table_plane` overrides `mesh.table_plane`, whose derivation assumes the
    cloud is an object-only crop of something standing ON the table. Tests
    that float a synthetic shape in mid-air must say where the table is;
    real runs never pass it.

    `prev_T` is the previous step's planned poses, (k,4,4) — the warm start
    the `continuity` variant needs. It is ignored (harmlessly) by every
    variant whose `continuity_bonus` is 0, which is every other variant.
    """
    params = params_for(variant)
    P = np.asarray(P_cloud, dtype=float)
    t0 = time.perf_counter()
    why = maturity_reason(P)
    if why:
        return abstained(run, step, variant, params, seed, why), None, {}

    secs: Dict[str, float] = {}
    intr_source = "caller" if intr else None
    try:
        # Inside the guard ON PURPOSE: a broken real run now becomes a
        # truthful abstain row naming the RuntimeError (loud in every score
        # log), not a silent D405/centroid default (2026-09-10).
        if intr:
            intr = dict(intr)
        else:
            intr, intr_source = _intrinsics_with_source(run, data)
        if cams is None:
            cams = recorded_cams(run, step, data)
        t = time.perf_counter()
        cloud_normals, nrule = meshmod.oriented_normals(P, cams)
        m, minfo = meshmod.reconstruct(P, seed=seed, cams=cams,
                                       method=str(params["mesher"]))
        secs["mesh"] = time.perf_counter() - t

        plane = ([float(x) for x in table_plane] if table_plane is not None
                 else meshmod.table_plane(P))

        t = time.perf_counter()
        S, SN, area = sample_surface(m, int(params["n_samples"]), seed,
                                     cloud=P, cloud_normals=cloud_normals)
        secs["sample"] = time.perf_counter() - t

        t = time.perf_counter()
        warm = (np.asarray(prev_T, dtype=float).reshape(-1, 4, 4)
                if prev_T is not None and len(prev_T)
                and float(params.get("continuity_bonus", 0.0)) > 0 else None)
        T, anchors, cinfo = candidate_views(S, SN, params, plane, cloud=P,
                                            extra_T=warm)
        secs["candidates"] = time.perf_counter() - t
        if not len(T):
            return (abstained(run, step, variant, params, seed,
                              "every candidate view fell under the table or "
                              "inside the object"),
                    m, {})

        t = time.perf_counter()
        cover, cosv, gate = visibility(m, T, S, SN, intr, params)
        secs["visibility"] = time.perf_counter() - t

        t = time.perf_counter()
        aff = pose_affinity(T, warm, float(params.get("continuity_radius_m", 0.0)))
        picks, term = greedy(cover, cosv, area, params, affinity=aff)
        secs["greedy"] = time.perf_counter() - t
    except Exception as exc:                                  # noqa: BLE001
        return (abstained(run, step, variant, params, seed,
                          f"planner raised {type(exc).__name__}: {exc}"),
                None, {})

    n_s = len(S)
    total_area = area * n_s
    cutoff = float(params["incidence_cutoff_deg"])

    # achieved incidence over the SELECTED views, gate-by-gate honest
    sel = np.array([p["view_idx"] for p in picks], dtype=int)
    best_cos = np.zeros(n_s)
    covered_by = np.full(n_s, -1, dtype=np.int32)
    for k, p in enumerate(picks):
        c = cosv[p["view_idx"]]
        better = c > best_cos
        best_cos = np.where(better, c, best_cos)
        covered_by[better] = k
    covered = best_cos > 0

    # the same selected views with the incidence gate lifted, so coverage at
    # 45/60/75 deg is one measurement instead of three planning runs
    if len(sel):
        _cov_o, cos_open, _g = visibility(m, T[sel], S, SN, intr, params,
                                          cutoff_deg=90.0)
        best_open = cos_open.max(0)
    else:
        best_open = np.zeros(n_s)

    def frac_at(deg: float) -> float:
        return float((best_open >= np.cos(np.deg2rad(deg))).sum()) / max(n_s, 1)

    any_cov = cover.any(0)
    coverable_area = float(any_cov.sum()) * area
    causes = {}
    if (~any_cov).any():
        worst = gate[:, ~any_cov].max(0)
        for code, name in GATE_NAMES.items():
            causes[f"unachievable_{name}"] = float((worst == code).sum()) / max(n_s, 1)
    stats = _angle_stats(best_cos[covered])

    views: List[PlannedView] = []
    cum = 0.0
    for k, p in enumerate(picks):
        ids = p["covered_ids"]
        c = cosv[p["view_idx"], ids]
        cum = float((p["best_after"] > 0).sum()) * area / max(total_area, 1e-12)
        vs = _angle_stats(c)
        views.append(PlannedView(
            view_id=k, T_base_cam=T[p["view_idx"]],
            standoff_m=float(params["standoff_m"]), gain=p["gain"],
            covered=ids.tolist(), area_m2=float(len(ids)) * area,
            mean_incidence_deg=vs["mean"], p95_incidence_deg=vs["p95"],
            cum_area_frac=cum))

    secs["total"] = time.perf_counter() - t0
    metrics = {
        "n_samples": int(n_s),
        "n_views": len(views),
        "terminator": term,
        "coverage_area_frac": float(covered.sum()) / max(n_s, 1),
        "coverage_coverable_frac": (float(covered.sum()) * area /
                                    max(coverable_area, 1e-12)),
        "coverage_45": frac_at(45.0), "coverage_60": frac_at(60.0),
        "coverage_75": frac_at(75.0),
        "coverable_area_frac": float(any_cov.sum()) / max(n_s, 1),
        "weighted_coverage": float(best_cos.sum()) / max(n_s, 1),
        "mean_best_incidence_deg": stats["mean"],
        "median_best_incidence_deg": stats["median"],
        "p95_best_incidence_deg": stats["p95"],
        "max_best_incidence_deg": stats["max"],
        "uncovered_area_m2": float((~covered).sum()) * area,
        "incidence_cutoff_deg": cutoff,
        # the clearance guard, measured on what was actually selected, so a
        # regression shows up in scores.jsonl and not only in a render
        "min_cam_clearance_m": (float(clearance_to(P, T[sel][:, :3, 3]).min())
                                if len(sel) else 0.0),
        "n_views_from_warm_start": (int((anchors[sel] < 0).sum())
                                    if len(sel) else 0),
        "mesh_area_ratio_cloud": float(minfo["area_ratio_cloud"]),
        "n_culled_unreached": int((~cover.any(1)).sum()),
        "mesh_area_m2": float(minfo["area_m2"]),
        # Input provenance — what the plan was ACTUALLY computed from, so a
        # fixture fallback firing on a real run can never hide again.
        "intrinsics_source": intr_source,
        "normals_rule": nrule,
        "n_recorded_cams": int(len(cams)) if cams is not None else 0,
        "seconds": round(secs["total"], 3),
        "seconds_split": {k: round(v, 3) for k, v in secs.items()},
    }
    metrics.update(cinfo)
    metrics.update(causes)

    plan = StepPlan(
        run=run, step=int(step), variant=variant,
        version=fingerprint(variant, params), seed=int(seed),
        table_plane=plane,
        mesh={"path": f"meshes/step_{int(step):03d}.ply", **minfo},
        samples={"path": f"samples/step_{int(step):03d}.npz", "count": int(n_s),
                 "area_per_sample_m2": float(area)},
        params=params, views=views, metrics=metrics)
    arrays = {"points": S, "normals": SN,
              "area": np.full(n_s, area), "best_cos": best_cos,
              "covered_by": covered_by}
    return plan, m, arrays


# --- cross-evaluation ------------------------------------------------------
def score_views(m, T: np.ndarray, P: np.ndarray, N: np.ndarray,
                intr: Dict[str, float], params: Dict[str, Any]
                ) -> Dict[str, float]:
    """Score an *arbitrary* set of poses against an *arbitrary* surface.

    The half of the POC's core question that `plan_step` structurally cannot
    answer. `plan_step` scores the step-k plan on the step-k mesh, and that
    mesh grows 2.8-3.4x over a run, so its coverage *falls* from a trivial
    0.99 on a 1.1k-point blob to 0.63-0.76 on the finished surface. No
    threshold crossing can be read off a curve whose denominator moves.

    Feeding the step-k poses and the FINAL step's mesh + samples to this
    function fixes the denominator: every step is then measured on the same
    surface with the same yardstick, the curve rises monotonically, and
    "after which step is the geometry good enough" becomes a number.

    `params` is the yardstick (gates + intrinsics), not the plan's own
    parameters: comparing a `cos75` plan against a 60 deg reference is the
    point, not a bug — every variant is measured with one ruler.
    """
    T = np.asarray(T, dtype=float).reshape(-1, 4, 4)
    n = len(P)
    if not len(T) or not n:
        return {"coverage": 0.0, "mean_incidence_deg": 0.0,
                "p95_incidence_deg": 0.0, "weighted_coverage": 0.0,
                "n_views": int(len(T))}
    cover, cosv, _gate = visibility(m, T, P, N, intr, params)
    covered = cover.any(0)
    best = cosv.max(0)
    stats = _angle_stats(best[covered])
    return {"coverage": float(covered.sum()) / n,
            "mean_incidence_deg": stats["mean"],
            "p95_incidence_deg": stats["p95"],
            "weighted_coverage": float(best.sum()) / n,
            "n_views": int(len(T))}


def save_arrays(arrays: Dict[str, np.ndarray], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
