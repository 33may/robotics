"""`ems` — probabilistic superquadric recovery (Liu et al., CVPR 2022).

Thin wrapper over the vendored upstream implementation
(`inspection/fit/vendor/ems`, MIT — see its ATTRIBUTION.md). We contribute
nothing to the algorithm; the work here is the contract mapping, the
maturity gate, and a sanity check that catches EMS's two known ways of
going wrong on a partial cloud.

**It is deterministic.** EMS has no RNG: initialisation is an eigen-analysis
of the cloud and the search is a bounded trust-region solve. Verified by
running the same cloud twice on all three cups — identical to the last bit.
`seed` is therefore accepted and ignored, which the version string records.

**Known failure modes, both observed on our cups.**

- `scipy.optimize.least_squares` raises `ValueError: Initial guess is
  outside of provided bounds` on some small/anisotropic clouds (seen at
  N=50 on 2408-cup5 and N=200 on 2408-cup4). Caught, turned into an abstain.
- `eps2` drifts towards 2.0, which is the pinched, star-shaped corner of
  the superquadric family and never what a mug looks like — 2408-cup4's
  final cloud recovers eps=[0.37, 1.89] with a 129 mm axis on a 93 mm
  cloud. That is a real answer from the method, so it is emitted, not
  suppressed; the degeneracy is named in `reason` and the uniform score
  will show it.

`scale` is stored as SEMI-axes, matching both EMS and the contract's
`superquadric: scale[3], eps[2]`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from inspection.fit import common
from inspection.fit.contract import (Decomposition, Primitive, abstained,
                                     pose_from)

NAME = "ems"
SEMVER = "1.0"

PARAMS = dict(
    outlier_ratio=0.1,
    max_iteration_em=20,
    tolerance_em=1e-3,
    relative_tolerance_em=1e-1,
    max_opti_iterations=3,
    max_switch=2,
    adaptive_upper_bound=True,   # bounds the recovered volume by the cloud's
    rescale=True,                # upstream default; our clouds are ~0.1 m
    # sanity: a superquadric more than 3x the cloud's own half-extent is not
    # a description of the cloud, it is a solver that ran away.
    max_scale_ratio=3.0,
    min_scale_m=0.002,
    # eps outside this is the degenerate corner of the family; still emitted,
    # but flagged in `reason`.
    eps_degenerate_above=1.6,
    seed_is_ignored=True,
)
VERSION = common.version_string(NAME, SEMVER, PARAMS)


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

    from inspection.fit.vendor.ems.EMS_recovery import EMS_recovery
    try:
        sq, _p = EMS_recovery(
            P, OutlierRatio=prm["outlier_ratio"],
            MaxIterationEM=prm["max_iteration_em"],
            ToleranceEM=prm["tolerance_em"],
            RelativeToleranceEM=prm["relative_tolerance_em"],
            MaxOptiIterations=prm["max_opti_iterations"],
            MaxiSwitch=prm["max_switch"],
            AdaptiveUpperBound=prm["adaptive_upper_bound"],
            Rescale=prm["rescale"])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return abstained(NAME, version, f"EMS solver failed: {exc}")

    scale = np.asarray(sq.scale, float)
    eps = np.asarray(sq.shape, float)
    t = np.asarray(sq.translation, float)
    R = np.asarray(sq.RotM, float)
    if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(eps))
            and np.all(np.isfinite(t)) and np.all(np.isfinite(R))):
        return abstained(NAME, version, "EMS returned non-finite parameters")

    half = (P.max(0) - P.min(0)) / 2.0
    if scale.max() > prm["max_scale_ratio"] * max(half.max(), 1e-6):
        return abstained(
            NAME, version,
            f"recovered superquadric {scale.max() * 1000:.0f} mm exceeds "
            f"{prm['max_scale_ratio']:.0f}x the cloud half-extent "
            f"{half.max() * 1000:.0f} mm")
    if scale.min() < prm["min_scale_m"]:
        return abstained(NAME, version,
                         f"degenerate scale {scale.min() * 1000:.1f} mm")

    note = "superquadric recovered"
    if float(eps.max()) > prm["eps_degenerate_above"]:
        note = (f"recovered, but eps={np.round(eps, 2).tolist()} is in the "
                f"pinched corner of the superquadric family")

    prim = Primitive(kind="superquadric", pose=pose_from(R, t),
                     params={"scale": scale.tolist(), "eps": eps.tolist()},
                     label="hull", inliers=[], residual_mm=0.0,
                     support=0)
    return Decomposition(method=NAME, version=version, primitives=[prim],
                         reason=note)
