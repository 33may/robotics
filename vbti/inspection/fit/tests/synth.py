"""Synthetic clouds with known answers — the only ground truth we have.

Every generator takes a seed and returns `(N,3)` metres at roughly the scale
and noise of a real step cloud: ~0.1 m objects, 1.5 mm depth noise, and
point spacing near the 3 mm voxel the accumulator uses. A fitter that cannot
recover these has no business on a cup.
"""
from __future__ import annotations

import numpy as np

NOISE_M = 0.0015


def box(n=6000, extents=(0.06, 0.05, 0.09), center=(0.1, -0.35, 0.06),
        seed=0, noise=NOISE_M):
    """Hollow box shell: n points spread over the six faces."""
    rng = np.random.default_rng(seed)
    h = np.asarray(extents, float) / 2
    m = n // 6
    pts = []
    for ax in range(3):
        for s in (-1, 1):
            q = rng.uniform(-1, 1, (m, 3)) * h
            q[:, ax] = s * h[ax]
            pts.append(q)
    P = np.vstack(pts)
    return P + rng.normal(0, noise, P.shape) + np.asarray(center, float)


def cylinder(n=6000, radius=0.035, length=0.09, center=(0.1, -0.35, 0.06),
             axis=(0, 0, 1), seed=0, noise=NOISE_M, arc_deg=360.0):
    """Open tube (no caps) — what a cup body actually gives the sensor."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(arc_deg)
    th = rng.uniform(-a / 2, a / 2, n)
    z = rng.uniform(-length / 2, length / 2, n)
    P = np.stack([radius * np.cos(th), radius * np.sin(th), z], 1)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    ref = np.eye(3)[int(np.argmin(np.abs(ax)))]
    x = np.cross(ref, ax)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(ax, x), ax])
    return P @ R.T + rng.normal(0, noise, P.shape) + np.asarray(center, float)


def ellipsoid(n=6000, semi=(0.045, 0.035, 0.030), center=(0.1, -0.35, 0.06),
              seed=0, noise=NOISE_M):
    """A superquadric with eps = (1, 1) — the case EMS must nail."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    P = v * np.asarray(semi, float)
    return P + rng.normal(0, noise, P.shape) + np.asarray(center, float)


def stub(n=50, seed=0):
    """A 50-point scrap: the cloud every fitter must refuse."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, (n, 3)) + np.array([0.1, -0.35, 0.06])
