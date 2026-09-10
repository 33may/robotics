"""Synthetic clouds with an analytic answer — the only ground truth we have.

Everything is posed where the real objects live (centre ~ [0.10, -0.38, 0.05]
in base frame, Z up, metres) at roughly the real point spacing and noise, so
a table-cull test means something and a standoff test is at the real scale.

A sphere is the one shape whose ideal plan is known: views on a sphere of
radius r+standoff, boresight straight at the centre. A cylinder standing on
the table has a known answer too — a ring of horizontal views around the
side plus one from above.
"""
from __future__ import annotations

import numpy as np

NOISE_M = 0.0015
CENTER = (0.10, -0.38, 0.06)


def sphere_cloud(r: float = 0.04, n: int = 4000, center=CENTER, seed: int = 0,
                 noise: float = NOISE_M):
    """Full sphere shell. Ideal plan: near-uniform views at r+standoff."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    P = v * r
    return P + rng.normal(0, noise, P.shape) + np.asarray(center, float)


def cylinder_cloud(r: float = 0.03, h: float = 0.10, n: int = 5000,
                   center=CENTER, seed: int = 0, noise: float = NOISE_M,
                   caps: bool = True):
    """Z-axis tube with caps: the side wants a ring, the top wants one view."""
    rng = np.random.default_rng(seed)
    side_area = 2 * np.pi * r * h
    cap_area = 2 * np.pi * r ** 2 if caps else 0.0
    n_side = int(round(n * side_area / max(side_area + cap_area, 1e-9)))
    th = rng.uniform(-np.pi, np.pi, n_side)
    z = rng.uniform(-h / 2, h / 2, n_side)
    P = [np.stack([r * np.cos(th), r * np.sin(th), z], 1)]
    if caps:
        m = n - n_side
        for k, s in ((m // 2, 1.0), (m - m // 2, -1.0)):
            rad = r * np.sqrt(rng.uniform(0, 1, k))
            a = rng.uniform(-np.pi, np.pi, k)
            P.append(np.stack([rad * np.cos(a), rad * np.sin(a),
                               np.full(k, s * h / 2)], 1))
    Q = np.vstack(P)
    return Q + rng.normal(0, noise, Q.shape) + np.asarray(center, float)


def half_sphere_on_table(r: float = 0.045, n: int = 4000, table_z: float = 0.0,
                         center_xy=(0.10, -0.38), seed: int = 0,
                         noise: float = NOISE_M):
    """A dome sitting ON the table: nothing below `table_z`.

    The cloud starts 8 mm above the plane, exactly like the real step clouds
    (`cell/geometry.py ABOVE_TABLE_M`), so `mesh.table_plane` has the same
    job here as on a real cup.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(int(n * 2.4), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > 0.18][:n]                       # dome, cut 8 mm up-ish
    P = v * r
    P += rng.normal(0, noise, P.shape)
    P[:, 0] += center_xy[0]
    P[:, 1] += center_xy[1]
    P[:, 2] += table_z
    return P


def cup_cloud(r: float = 0.048, h: float = 0.085, n: int = 5000,
              wall: float = 0.003, center_xy=(0.10, -0.38), table_z: float = 0.0,
              seed: int = 0, noise: float = NOISE_M):
    """An open-topped cup on the table: outer wall, inner wall, floor, rim.

    The shape the clearance guard exists for. The inner wall's normals point
    at the axis, so the ideal view of an inner patch — one standoff along the
    normal — is 100 mm inward of a 96 mm-wide cavity, i.e. *through* the far
    wall. Every guard that only looks at the table plane happily returns it.
    """
    rng = np.random.default_rng(seed)
    a_out = 2 * np.pi * r * h
    a_in = 2 * np.pi * (r - wall) * h
    a_flr = np.pi * (r - wall) ** 2
    tot = a_out + a_in + a_flr
    parts = []
    for area, rad, kind in ((a_out, r, "wall"), (a_in, r - wall, "wall"),
                            (a_flr, r - wall, "floor")):
        k = max(int(round(n * area / tot)), 8)
        th = rng.uniform(-np.pi, np.pi, k)
        if kind == "wall":
            z = rng.uniform(0.0, h, k)
            parts.append(np.stack([rad * np.cos(th), rad * np.sin(th), z], 1))
        else:
            rr = rad * np.sqrt(rng.uniform(0, 1, k))
            parts.append(np.stack([rr * np.cos(th), rr * np.sin(th),
                                   np.full(k, wall)], 1))
    th = np.linspace(-np.pi, np.pi, 200, endpoint=False)       # the rim
    rr = rng.uniform(r - wall, r, 200)
    parts.append(np.stack([rr * np.cos(th), rr * np.sin(th), np.full(200, h)], 1))
    P = np.vstack(parts) + rng.normal(0, noise, (sum(len(p) for p in parts), 3))
    P[:, 0] += center_xy[0]
    P[:, 1] += center_xy[1]
    P[:, 2] += table_z + 0.008          # the recorder's above-table margin
    return P


def inner_wall_patches(r: float = 0.05, n: int = 64, z: float = 0.06,
                       center_xy=(0.10, -0.38)):
    """(points, normals) on a cavity wall, normals pointing at the axis.

    Hand-built rather than meshed so the clearance test has no reconstruction
    in it: the geometry alone decides where the ideal cameras land.
    """
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.stack([center_xy[0] + r * np.cos(a), center_xy[1] + r * np.sin(a),
                  np.full(n, z)], 1)
    N = np.stack([-np.cos(a), -np.sin(a), np.zeros(n)], 1)
    return P, N


def two_plates(gap: float = 0.03, side: float = 0.08, n: int = 4000,
               center=CENTER, seed: int = 0, noise: float = 0.0005):
    """Two parallel squares `gap` apart, normals along +X — an occlusion rig.

    A camera on the -X side sees the near plate and nothing of the far one;
    if the raycast says otherwise the occlusion test is broken.
    """
    rng = np.random.default_rng(seed)
    m = n // 2
    out = []
    for s in (-0.5, 0.5):
        q = np.zeros((m, 3))
        q[:, 0] = s * gap
        q[:, 1] = rng.uniform(-side / 2, side / 2, m)
        q[:, 2] = rng.uniform(-side / 2, side / 2, m)
        out.append(q)
    P = np.vstack(out) + rng.normal(0, noise, (2 * m, 3))
    return P + np.asarray(center, float)


def stub(n: int = 150, seed: int = 0):
    """A scrap the planner must refuse rather than mesh."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.004, (n, 3)) + np.asarray(CENTER, float)


def line(n: int = 60, seed: int = 0):
    """A pathological 1-D cloud: no surface exists, nothing may crash."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 0.12, n)
    P = np.stack([t, np.zeros(n), np.zeros(n)], 1)
    return P + rng.normal(0, 0.0003, P.shape) + np.asarray(CENTER, float)


INTR = {"fx": 431.64349365234375, "fy": 431.2937927246094,
        "ppx": 417.2223205566406, "ppy": 237.80137634277344,
        "width": 848, "height": 480}
