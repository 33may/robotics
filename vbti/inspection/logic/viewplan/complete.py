#!/usr/bin/env python3
"""Surface-completion algorithm registry for the truthbox benchmark.

Every algorithm is `f(P: (N,3) float, cams: (C,3)|None) -> TriangleMesh`,
reconstructing the object from a partial per-step cloud, filling unseen
walls as well as it can. Scored by `truthbox.py bench` against Anton's
GT cuboids (face coverage / surface error / hallucination) — see
docs/research/view-planning/completion/GUIDE.md for the loop protocol.

HARD RULE: nothing in this file may read `truth/box.json`. Priors must
come from the CLOUD (e.g. `min_yaw_aabb`), never from the ground truth.

`boxguard` graduated to `vbti/inspection/logic/fit/box.py` (the shipped cuboid
guard); it stays registered here for the bench.

    p vbti/inspection/logic/viewplan/complete.py list
    p vbti/inspection/logic/viewplan/complete.py show --run 0809-box6 --step 26 --algo poisson20
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from inspection.fit.box import boxguard, boxguard_core, boxmesh
from inspection.viewplan import mesh as meshmod


def _poisson(P: np.ndarray, cams, trim_m: float):
    m, _info = meshmod.reconstruct(P, cams=cams, method="poisson",
                                   trim_m=trim_m)
    return m


def poisson6(P: np.ndarray, cams=None):
    """The shipped reconstruction: Poisson, 6 mm density trim (control)."""
    return _poisson(P, cams, meshmod.TRIM_M)


def poisson20(P: np.ndarray, cams=None):
    """Membrane baseline: Poisson with the trim loosened to 20 mm
    (measured sweet spot, /tmp/membrane_check sweep 2026-09-10)."""
    return _poisson(P, cams, 0.02)


def hull(P: np.ndarray, cams=None):
    """Convex hull — the no-blind-spots floor. Caps every concavity."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.asarray(P, dtype=float)))
    m, _ = pc.compute_convex_hull()
    m.compute_vertex_normals()
    return m


def star(P: np.ndarray, cams=None, seed: int = 0,
         min_members: int = 3, iters: int = 300):
    """Radial (star-shaped) completion on an icosphere — GUIDE 2.3.

    Radius-as-a-function-of-direction from the cloud center: measured
    directions carry the median point radius, unseen directions are
    Laplacian-filled from sphere neighbours. Watertight by construction:
    every centroid ray hits surface (Anton's no-blind-spots requirement,
    literal). Vertex colors: white = measured, red = interpolated —
    the planner can trust white and treat red as "go look there".
    Vertices below the table plane are clamped onto it.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree
    P = np.asarray(P, dtype=float)
    # Stray points above the object turned into radial spikes (box6 s12
    # crown) that Laplacian fill smeared into a dome (s26 render) — scrub
    # them before any radius is measured. Cloud-only, deterministic.
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    P = np.asarray(pc.points)
    # Center at the OBJECT's middle, not the cloud's mean: a top-heavy
    # partial cloud pulls P.mean into the top slab, upward rays measure
    # nothing, and the fill inflates a dome from brim radii (box6 s12/s26
    # renders, fix cycle 2, GUIDE 2.3 permitted fix). min_yaw footprint
    # center + mid-height between table and the 99.5th z percentile puts
    # the ray origin inside the box volume for any partial view.
    from inspection.cell.geometry import min_yaw_aabb
    ctr_xy, _dims, _yaw = min_yaw_aabb(P, pct=1.0)
    aa, bb, cc_, dd_ = meshmod.table_plane(P)
    z_t = -(aa * ctr_xy[0] + bb * ctr_xy[1] + dd_) / cc_
    c = np.array([ctr_xy[0], ctr_xy[1],
                  0.5 * (z_t + float(np.percentile(P[:, 2], 99.5)))])

    ico = o3d.geometry.TriangleMesh.create_icosahedron(1.0)
    ico = ico.subdivide_midpoint(number_of_iterations=4)      # 2562 verts
    V = np.asarray(ico.vertices)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    tris = np.asarray(ico.triangles)

    d = P - c
    r = np.linalg.norm(d, axis=1)
    keep = r > 1e-9
    u = d[keep] / r[keep, None]
    owner = cKDTree(V).query(u)[1]

    order = np.argsort(owner)
    owner_s, r_s = owner[order], r[keep][order]
    starts = np.searchsorted(owner_s, np.arange(len(V) + 1))
    rad = np.full(len(V), np.nan)
    counts = np.diff(starts)
    for i in np.nonzero(counts > 0)[0]:
        rad[i] = np.median(r_s[starts[i]:starts[i + 1]])
    measured = counts >= min_members
    rad[~measured & (counts > 0)] = np.nan       # too thin to trust

    # Laplacian fill: Jacobi over the sphere adjacency, measured clamped.
    nbr: Dict[int, set] = {i: set() for i in range(len(V))}
    for a, b, cc in tris:
        nbr[a].update((b, cc)); nbr[b].update((a, cc)); nbr[cc].update((a, b))
    nbr_idx = [np.fromiter(s, int) for s in nbr.values()]

    # One neighbour-median pass over measured radii: an isolated bin fed
    # only by a surviving outlier still reads huge — its ~6 neighbours
    # vote it back onto the surface. Single pass, no shrinkage loops.
    snap = rad.copy()
    for i in np.nonzero(measured)[0]:
        vals = snap[nbr_idx[i]]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 3:
            rad[i] = float(np.median(np.append(vals, snap[i])))

    fill = np.where(np.isnan(rad), np.nanmedian(rad), rad)
    free = ~measured
    for _ in range(iters):
        means = np.array([fill[n].mean() for n in nbr_idx])
        fill[free] = means[free]
    rad = fill

    pos = c + rad[:, None] * V
    a, b, cz, dd = meshmod.table_plane(P)
    z_plane = -(a * pos[:, 0] + b * pos[:, 1] + dd) / cz
    below = pos[:, 2] < z_plane
    pos[below, 2] = z_plane[below]

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(pos), o3d.utility.Vector3iVector(tris))
    col = np.where(measured[:, None], [0.9, 0.9, 0.9], [0.9, 0.25, 0.25])
    m.vertex_colors = o3d.utility.Vector3dVector(col)
    m.compute_vertex_normals()
    return m


def boxsnap(P: np.ndarray, cams=None, seed: int = 0):
    """Min-yaw cuboid prior — GUIDE 2.4. CLOUD-only, never reads GT.

    The object class is known (a box on a table), so snap the tightest
    yaw-searched cuboid to the cloud: footprint from `min_yaw_aabb`,
    bottom on the probed table plane, height from the z tail. Expected
    to under-read any wholly-unseen far wall mid-run (the box ends where
    the points end) and to flatten every non-cuboid detail.
    """
    import open3d as o3d
    from inspection.cell.geometry import min_yaw_aabb
    P = np.asarray(P, dtype=float)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    P = np.asarray(pc.points)

    center, dims, yaw = min_yaw_aabb(P, pct=1.0)
    aa, bb, cc_, dd_ = meshmod.table_plane(P)
    z_t = -(aa * center[0] + bb * center[1] + dd_) / cc_
    h = max(float(np.percentile(P[:, 2], 99.5)) - z_t, 0.01)

    m = o3d.geometry.TriangleMesh.create_box(float(dims[0]),
                                             float(dims[1]), h)
    m.translate([-float(dims[0]) / 2, -float(dims[1]) / 2, 0.0])
    R = m.get_rotation_matrix_from_axis_angle([0.0, 0.0, float(yaw)])
    m.rotate(R, center=[0.0, 0.0, 0.0])
    m.translate([float(center[0]), float(center[1]), z_t])
    m.compute_vertex_normals()
    return m


def planes(P: np.ndarray, cams=None, seed: int = 0):
    """Face-offset refinement of the cuboid prior — GUIDE 2.5, simplified.

    Deviation from the guide's RANSAC recipe, logged: instead of
    segment_plane + polygon intersection bookkeeping (the predicted
    timebox killer), each face offset is refined from its own evidence —
    the median coordinate of the points in a 15 mm slab at the min-yaw
    extent. A face with < 300 slab points keeps the raw extent (the
    boxsnap behaviour). Deterministic, no RANSAC nondeterminism.
    """
    import open3d as o3d
    from inspection.cell.geometry import min_yaw_aabb
    P = np.asarray(P, dtype=float)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    P = np.asarray(pc.points)

    center, dims, yaw = min_yaw_aabb(P, pct=1.0)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[cy, sy], [-sy, cy]])
    xy = (P[:, :2] - center[:2]) @ R2.T          # yaw frame, centred

    off = {}
    for axis, sign in ((0, +1), (0, -1), (1, +1), (1, -1)):
        ext = sign * dims[axis] / 2
        slab = np.abs(xy[:, axis] - ext) < 0.015
        off[(axis, sign)] = float(np.median(xy[slab, axis])) \
            if slab.sum() >= 300 else float(ext)
    aa, bb, cc_, dd_ = meshmod.table_plane(P)
    z_t = -(aa * center[0] + bb * center[1] + dd_) / cc_
    top = float(np.percentile(P[:, 2], 99.5))
    zs = np.abs(P[:, 2] - top) < 0.015
    z_top = float(np.median(P[zs, 2])) if zs.sum() >= 300 else top

    l = off[(0, +1)] - off[(0, -1)]
    w = off[(1, +1)] - off[(1, -1)]
    h = max(z_top - z_t, 0.01)
    m = o3d.geometry.TriangleMesh.create_box(max(l, 0.01), max(w, 0.01), h)
    m.translate([off[(0, -1)], off[(1, -1)], 0.0])
    Rz = m.get_rotation_matrix_from_axis_angle([0.0, 0.0, float(yaw)])
    m.rotate(Rz, center=[0.0, 0.0, 0.0])
    m.translate([float(center[0]), float(center[1]), z_t])
    m.compute_vertex_normals()
    return m


def boxsq(P: np.ndarray, cams=None, seed: int = 0):
    """boxguard + near-square extension of UNSEEN sides (mid-run fix).

    Mid-run, a wholly-unseen far wall makes every envelope estimator
    under-read the box ("the box ends where the points end" — boxsnap's
    documented mid failure; min-mid 0.000 across the whole board). This
    candidate guesses the unseen wall with an object prior: these boxes
    are near-square in footprint (aspect 0.87-1.10 — measured on the
    FINAL-step CLOUDS of the 9 runs, i.e. recorded data, NOT truth/;
    a dataset-level prior, disclosed as such). Per axis: if exactly one
    side lacks wall-band evidence and the OTHER axis has both walls
    confirmed, place the unseen wall at seen_wall + other_axis_dim
    (never shrinking below the envelope, extension capped at 80 mm).
    NOT REGISTERED — killed-by-design before bench (LOG iter 11): the
    prior's intrinsic error (|dim_x-dim_y| = 5-25 mm on these boxes,
    measured box9 s8: 24 mm) is at or above tau, capping unseen-face
    coverage far below the 0.60 mid bar; and the band-evidence conf
    test misclassifies in both directions (box6 final y- reads unseen,
    box9 s8 partial walls read seen). Kept as documentation.
    """
    meta = boxguard_core(P)
    off, conf, yaw, center, z_t, h = (meta["off"], meta["conf"],
                                      meta["yaw"], meta["center"],
                                      meta["z_t"], meta["h"])
    for axis in (0, 1):
        oth = 1 - axis
        if not (conf[(oth, +1)] and conf[(oth, -1)]):
            continue
        dim_oth = off[(oth, +1)] - off[(oth, -1)]
        for sign in (+1, -1):
            if conf[(axis, sign)] or not conf[(axis, -sign)]:
                continue                       # only one-unseen cases
            guess = off[(axis, -sign)] + sign * dim_oth
            lo = off[(axis, sign)]             # envelope: never shrink
            if sign > 0:
                off[(axis, sign)] = min(max(guess, lo), lo + 0.08)
            else:
                off[(axis, sign)] = max(min(guess, lo), lo - 0.08)

    return boxmesh(off, h, yaw, center, z_t)


#: Registry the bench iterates. The Ralph loop appends candidates here.
ALGOS: Dict[str, Callable] = {
    "poisson6": poisson6,
    "poisson20": poisson20,
    "hull": hull,
    "star": star,
    "boxsnap": boxsnap,
    "planes": planes,
    "boxguard": boxguard,
}


# ---------------------------------------------------------------- CLI
def list_():
    for name, fn in ALGOS.items():
        print(f"{name:12s} {(fn.__doc__ or '').splitlines()[0]}")


def show(run: str, step: int, algo: str = "poisson20",
         out: Optional[str] = None):
    """Render one algorithm's mesh from the 4 side views -> PNG path."""
    import cv2
    import open3d as o3d
    from inspection.viewplan import regions
    from inspection.viewplan import render as rendermod
    from inspection.viewplan.planner import recorded_cams, session_intrinsics
    run = str(run)
    shell = regions.shell_of(run)
    P = np.asarray(o3d.io.read_point_cloud(str(
        regions.DERIVED / run / "steps" / f"step_{int(step):03d}.ply")).points)
    m = ALGOS[algo](P, recorded_cams(run))
    scene = rendermod.scene_of(m)
    intr = session_intrinsics(run)
    row = []
    for h in (0, 3, 6, 9):
        img, _ = rendermod.render_view(
            scene, regions.orig_pose(shell, h, 0), intr, scale=0.4,
            window=(0.07, shell["r"] + 0.05))
        img = np.ascontiguousarray(img)
        cv2.putText(img, f"h{h} {algo} s{int(step)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        row.append(img)
    dst = out or f"/tmp/completion_bench/{algo}/{run}_s{int(step):03d}.png"
    import pathlib
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, np.concatenate(row, axis=1)[:, :, ::-1])
    print(dst)


def rerun(run: str, algo: str = "boxguard", out: Optional[str] = None,
          open: bool = False):                    # noqa: A002
    """Log every step's cloud + completion mesh to a scrubbable .rrd.

    One 3D view, a `step` timeline: green-tinted cloud points over the
    algo's mesh, so the inferred faces (no points on them) read at a
    glance. Same viewer conventions as `regions.run`.
    """
    import open3d as o3d
    import rerun as rr
    import rerun.blueprint as rrb
    from inspection.viewplan import regions
    run = str(run)
    fn = ALGOS[algo]
    plys = sorted((regions.DERIVED / run / "steps").glob("step_*.ply"))
    if not plys:
        raise SystemExit(f"no derived step clouds for {run}")

    app_id = f"viewplan-complete-{algo}-{run}"
    rr.init(app_id)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for p in plys:
        step = int(p.stem.split("_")[1])
        rr.set_time("step", sequence=step)
        pcd = o3d.io.read_point_cloud(str(p))
        P = np.asarray(pcd.points)
        C = ((np.asarray(pcd.colors) * 255).astype(np.uint8)
             if pcd.has_colors() else
             np.full((len(P), 3), [80, 220, 110], np.uint8))
        rr.log("world/cloud", rr.Points3D(P, colors=C, radii=[0.0012]))
        m = fn(P)
        V = np.asarray(m.vertices)
        T = np.asarray(m.triangles)
        vc = (np.asarray(m.vertex_colors) * 255).astype(np.uint8) \
            if m.has_vertex_colors() else None
        rr.log("world/box", rr.Mesh3D(
            vertex_positions=V, triangle_indices=T, vertex_colors=vc,
            albedo_factor=[170, 170, 185] if vc is None else None))
        print(f"  step {step:3d}: {len(P):6d} pts -> "
              f"{len(V)} verts / {len(T)} tris")

    dst = Path(out) if out else \
        regions.DERIVED / run / "rrd" / f"{run}-complete-{algo}.rrd"
    dst.parent.mkdir(parents=True, exist_ok=True)
    bp = rrb.Blueprint(
        rrb.Spatial3DView(origin="world", name=f"{run} {algo}"),
        collapse_panels=True)
    rr.save(str(dst), default_blueprint=bp)
    print(f"-> {dst}")
    if open:
        import subprocess
        import sys
        viewer = Path(sys.executable).parent / "rerun"
        subprocess.Popen([str(viewer if viewer.exists() else "rerun"),
                          str(dst)], start_new_session=True)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"list": list_, "show": show, "rerun": rerun})
