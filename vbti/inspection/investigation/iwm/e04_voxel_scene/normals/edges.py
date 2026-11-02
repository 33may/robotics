"""e04 / normals, turn 2: normals near a face edge, without breaking the interior.

Test points: V surface points of each e03 face (c5.2 top / front / bevel, c5.6 panels,
b1.1 top / front, b1.4 front), binned by EDGE DISTANCE = distance to the nearest V point of
the object that is off the face plane (> OFF_M), i.e. where the surface leaves the face.
Reference: the face plane (RANSAC 2 mm + PCA over the whole face). Plus the 34 detail
points of turn 1 (non-regression).

Methods, all at radius R (16 mm), flipped to the capture camera:
  N2    RANSAC (3 mm) largest plane + PCA on inliers        (turn-1 default)
  M3    multi-plane RANSAC: up to 3 planes, keep the plane the point lies on
  M1    shifted neighbourhoods: 13 regions of radius R containing the point (centre + 12
        offsets of R/2), PCA each, keep the flattest region whose plane passes within 3 mm
  M2    Hough-style voting: 400 random triplets, votes only from planes within 2 mm of the
        point, densest normal cluster wins, refined by PCA on its supporting points
  M5    semantic: RANSAC + PCA on the points of the point's OWN FACE within R (needs T)

Outputs evidence/normals/edges_<mm>mm.jpg, edges_field_<frame>.jpg, edges.json
Run:  p vbti/inspection/investigation/iwm/e04_voxel_scene/normals/edges.py [--sizes 1 2]
"""
import argparse
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene.build import device, load_steps
from inspection.investigation.iwm.e04_voxel_scene.normals.normals import (EVID, FRAMES, MIN_PTS, NEAR_M, Frame, Surface,
                                                                          ang, arrows_panel, orient, pca, ransac_pca,
                                                                          unit)
from inspection.investigation.iwm.e04_voxel_scene.render import K_scaled, cast, load_vbg

R = 0.016
OFF_M = 0.006                     # a point is "off the face" beyond this distance from its plane
BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (20, 60)]
PER_BIN = 25
FACES = {"c5.2": [2, 3, 5], "c5.6": [2, 3], "b1.1": [51, 39], "b1.4": [0]}


def m3(Q, p, thr=0.003, max_planes=3, min_frac=0.12):
    rest, best = Q, None
    for _ in range(max_planes):
        if len(rest) < max(10, min_frac * len(Q)):
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rest))
        (a, b, c, d), inl = pc.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=600)
        if len(inl) < min_frac * len(Q):
            break
        dist = abs(np.array([a, b, c]) @ p + d)
        if best is None or dist < best[0] - 0.001:
            best = (dist, rest[inl])
        rest = np.delete(rest, inl, axis=0)
    return None if best is None else pca(best[1])[0]


def m1(surf, p, r=R):
    best = None
    dirs = [np.zeros(3)]
    rng = np.random.default_rng(0)
    v = rng.normal(size=(12, 3))
    dirs += list(v / np.linalg.norm(v, axis=1)[:, None])
    for dv in dirs:
        c = p + dv * r / 2
        Q = surf.near(c, r)
        if len(Q) < 30:
            continue
        n, fl = pca(Q)
        if n is None or abs(n @ (p - Q.mean(0))) > 0.003:
            continue
        if best is None or fl < best[0]:
            best = (fl, n)
    return None if best is None else best[1]


def m2(Q, p, n_tri=400, thr=0.002, cos_tol=np.cos(np.radians(8))):
    if len(Q) < 30:
        return None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(Q), size=(n_tri, 3))
    A, B, Cc = Q[idx[:, 0]], Q[idx[:, 1]], Q[idx[:, 2]]
    N = np.cross(B - A, Cc - A)
    L = np.linalg.norm(N, axis=1)
    ok = L > 1e-9
    N, A = N[ok] / L[ok][:, None], A[ok]
    ok = np.abs(np.sum(N * (p - A), 1)) < thr               # plane passes through the point
    N = N[ok]
    if len(N) < 10:
        return None
    N = N * np.sign(N[:, 2:3] + 1e-9)                      # fold the sign
    S = np.abs(N @ N.T) > cos_tol
    k = int(np.argmax(S.sum(1)))
    cand = unit(np.mean(N[S[k]] * np.sign(N[S[k]] @ N[k])[:, None], 0))
    sup = Q[np.abs((Q - p) @ cand) < 0.003]
    n, _ = pca(sup) if len(sup) >= 10 else (cand, None)
    return n


def methods(surf, p, face_pts_kd, face_pts):
    Q = surf.near(p, R)
    out = {"N2": surf.n2(p, R)[0], "M3": m3(Q, p), "M1": m1(surf, p), "M2": m2(Q, p)}
    own = face_pts[face_pts_kd.query_ball_point(p, R)]
    out["M5"] = ransac_pca(own, 0.003)[0] if len(own) >= 20 else None
    return out


def run(mm, frames, dev):
    rows, fields = [], {}
    for run_id in sorted({fr.info["run"] for fr in frames.values()}):
        steps, intr, scale = load_steps(run_id)
        vbg = load_vbg(run_id, "scene", mm, dev)
        surf = Surface(vbg, mm / 1000)
        obj = np.load(C.RUNS_DIR / run_id / "fused" / "cloud.npy")
        lo, hi = obj.min(0) - 0.01, obj.max(0) + 0.01
        in_obj = np.all((surf.P > lo) & (surf.P < hi), 1) & (surf.P[:, 2] > 0.004)
        for fid, fr in frames.items():
            if fr.info["run"] != run_id:
                continue
            st = next(s for s in steps if s["id"] == fr.info["step"])
            cam = st["T"][:3, 3]
            for face in FACES[fid]:
                fpts = surf.belongs(fr.points_of(face))
                if len(fpts) < 200:
                    continue
                n_ref, _ = ransac_pca(fpts, 0.002)
                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fpts))
                (a, b, c, d), inl = pc.segment_plane(0.002, 3, 2000)
                fpts = fpts[inl]                                  # the face proper
                n_ref = orient(pca(fpts)[0], cam - fpts.mean(0))
                plane_d = (surf.P[in_obj] - fpts.mean(0)) @ n_ref
                off = surf.P[in_obj][np.abs(plane_d) > OFF_M]
                if len(off) < 10:
                    continue
                edge_d = cKDTree(off).query(fpts)[0] * 1000
                fkd = cKDTree(fpts)
                rng = np.random.default_rng(face)
                for b0, b1 in BINS:
                    cand = np.where((edge_d >= b0) & (edge_d < b1))[0]
                    for j in rng.choice(cand, min(PER_BIN, len(cand)), replace=False):
                        p = fpts[j]
                        res = methods(surf, p, fkd, fpts)
                        rows.append({"set": "edge", "voxel_mm": mm, "frame": fid, "face": fr.nodes[face]["name"],
                                     "edge_mm": round(float(edge_d[j]), 1), "bin": f"{b0}-{b1}",
                                     "angle": {k: (None if n is None else round(ang(n, n_ref), 2))
                                               for k, n in res.items()}})
                fields.setdefault(fid, []).append((fpts, fkd, n_ref))
            # non-regression: turn-1 detail points
            for i, nd in fr.nodes.items():
                if not (nd["depth"] > FRAMES[fid] and nd["depth"] >= 1 and nd["n_points"] >= MIN_PTS
                        and fr.ancestor_at(i, FRAMES[fid]) != i):
                    continue
                f_id = fr.ancestor_at(i, FRAMES[fid])
                fpts = surf.belongs(fr.points_of(f_id))
                det = surf.belongs(fr.points_of(i))
                if len(det) < 10 or len(fpts) < 50:
                    continue
                n_ref = orient(ransac_pca(fpts, 0.002)[0], cam - fpts.mean(0))
                p = surf.P[surf.kd.query(det.mean(0))[1]]
                res = methods(surf, p, cKDTree(fpts), fpts)
                rows.append({"set": "detail", "voxel_mm": mm, "frame": fid, "name": nd["name"],
                             "angle": {k: (None if n is None else round(ang(n, n_ref), 2)) for k, n in res.items()}})
            if mm == 1 or mm == 2:
                field_figure(fid, fields.get(fid, []), surf, vbg, st, intr, mm, run_id)
        del vbg
    return rows


def field_figure(fid, faces, surf, vbg, st, intr, mm, run_id):
    from PIL import Image
    if not faces:
        return
    K = K_scaled(intr)
    _, col, _ = cast(vbg, mm / 1000, st["T"], K)
    cam = st["T"][:3, 3]
    flipped = int(json.load(open(C.RUNS_DIR / run_id / "steps" / f"{st['id']:03d}" / "step.json"))
                  .get("rgb_rotation_deg", 0) or 0) == 180
    grid = []
    for fpts, fkd, _ in faces:
        g = np.unique(np.floor(fpts / 0.008).astype(int), axis=0, return_index=True)[1]
        grid += [(q, fpts, fkd) for q in fpts[g]]
    panels = []
    for name in ["N2", "M3", "M1", "M2", "M5"]:
        ns = []
        for q, fpts, fkd in grid:
            n = methods_one(name, surf, q, fkd, fpts)
            ns.append(orient(n, cam - q) if n is not None else None)
        pan = arrows_panel(col, [g[0] for g in grid], ns, st["T"], K, L=0.012)
        pan = np.ascontiguousarray(pan[::-1, ::-1]) if flipped else pan
        panels.append(C.label(pan, {"N2": "N2 RANSAC largest plane (turn 1)", "M3": "M3 multi-plane, plane through p",
                                    "M1": "M1 shifted neighbourhoods", "M2": "M2 voting, planes through p",
                                    "M5": "M5 own face only (needs T)"}[name], size=13))
    img = C.title_band(C.grid(panels, cols=3, pad=6, bg=(255, 255, 255)),
                       f"{fid} · V {mm:g} mm · normals on an 8 mm grid of the faces, r = 16 mm "
                       f"(face-on: a dot = correct, a long arrow = tilted)")
    Image.fromarray(img).save(EVID / f"edges_field_{fid}_{mm:g}mm.jpg", quality=88)


def methods_one(name, surf, p, fkd, fpts):
    if name == "N2":
        return surf.n2(p, R)[0]
    Q = surf.near(p, R)
    if name == "M3":
        return m3(Q, p)
    if name == "M1":
        return m1(surf, p)
    if name == "M2":
        return m2(Q, p)
    own = fpts[fkd.query_ball_point(p, R)]
    return ransac_pca(own, 0.003)[0] if len(own) >= 20 else None


def plot(rows, mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = [r for r in rows if r["set"] == "edge" and r["voxel_mm"] == mm]
    D = [r for r in rows if r["set"] == "detail" and r["voxel_mm"] == mm]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = [f"{a}-{b}" for a, b in BINS]
    cols = {"N2": "tab:orange", "M3": "tab:blue", "M1": "tab:green", "M2": "tab:purple", "M5": "tab:red"}
    for k, c in cols.items():
        med, p90, bad = [], [], []
        for lb in labels:
            v = np.array([r["angle"][k] for r in E if r["bin"] == lb and r["angle"][k] is not None])
            med.append(np.median(v) if len(v) else np.nan)
            p90.append(np.percentile(v, 90) if len(v) else np.nan)
            bad.append(100 * np.mean(v > 10) if len(v) else np.nan)
        ax[0].plot(labels, med, "-o", color=c, label=k)
        ax[1].plot(labels, p90, "-o", color=c, label=k)
        ax[2].plot(labels, bad, "-o", color=c, label=k)
    ax[0].set_title(f"median angle to face normal · V {mm:g} mm · {len(E)} face points")
    ax[1].set_title("p90 angle (°)")
    ax[2].set_title("share of points > 10° off (%)")
    for a in ax:
        a.set_xlabel("distance to the face edge (mm)")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVID / f"edges_{mm:g}mm.jpg", dpi=110)
    plt.close(fig)


def summary(rows):
    for mm in sorted({r["voxel_mm"] for r in rows}):
        print(f"\nV {mm:g} mm")
        for s, bins in [("edge", [f"{a}-{b}" for a, b in BINS]), ("detail", [None])]:
            for lb in bins:
                R_ = [r for r in rows if r["set"] == s and r["voxel_mm"] == mm and (lb is None or r["bin"] == lb)]
                line = f"  {s:6s} {lb or 'all':6s} n={len(R_):3d} "
                for k in ["N2", "M3", "M1", "M2", "M5"]:
                    v = np.array([r["angle"][k] for r in R_ if r["angle"][k] is not None])
                    line += f" {k} {np.median(v):4.1f}/{np.percentile(v, 90):4.1f}/{100 * np.mean(v > 10):3.0f}%" \
                        if len(v) else f" {k} -"
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=float, default=[1, 2])
    args = ap.parse_args()
    dev = device()
    frames = {f: Frame(f) for f in FRAMES}
    rows = []
    for mm in args.sizes:
        rows += run(mm, frames, dev)
        plot(rows, mm)
    json.dump(rows, open(EVID / "edges.json", "w"), indent=1)
    summary(rows)


if __name__ == "__main__":
    main()
