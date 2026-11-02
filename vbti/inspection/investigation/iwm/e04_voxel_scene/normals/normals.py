"""e04 / normals: compare ways to get a surface normal at a point of V.

Surface points of V = vbg.extract_point_cloud (zero crossings, weight > 0.5); their
own normals are Open3D's +-1 voxel TSDF central difference (= N0b).

Test points: e03 detail nodes (ports, labels, stickers) of c5.6, c5.2 (compute demo5) and
b1.1, b1.4 (box demo1), snapped to the nearest surface point of V.
Reference per test point: RANSAC plane (2 mm) + PCA on the inliers over the V points of
the FACE the detail sits on (its e03 ancestor at FACE_DEPTH), detail included.

Methods at a point p (all flipped towards the capture camera of the source frame):
  N0a  normal of the marching-cubes triangle the camera ray hits (today's viewer)
  N0b  TSDF central difference at the nearest surface point (today's renders)
  N1   Gaussian-weighted PCA over V points within r of p          r in RADII
  N2   RANSAC plane (3 mm) on the points within r, PCA on inliers  r in RADII
  N3   TSDF smoothed with a weight-aware Gaussian (sigma voxels), central difference at p
  N4f  face plane with the detail's own points removed (needs the masks)
  N4d  PCA over the detail's own points only

Per point and method: angle to the reference; spread (mean angle of the normals at 8
nearby surface points to their mean); radius stability (angle n(r) vs n(2r));
flatness lambda0 / sum(lambda); RANSAC inlier ratio.

Outputs  evidence/normals/<frame>_<mm>mm_field.jpg   arrows on the face per method
         evidence/normals/angle_vs_radius_<mm>mm.jpg
         evidence/normals/normals.json
Run:  p vbti/inspection/investigation/iwm/e04_voxel_scene/normals/normals.py [--sizes 2 1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene.build import BLOCK_RES, OUT, device, load_steps
from inspection.investigation.iwm.e04_voxel_scene.render import K_scaled, cast, load_vbg

HERE = Path(__file__).resolve().parent
EVID = HERE.parent / "evidence" / "normals"
E03 = C.ROOT / "inspection" / "data" / "iwm" / "e03_tree_3d"
FRAMES = {"c5.6": 1, "c5.2": 1, "b1.1": 1, "b1.4": 0}      # frame -> depth of the "face" ancestor
RADII = [4, 6, 8, 12, 16, 24]                                # mm
SIGMAS = [1, 2, 3]                                           # voxels, N3
NEAR_M = 0.003                                               # V point belongs to an e03 node if this close
MIN_PTS = 300


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(unit(a) @ unit(b)), -1, 1))))  # sign-free


def pca(P, w=None):
    if len(P) < 5:
        return None, None
    w = np.ones(len(P)) if w is None else w
    mu = (P * w[:, None]).sum(0) / w.sum()
    Q = (P - mu) * np.sqrt(w)[:, None]
    lam, vec = np.linalg.eigh(Q.T @ Q / w.sum())
    return vec[:, 0], float(lam[0] / max(lam.sum(), 1e-18))


def ransac_pca(P, thr):
    if len(P) < 10:
        return None, None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, inl = pc.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
    n, _ = pca(P[inl])
    return n, len(inl) / len(P)


def orient(n, toward):
    return None if n is None else (n if n @ toward > 0 else -n)


class Frame:
    def __init__(self, fid):
        d = json.load(open(E03 / fid / "full" / "nodes.json"))
        z = np.load(E03 / fid / "full" / "nodes.npz")
        self.id, self.info = fid, d["frame"]
        self.nodes = {n["id"]: n for n in d["nodes"]}
        self.P, self.node = z["points"], z["node"]

    def subtree(self, i):
        out, todo = set(), [i]
        while todo:
            k = todo.pop()
            out.add(k)
            todo += [c for c, n in self.nodes.items() if n["parent"] == k]
        return out

    def ancestor_at(self, i, depth):
        while self.nodes[i]["depth"] > depth:
            i = self.nodes[i]["parent"]
        return i

    def points_of(self, i):
        return self.P[np.isin(self.node, list(self.subtree(i)))]


class Surface:
    """Surface points of V + a KD-tree + the dense TSDF for N3."""

    def __init__(self, vbg, size_m):
        pc = vbg.extract_point_cloud(weight_threshold=0.5)
        self.P = pc.point.positions.cpu().numpy().astype(np.float64)
        self.N0 = pc.point.normals.cpu().numpy().astype(np.float64)
        self.kd = cKDTree(self.P)
        self.size = size_m
        hm = vbg.hashmap()
        buf = hm.active_buf_indices().cpu().numpy()
        self.keys = hm.key_tensor().cpu().numpy()[buf]
        self.tsdf = vbg.attribute("tsdf").cpu().numpy()[buf][..., 0]
        self.w = vbg.attribute("weight").cpu().numpy()[buf][..., 0]
        self.index = {tuple(k): j for j, k in enumerate(self.keys)}

    def dense(self, p, half):
        """Dense (tsdf, weight) cube of 2*half+1 voxels centred on the voxel of p; [x, y, z] order."""
        c = np.floor(p / self.size).astype(int)
        n = 2 * half + 1
        T, W = np.zeros((n, n, n), np.float32), np.zeros((n, n, n), np.float32)
        lo = c - half
        for bx in range(lo[0] // BLOCK_RES, (lo[0] + n - 1) // BLOCK_RES + 1):
            for by in range(lo[1] // BLOCK_RES, (lo[1] + n - 1) // BLOCK_RES + 1):
                for bz in range(lo[2] // BLOCK_RES, (lo[2] + n - 1) // BLOCK_RES + 1):
                    j = self.index.get((bx, by, bz))
                    if j is None:
                        continue
                    blk_t = self.tsdf[j].transpose(2, 1, 0)   # stored [z][y][x] -> [x][y][z]
                    blk_w = self.w[j].transpose(2, 1, 0)
                    g0 = np.array([bx, by, bz]) * BLOCK_RES
                    a0 = np.maximum(g0, lo)
                    a1 = np.minimum(g0 + BLOCK_RES, lo + n)
                    if np.any(a1 <= a0):
                        continue
                    s = tuple(slice(a - g, b - g) for a, b, g in zip(a0, a1, g0))
                    d = tuple(slice(a - l, b - l) for a, b, l in zip(a0, a1, lo))
                    T[d], W[d] = blk_t[s], blk_w[s]
        return T, W, c - lo

    def n3(self, p, sigma):
        half = int(3 * sigma) + 3
        T, W, c = self.dense(p, half)
        if W[tuple(c)] <= 0:
            return None
        num = gaussian_filter(W * T, sigma)
        den = gaussian_filter(W, sigma)
        S = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0)
        x, y, z = c
        g = np.array([S[x + 1, y, z] - S[x - 1, y, z], S[x, y + 1, z] - S[x, y - 1, z], S[x, y, z + 1] - S[x, y, z - 1]])
        return unit(g) if np.linalg.norm(g) > 0 else None

    def near(self, p, r):
        return self.P[self.kd.query_ball_point(p, r)]

    def n1(self, p, r):
        Q = self.near(p, r)
        if len(Q) < 5:
            return None, None
        w = np.exp(-np.sum((Q - p) ** 2, 1) / (2 * (r / 2) ** 2))
        return pca(Q, w)

    def n2(self, p, r):
        return ransac_pca(self.near(p, r), 0.003)

    def belongs(self, pts):
        """V surface points within NEAR_M of a set of e03 points."""
        kd = cKDTree(pts)
        d, _ = kd.query(self.P, distance_upper_bound=NEAR_M)
        return self.P[np.isfinite(d)]


def check_voxel_order(surf):
    """The dense cube's zero crossing must sit on extract_point_cloud's surface (order/offset check)."""
    p = surf.P[len(surf.P) // 2]
    T, W, c = surf.dense(p, 6)
    idx = np.argwhere((W > 0) & (np.abs(T) < 0.15))
    if not len(idx):
        return None
    lo = np.floor(p / surf.size).astype(int) - 6
    centres = (idx + lo) * surf.size
    return float(cKDTree(surf.P).query(centres)[0].mean() * 1000)


def method_normals(surf, mesh_rs, p, cam, det_pts, face_minus):
    out = {}
    toward = cam - p
    d = unit(p - cam)
    a = mesh_rs.cast_rays(o3c.Tensor(np.concatenate([cam, d])[None].astype(np.float32)))
    out["N0a"] = (orient(a["primitive_normals"].numpy()[0].astype(float), toward), None) \
        if np.isfinite(a["t_hit"].numpy()[0]) else (None, None)
    out["N0b"] = (orient(surf.N0[surf.kd.query(p)[1]], toward), None)
    for r in RADII:
        n, fl = surf.n1(p, r / 1000)
        out[f"N1_r{r}"] = (orient(n, toward), fl)
        n, inl = surf.n2(p, r / 1000)
        out[f"N2_r{r}"] = (orient(n, toward), inl)
    for s in SIGMAS:
        out[f"N3_s{s}"] = (orient(surf.n3(p, s), toward), None)
    n, inl = ransac_pca(face_minus, 0.002) if len(face_minus) >= 30 else (None, None)
    out["N4f"] = (orient(n, toward), inl)
    n, fl = pca(det_pts) if len(det_pts) >= 10 else (None, None)
    out["N4d"] = (orient(n, toward), fl)
    return out


def spread(surf, p, fn, k=8):
    """Mean angle of the normals at k surface points within 3 mm of p to their mean (sign-aligned)."""
    idx = surf.kd.query_ball_point(p, 0.003)
    if len(idx) < 3:
        return None
    rng = np.random.default_rng(0)
    pick = surf.P[rng.choice(idx, min(k, len(idx)), replace=False)]
    ns = [fn(q) for q in pick]
    ns = [n for n in ns if n is not None]
    if len(ns) < 3:
        return None
    ref = ns[0]
    ns = np.array([n if n @ ref > 0 else -n for n in ns])
    m = unit(ns.mean(0))
    return float(np.mean([ang(n, m) for n in ns]))


def arrows_panel(img, pts, nrm, T, K, L=0.015, col=(255, 0, 255)):
    im = Image.fromarray(img.copy())
    dr = ImageDraw.Draw(im)
    Ti = np.linalg.inv(T)
    Kn = K.cpu().numpy()

    def proj(x):
        c = Ti[:3, :3] @ x + Ti[:3, 3]
        return (Kn[0, 0] * c[0] / c[2] + Kn[0, 2], Kn[1, 1] * c[1] / c[2] + Kn[1, 2])

    for p, n in zip(pts, nrm):
        if n is None:
            continue
        a, b = proj(p), proj(p + n * L)
        dr.line([a, b], fill=col, width=2)
        dr.ellipse([a[0] - 2, a[1] - 2, a[0] + 2, a[1] + 2], fill=(255, 255, 0))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=float, default=[2, 1])
    args = ap.parse_args()
    EVID.mkdir(parents=True, exist_ok=True)
    dev = device()
    results, checks = [], {}
    frames = {f: Frame(f) for f in FRAMES}
    for mm in args.sizes:
        for run in sorted({fr.info["run"] for fr in frames.values()}):
            steps, intr, scale = load_steps(run)
            vbg = load_vbg(run, "scene", mm, dev)
            surf = Surface(vbg, mm / 1000)
            checks[f"{run}_{mm:g}mm"] = check_voxel_order(surf)
            mesh = o3d.io.read_triangle_mesh(str(OUT / run / "scene" / f"{mm:g}mm" / "mesh.ply"))
            rs = o3d.t.geometry.RaycastingScene()
            rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            K = K_scaled(intr)
            for fid, fr in frames.items():
                if fr.info["run"] != run:
                    continue
                st = next(s for s in steps if s["id"] == fr.info["step"])
                cam = st["T"][:3, 3]
                face_cache = {}
                tests = [i for i, n in fr.nodes.items() if n["depth"] > FRAMES[fid] and n["depth"] >= 1
                         and n["n_points"] >= MIN_PTS and fr.ancestor_at(i, FRAMES[fid]) != i]
                for i in tests:
                    f_id = fr.ancestor_at(i, FRAMES[fid])
                    if f_id not in face_cache:
                        fpts = surf.belongs(fr.points_of(f_id))
                        n_ref, inl = ransac_pca(fpts, 0.002)
                        face_cache[f_id] = (fpts, orient(n_ref, cam - fpts.mean(0)), inl)
                    fpts, n_ref, ref_inl = face_cache[f_id]
                    det_e03 = fr.points_of(i)
                    det = surf.belongs(det_e03)
                    if n_ref is None or len(det) < 10:
                        continue
                    kd_det = cKDTree(det_e03)
                    keep = ~np.isfinite(kd_det.query(fpts, distance_upper_bound=NEAR_M)[0])
                    p = surf.P[surf.kd.query(det.mean(0))[1]]
                    nm = method_normals(surf, rs, p, cam, det, fpts[keep])
                    row = {"frame": fid, "voxel_mm": mm, "node": i, "name": fr.nodes[i]["name"],
                           "size_mm": float(max(fr.nodes[i]["extent_mm"][:2])), "face": fr.nodes[f_id]["name"],
                           "face_inlier_ratio": round(ref_inl, 3), "methods": {}}
                    for k, (n, extra) in nm.items():
                        row["methods"][k] = {"angle_to_face": None if n is None else round(ang(n, n_ref), 2),
                                             "extra": None if extra is None else round(extra, 4)}
                    for r in RADII:
                        a, b = nm[f"N1_r{r}"][0], nm.get(f"N1_r{2 * r}", (None,))[0]
                        if a is not None and b is not None:
                            row["methods"][f"N1_r{r}"]["stability_vs_2r"] = round(ang(a, b), 2)
                    for k, fn in {"N0b": lambda q: surf.N0[surf.kd.query(q)[1]],
                                  "N1_r8": lambda q: surf.n1(q, 0.008)[0], "N1_r16": lambda q: surf.n1(q, 0.016)[0],
                                  "N2_r16": lambda q: surf.n2(q, 0.016)[0], "N3_s2": lambda q: surf.n3(q, 2)}.items():
                        row["methods"][k]["spread"] = None if (s := spread(surf, p, fn)) is None else round(s, 2)
                    results.append(row)
                # arrow field on the frame's face(s): grid of V points on the faces, 12 mm apart
                if mm == min(args.sizes) or mm == 2:
                    allf = np.concatenate([v[0] for v in face_cache.values()]) if face_cache else np.empty((0, 3))
                    if len(allf):
                        g = np.unique(np.floor(allf / 0.012).astype(int), axis=0, return_index=True)[1]
                        grid = allf[g]
                        grid = grid[cKDTree(allf).query_ball_point(grid, 0.004, return_length=True) > 5]
                        _, col, _ = cast(vbg, mm / 1000, st["T"], K)
                        flipped = int(json.load(open(C.RUNS_DIR / run / "steps" / f"{st['id']:03d}" / "step.json"))
                                      .get("rgb_rotation_deg", 0) or 0) == 180
                        panels = []
                        for title, fn in [("N0b TSDF gradient (today)", lambda q: surf.N0[surf.kd.query(q)[1]]),
                                          ("N1 PCA r = 8 mm", lambda q: surf.n1(q, 0.008)[0]),
                                          ("N1 PCA r = 16 mm", lambda q: surf.n1(q, 0.016)[0]),
                                          ("N2 RANSAC r = 16 mm", lambda q: surf.n2(q, 0.016)[0]),
                                          ("N3 smoothed TSDF, sigma 2 vox", lambda q: surf.n3(q, 2))]:
                            ns = [orient(fn(q), cam - q) for q in grid]
                            pan = arrows_panel(col, grid, ns, st["T"], K)
                            if flipped:
                                pan = np.ascontiguousarray(pan[::-1, ::-1])
                            pan = C.label(pan, title, size=13)
                            panels.append(pan)
                        img = C.title_band(C.grid(panels, cols=3, pad=6, bg=(255, 255, 255)),
                                           f"{fid} ({run} step {st['id']}) · V {mm:g} mm · normals on a 12 mm grid "
                                           f"of the face, arrow 15 mm")
                        Image.fromarray(img).save(EVID / f"{fid}_{mm:g}mm_field.jpg", quality=88)
            del vbg
        plot_angle_vs_radius(results, mm)
    json.dump({"voxel_order_check_mm": checks, "results": results}, open(EVID / "normals.json", "w"), indent=1)
    summary(results)


def plot_angle_vs_radius(results, mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in results if r["voxel_mm"] == mm]
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for meth, c in [("N1", "tab:blue"), ("N2", "tab:orange")]:
        med = [np.nanmedian([r["methods"][f"{meth}_r{rr}"]["angle_to_face"] or np.nan for r in rows]) for rr in RADII]
        p90 = [np.nanpercentile([r["methods"][f"{meth}_r{rr}"]["angle_to_face"] or np.nan for r in rows], 90)
               for rr in RADII]
        ax[0].plot(RADII, med, "-o", color=c, label=f"{meth} median")
        ax[0].plot(RADII, p90, "--", color=c, label=f"{meth} p90")
    for k, c in [("N0a", "gray"), ("N0b", "black"), ("N3_s2", "tab:green"), ("N4f", "tab:red"), ("N4d", "tab:purple")]:
        v = np.nanmedian([r["methods"][k]["angle_to_face"] if r["methods"][k]["angle_to_face"] is not None else np.nan
                          for r in rows])
        ax[0].axhline(v, color=c, lw=1, ls=":", label=f"{k} median {v:.1f}°")
    ax[0].set_xlabel("radius (mm)")
    ax[0].set_ylabel("angle to the face plane normal (°)")
    ax[0].set_title(f"V {mm:g} mm · {len(rows)} detail points")
    ax[0].legend(fontsize=7)
    for r in rows:
        ax[1].plot(RADII, [r["methods"][f"N1_r{rr}"]["angle_to_face"] for rr in RADII], color="tab:blue", alpha=0.3)
    ax[1].set_xlabel("radius (mm)")
    ax[1].set_ylabel("N1 angle to face (°), one line per detail")
    ax[1].set_ylim(0, 45)
    fig.tight_layout()
    fig.savefig(EVID / f"angle_vs_radius_{mm:g}mm.jpg", dpi=110)
    plt.close(fig)


def summary(results):
    for mm in sorted({r["voxel_mm"] for r in results}):
        rows = [r for r in results if r["voxel_mm"] == mm]
        print(f"\nV {mm:g} mm, {len(rows)} detail points: median / p90 angle to the face normal")
        keys = ["N0a", "N0b"] + [f"N1_r{r}" for r in RADII] + [f"N2_r{r}" for r in RADII] + \
               [f"N3_s{s}" for s in SIGMAS] + ["N4f", "N4d"]
        for k in keys:
            v = np.array([r["methods"][k]["angle_to_face"] for r in rows if r["methods"][k]["angle_to_face"] is not None])
            sp = [r["methods"][k].get("spread") for r in rows if r["methods"][k].get("spread") is not None]
            print(f"  {k:8s} n={len(v):3d}  median {np.median(v):5.1f}°  p90 {np.percentile(v, 90):5.1f}°"
                  + (f"  spread median {np.median(sp):4.1f}°" if sp else ""))


if __name__ == "__main__":
    main()
