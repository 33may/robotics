"""e06 I2: the same physical point seen from two views of one run, lifted with each view's
own depth and recorded pose; the per-view translation that explains all pairs of the run.

Port of the 2026-10-06 registration probe (session scratchpad registration_probe/m1_*.py),
reduced to what the fit needs:
  per pair (i, j) of one run: SIFT inside the eroded object masks, ratio + mutual test,
  essential-matrix RANSAC on the recorded intrinsics, both sides lifted (median depth in a
  5x5 window, depth-edge pixels dropped), a rigid RANSAC on the lifted points to drop the
  remaining mismatches; mean offset vector (base, mm), its consistency, the epipolar misfit
  of the matches against the recorded poses (px), the median ranges.
  per run: v_i (base and camera frame) from least squares over the pairs, sum v_i = 0.
Matches are saved with their CAMERA-frame points so fit.py can re-lift them under any model.

Run:  p inspection/investigation/iwm/e06_view_offset/pairs.py [--runs ...]
Out:  evidence/pairs.json, data/iwm/e06_view_offset/pairs/<run>.npz, evidence/pairs_matches_<run>_<i>_<j>.jpg (a few)
"""
import argparse
import itertools
import json

import cv2
import numpy as np

from inspection.investigation.iwm.e06_view_offset import common as C

MAX_FEAT = 4000
RATIO = 0.75
RANSAC_PX = 2.0
MIN_INL = 15
LIFT_WIN = 2
LIFT_LO, LIFT_HI = 0.10, 0.80
EDGE_SPREAD_M = 0.004
RIGID_THR_M = 0.006
RIGID_MAX_ROT_DEG = 10.0
ERODE_PX = 9
MAX_SPILL = 0.5           # skip a view whose mask spilled (geometry.dropped / offered)


def lift(v, uv, win=LIFT_WIN):
    """(u, v) pixels (raw orientation) -> camera-frame points with the view's own aligned depth;
    NaN where no depth; edge flag where the window spans a depth edge."""
    d = v["depth"].astype(np.float64) * v["scale"]
    h, w = d.shape
    K = v["K"]
    pts = np.full((len(uv), 3), np.nan)
    edge = np.zeros(len(uv), bool)
    for k, (u, vv) in enumerate(uv):
        ui, vi = int(round(u)), int(round(vv))
        if not (win <= ui < w - win and win <= vi < h - win):
            continue
        patch = d[vi - win:vi + win + 1, ui - win:ui + win + 1].ravel()
        patch = patch[(patch > LIFT_LO) & (patch < LIFT_HI)]
        if len(patch) < 5:
            continue
        z = np.median(patch)
        edge[k] = (np.percentile(patch, 90) - np.percentile(patch, 10)) > EDGE_SPREAD_M
        pts[k] = [(u - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
    return pts, edge


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def rigid_ransac(A, B, ok, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    best = np.zeros(len(A), bool)
    if len(idx) < 3:
        return best
    for _ in range(iters):
        s = rng.choice(idx, 3, replace=False)
        R, t = kabsch(A[s], B[s])
        if np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) > RIGID_MAX_ROT_DEG:
            continue
        r = np.linalg.norm(A[idx] @ R.T + t - B[idx], axis=1)
        inl = np.zeros(len(A), bool)
        inl[idx[r < RIGID_THR_M]] = True
        if inl.sum() > best.sum():
            best = inl
    if best.sum() >= 3:
        R, t = kabsch(A[best], B[best])
        r = np.linalg.norm(A[idx] @ R.T + t - B[idx], axis=1)
        best = np.zeros(len(A), bool)
        best[idx[r < RIGID_THR_M]] = True
    return best


def pose_F(Ti, Tj, K):
    T_ji = np.linalg.inv(Tj) @ Ti
    R, t = T_ji[:3, :3], T_ji[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ki = np.linalg.inv(K)
    return Ki.T @ tx @ R @ Ki


def epi_dist(F, x1, x2):
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])
    l2 = h1 @ F.T
    l1 = h2 @ F
    e = np.abs(np.sum(h2 * l2, 1))
    return 0.5 * (e / np.hypot(l2[:, 0], l2[:, 1]) + e / np.hypot(l1[:, 0], l1[:, 1]))


def features(v, sift):
    g = v["geometry"]
    if g.get("offered") and g.get("dropped", 0) / g["offered"] > MAX_SPILL:
        return None, "mask spilled"
    if v["mask"] is None:
        return None, "no mask"
    gray = cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2GRAY)
    m = cv2.erode(v["mask"].astype(np.uint8) * 255, np.ones((ERODE_PX, ERODE_PX), np.uint8))
    kp, des = sift.detectAndCompute(gray, m)
    if des is None or len(kp) < MIN_INL:
        return None, f"{0 if des is None else len(kp)} features"
    return (np.float32([k.pt for k in kp]), des), None


def draw(vi, vj, x1, x2, path, title):
    h = max(vi["rgb"].shape[0], vj["rgb"].shape[0])
    img = np.zeros((h, vi["rgb"].shape[1] + vj["rgb"].shape[1], 3), np.uint8)
    img[:vi["rgb"].shape[0], :vi["rgb"].shape[1]] = vi["rgb"]
    img[:vj["rgb"].shape[0], vi["rgb"].shape[1]:] = vj["rgb"]
    off = vi["rgb"].shape[1]
    for (a, b) in zip(x1, x2):
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]) + off, int(b[1])), (0, 255, 0), 1)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(img, None, fx=0.6, fy=0.6), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_pairs(run, sift, out_dir, n_fig=0):
    vs = [v for v in C.views([run])]
    K = vs[0]["K"]
    feats, skipped = {}, {}
    for v in vs:
        f, why = features(v, sift)
        if f is None:
            skipped[v["step"]] = why
        else:
            feats[v["step"]] = (v, *f)
    cen = np.mean([v["cam"] for v in vs], axis=0)
    pairs, store, figs = [], {}, 0
    for a, b in itertools.combinations(sorted(feats), 2):
        vi, ki, di = feats[a]
        vj, kj, dj = feats[b]
        bf = cv2.BFMatcher(cv2.NORM_L2)
        mm = bf.knnMatch(di, dj, k=2)
        back = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(dj, di, k=1) if m}
        good = [m[0] for m in mm if len(m) == 2 and m[0].distance < RATIO * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
        ci, cj = vi["cam"] - cen, vj["cam"] - cen
        ang = float(np.degrees(np.arccos(np.clip(ci @ cj / np.linalg.norm(ci) / np.linalg.norm(cj), -1, 1))))
        rec = {"run": run, "i": a, "j": b, "id_i": vi["id"], "id_j": vj["id"], "fam_i": vi["family"], "fam_j": vj["family"],
               "angle_deg": round(ang, 1), "n_ratio": len(good)}
        if len(good) < MIN_INL:
            pairs.append(rec)
            continue
        x1 = ki[[g.queryIdx for g in good]]
        x2 = kj[[g.trainIdx for g in good]]
        E, inl = cv2.findEssentialMat(x1, x2, K, cv2.RANSAC, 0.999, RANSAC_PX)
        if E is None:
            pairs.append(rec)
            continue
        inl = inl.ravel().astype(bool)
        x1, x2 = x1[inl], x2[inl]
        Pi, ei = lift(vi, x1)
        Pj, ej = lift(vj, x2)
        ok = ~np.isnan(Pi[:, 0]) & ~np.isnan(Pj[:, 0]) & ~ei & ~ej
        Bi, Bj = C.to_base(np.nan_to_num(Pi), vi["T"]), C.to_base(np.nan_to_num(Pj), vj["T"])
        keep = rigid_ransac(Bi, Bj, ok)
        rec.update(n_essential=int(ok.sum()), n_rigid=int(keep.sum()))
        if keep.sum() < MIN_INL:
            pairs.append(rec)
            continue
        # degenerate consensus (a blob or a line of matches) says nothing about a 3D offset
        def spread(x):
            return np.linalg.svd(x - x.mean(0), compute_uv=False) / np.sqrt(len(x))
        si, sj = spread(x1[keep]), spread(x2[keep])
        if min(si[0], sj[0]) < 20 or min(si[1], sj[1]) < 5:
            rec["rejected"] = f"degenerate match set (spread {si.round(0).tolist()} / {sj.round(0).tolist()} px)"
            pairs.append(rec)
            continue
        D = (Bj[keep] - Bi[keep]) * 1000
        dist = np.linalg.norm(D, axis=1)
        mean_vec = D.mean(0)
        epx = epi_dist(pose_F(vi["T"], vj["T"], K), x1[keep], x2[keep])
        R, t = kabsch(Bi[keep], Bj[keep])
        res = np.linalg.norm(Bi[keep] @ R.T + t - Bj[keep], axis=1) * 1000
        rec.update(n=int(keep.sum()), med_mm=round(float(np.median(dist)), 2), p90_mm=round(float(np.percentile(dist, 90)), 2),
                   max_mm=round(float(dist.max()), 2), mean_vec_mm=mean_vec.round(2).tolist(),
                   mean_vec_norm_mm=round(float(np.linalg.norm(mean_vec)), 2),
                   consistency=round(float(np.linalg.norm(mean_vec) / np.mean(dist)), 3),
                   resid_after_mean_mm=round(float(np.median(np.linalg.norm(D - mean_vec, axis=1))), 2),
                   rigid_rot_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
                   rigid_resid_med_mm=round(float(np.median(res)), 2),
                   epi_pose_med_px=round(float(np.median(epx)), 2), epi_pose_p90_px=round(float(np.percentile(epx, 90)), 2),
                   range_i_m=round(float(np.median(Pi[keep][:, 2])), 3), range_j_m=round(float(np.median(Pj[keep][:, 2])), 3))
        store[f"{a}_{b}"] = {"pc_i": Pi[keep].astype(np.float32), "pc_j": Pj[keep].astype(np.float32),
                             "px_i": x1[keep], "px_j": x2[keep]}
        if figs < n_fig:
            draw(vi, vj, x1[keep], x2[keep], C.EVID / f"pairs_matches_{run}_{a}_{b}.jpg",
                 f"{run} {a}-{b}: {keep.sum()} matches, mean offset {np.linalg.norm(mean_vec):.1f} mm")
            figs += 1
        pairs.append(rec)
        print(f"  {run} {a:2d}-{b:2d} ang {ang:5.1f} n {keep.sum():4d} med {rec['med_mm']:5.1f} vec {np.round(mean_vec, 1)} "
              f"cons {rec['consistency']:.2f} epi {rec['epi_pose_med_px']:5.1f}px range {rec['range_i_m']:.2f}/{rec['range_j_m']:.2f}")
    # per-view translations from the pair vectors: mean_vec(i, j) ~ v_j - v_i, sum v = 0
    good = [p for p in pairs if "mean_vec_mm" in p]
    per_view = {}
    if good:
        steps = sorted({p["i"] for p in good} | {p["j"] for p in good})
        ix = {s: k for k, s in enumerate(steps)}
        A = np.zeros((len(good) + 1, len(steps)))
        B = np.zeros((len(good) + 1, 3))
        for r, p in enumerate(good):
            A[r, ix[p["j"]]] = 1
            A[r, ix[p["i"]]] = -1
            B[r] = p["mean_vec_mm"]
        A[-1] = 1
        t, *_ = np.linalg.lstsq(A, B, rcond=None)
        res = np.linalg.norm(A[:-1] @ t - B[:-1], axis=1)
        for s in steps:
            v = feats[s][0]
            vb = t[ix[s]]
            per_view[s] = {"id": v["id"], "family": v["family"], "address": v["address"], "elev_deg": round(C.elevation_deg(v["T"]), 1),
                           "v_base_mm": vb.round(2).tolist(), "v_cam_mm": (v["T"][:3, :3].T @ vb).round(2).tolist(),
                           "norm_mm": round(float(np.linalg.norm(vb)), 2), "n_pairs": int(sum(1 for p in good if s in (p["i"], p["j"])))}
        per_view["_model_residual_mm"] = {"median": round(float(np.median(res)), 2), "max": round(float(res.max()), 2)}
    if store:
        np.savez_compressed(out_dir / f"{run}.npz", **{f"{k}/{kk}": vv for k, d in store.items() for kk, vv in d.items()})
    return {"run": run, "skipped": skipped, "pairs": pairs, "per_view": per_view}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--figs", type=int, default=2, help="match figures per run")
    a = ap.parse_args()
    sift = cv2.SIFT_create(nfeatures=MAX_FEAT, contrastThreshold=0.02)
    out_dir = C.DATA / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run in (a.runs or C.runs()):
        print(run)
        results.append(run_pairs(run, sift, out_dir, a.figs))
    json.dump({"runs": results, "settings": {"max_feat": MAX_FEAT, "ratio": RATIO, "ransac_px": RANSAC_PX, "min_inl": MIN_INL,
                                             "lift_win": LIFT_WIN, "edge_spread_m": EDGE_SPREAD_M, "rigid_thr_m": RIGID_THR_M,
                                             "erode_px": ERODE_PX, "max_spill": MAX_SPILL}},
              open(C.EVID / "pairs.json", "w"), indent=1, cls=C.NpEncoder)
    C.run_record(out_dir, [{"run": r["run"]} for r in results], {"ratio": RATIO, "ransac_px": RANSAC_PX, "rigid_thr_m": RIGID_THR_M})
    n = sum(1 for r in results for p in r["pairs"] if "mean_vec_mm" in p)
    print(f"{n} valid pairs over {sum(1 for r in results if r['per_view'])} runs")


if __name__ == "__main__":
    main()
