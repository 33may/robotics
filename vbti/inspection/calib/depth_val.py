#!/usr/bin/env python3
"""Depth-calibration validation on a recorded run (read-only on the run).

Target: one yellow dot on a red object, seen from every captured view.

  1. detect the dot in each view's colour image (saturated yellow, ringed by red —
     rejects the yellow/black pole and the plywood);
  2. TRUTH WITHOUT DEPTH: least-squares intersection of the camera rays through the dot
     (poses + colour intrinsics only — the depth calibration cannot touch it);
  3. per view: the dot lifted with that view's own depth -> base; its error along the ray
     vs the truth = the depth error at that range;
  4. per view: the table plane from that view's depth (object masked out) vs the
     probed slab (calib/table_probe_2026-08-18.yaml).

Same run before / after a calibration = same poses, so the numbers compare directly.

    p vbti/inspection/calib/depth_val.py --run=1310-calib-red
Out: vbti/inspection/data/depth_val/<run>/{depth_val.json, dots.jpg}
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "inspection" / "data" / "runs"
OUT = ROOT / "inspection" / "data" / "depth_val"
PROBE = Path(__file__).resolve().parent / "table_probe_2026-08-18.yaml"

YELLOW = ((12, 100, 120), (40, 255, 255))      # HSV; the dot drifts to H 13-18 in warm light (1310 step 4)
RED = [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (180, 255, 255))]
RING_PX, RING_RED_MIN = 6, 0.5                  # ring around the blob must be >= 50 % red
TABLE_RANGE = (0.08, 0.9)


def K_of(i):
    return np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1.0]])


def run_dir(run):
    """A run id under data/runs, or a path to any run directory."""
    return Path(run) if Path(run).is_dir() else RUNS / run


def load(run, step):
    sd = run_dir(run) / "steps" / f"{step:03d}"
    st = json.load(open(sd / "step.json"))
    if st.get("outcome") != "captured":
        return None
    rgb = cv2.cvtColor(cv2.imread(str(sd / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(sd / "depth_aligned.npy")
    if int(st.get("rgb_rotation_deg") or 0) == 180:         # disk images are upright; T is raw
        rgb, depth = rgb[::-1, ::-1], depth[::-1, ::-1]
    return {"step": step, "rgb": np.ascontiguousarray(rgb), "depth": np.ascontiguousarray(depth),
            "T": np.array(st["T_base_cam"], float), "address": st["view"].get("address")}


def red_mask(hsv):
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in RED:
        m |= cv2.inRange(hsv, lo, hi) > 0
    return m


def find_dot(rgb):
    """Pixel centroid (u, v), blob mask, or (None, why)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yel = (cv2.inRange(hsv, *YELLOW) > 0).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = red_mask(hsv)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(yel)
    best = None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 15:
            continue
        blob = lab == k
        ring = cv2.dilate(blob.astype(np.uint8), np.ones((2 * RING_PX + 1,) * 2, np.uint8)).astype(bool) & ~blob
        frac = red[ring].mean()
        if frac >= RING_RED_MIN and (best is None or frac > best[0]):
            best = (frac, blob, cent[k])
    if best is None:
        return None, None, "no yellow blob ringed by red"
    return best[2], best[1], f"ring red {best[0]:.2f}, {int(best[1].sum())} px"


def triangulate(rays):
    """Least-squares point closest to all rays (origin o, unit direction d)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    x = np.linalg.solve(A, b)
    return x, [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o))) for o, d in rays]


def truth_plane():
    P = np.array([p["base"] for p in yaml.safe_load(open(PROBE))["points"]], float)
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    n = n if n[2] > 0 else -n
    return n, c


def table_height(v, K, scale, obj, n_true, c_true, rng):
    z = v["depth"].astype(float) * scale
    ok = (z > TABLE_RANGE[0]) & (z < TABLE_RANGE[1]) & ~obj
    vv, uu = np.nonzero(ok)
    if len(vv) > 60000:
        k = rng.choice(len(vv), 60000, replace=False)
        vv, uu = vv[k], uu[k]
    zz = z[vv, uu]
    P = np.column_stack([(uu - K[0, 2]) / K[0, 0] * zz, (vv - K[1, 2]) / K[1, 1] * zz, zz])
    B = P @ v["T"][:3, :3].T + v["T"][:3, 3]
    d = (B - c_true) @ n_true                              # signed height above the probed slab
    near = np.abs(d) < 0.04                                # the table, not walls/floor
    if near.sum() < 2000:
        return None
    best = None
    for _ in range(200):                                   # RANSAC plane in base
        s = B[near][rng.choice(near.sum(), 3, replace=False)]
        nn = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nn) < 1e-9:
            continue
        nn /= np.linalg.norm(nn)
        inl = np.abs((B[near] - s[0]) @ nn) < 0.003
        if best is None or inl.sum() > best.sum():
            best = inl
    h = d[near][best]
    return {"h_mm": round(float(np.median(h)) * 1000, 2), "n": int(best.sum()),
            "range_m": round(float(np.median(zz[near][best])), 3)}


def main(run: str = "1310-calib-red"):
    rd = run_dir(run)
    run = rd.name
    sess = json.load(open(rd / "session.json"))
    K, scale = K_of(sess["intrinsics"]["color"]), sess["depth_scale_m_per_unit"]
    n_true, c_true = truth_plane()
    rng = np.random.default_rng(0)
    steps = sorted(int(p.name) for p in (rd / "steps").iterdir() if p.name.isdigit())
    views, rays, tiles = [], [], []
    for s in steps:
        v = load(rd, s)
        if v is None:
            continue
        uv, blob, why = find_dot(v["rgb"])
        rec = {"step": s, "address": v["address"], "detect": why}
        tile = cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2BGR).copy()
        if uv is not None:
            u, w = uv
            d_cam = np.linalg.inv(K) @ np.array([u, w, 1.0])
            d_cam /= np.linalg.norm(d_cam)
            o, d = v["T"][:3, 3], v["T"][:3, :3] @ d_cam
            rays.append((o, d))
            core = cv2.erode(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            core = core if core.sum() >= 5 else blob
            z = v["depth"][core].astype(float) * scale
            z = z[(z > 0.05) & (z < 1.5)]
            rec.update(uv=[round(float(u), 1), round(float(w), 1)], blob_px=int(blob.sum()),
                       depth_valid=f"{len(z)}/{int(core.sum())}")
            if len(z) >= 3:
                zc = float(np.median(z))
                p_cam = np.array([(u - K[0, 2]) / K[0, 0] * zc, (w - K[1, 2]) / K[1, 1] * zc, zc])
                rec.update(z_m=round(zc, 4), p_base=(v["T"][:3, :3] @ p_cam + v["T"][:3, 3]).tolist(),
                           _o=o, _d=d)
            cv2.circle(tile, (int(u), int(w)), 22, (0, 255, 0), 3)
        else:
            cv2.putText(tile, "NO DOT", (40, 120), 0, 3, (0, 0, 255), 6)
        hsv = cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2HSV)
        obj = cv2.dilate(red_mask(hsv).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        rec["table"] = table_height(v, K, scale, obj, n_true, c_true, rng)
        cv2.putText(tile, f"step {s} {v['address']}", (20, 60), 0, 1.8, (0, 255, 0), 4)
        tiles.append(cv2.resize(tile, (640, 360)))
        views.append(rec)

    out = {"run": run, "n_views": len(views), "n_dots": len(rays)}
    if len(rays) >= 2:
        X, miss = triangulate(rays)
        out["dot_truth_base_m"] = X.round(5).tolist()
        out["ray_miss_mm"] = {"median": round(float(np.median(miss)) * 1000, 2), "max": round(float(max(miss)) * 1000, 2)}
        errs = []
        for r in views:
            if "_o" not in r:
                continue
            p = np.array(r["p_base"])
            along = float((p - r["_o"]) @ r["_d"] - (X - r["_o"]) @ r["_d"])   # + = depth reads long
            r.update(range_true_m=round(float((X - r["_o"]) @ r["_d"]), 4), along_mm=round(along * 1000, 2),
                     off_mm=round(float(np.linalg.norm(p - X)) * 1000, 2),
                     along_pct=round(along / float((X - r["_o"]) @ r["_d"]) * 100, 2))
            errs.append(r["along_mm"])
        out["dot_along_mm"] = {"median_abs": round(float(np.median(np.abs(errs))), 2), "max_abs": round(float(np.max(np.abs(errs))), 2)}
    th = [r["table"]["h_mm"] for r in views if r.get("table")]
    out["table_h_mm"] = {"median_abs": round(float(np.median(np.abs(th))), 2), "max_abs": round(float(np.max(np.abs(th))), 2)} if th else None
    for r in views:
        r.pop("_o", None), r.pop("_d", None)
    out["views"] = views

    od = OUT / run
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "depth_val.json", "w"), indent=1)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(od / "dots.jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))

    print(f"{run}: {len(rays)}/{len(views)} views with the dot; truth {out.get('dot_truth_base_m')}, ray miss {out.get('ray_miss_mm')}")
    print(f"{'step':>4} {'cell':>8} {'range m':>8} {'dot along mm':>13} {'along %':>8} {'dot off mm':>10} {'table h mm':>11}  detect")
    for r in views:
        print(f"{r['step']:>4} {str(r['address']):>8} {str(r.get('range_true_m', '-')):>8} {str(r.get('along_mm', '-')):>13} "
              f"{str(r.get('along_pct', '-')):>8} {str(r.get('off_mm', '-')):>10} {str((r['table'] or {}).get('h_mm', '-')):>11}  {r['detect']}")
    print(f"SUMMARY dot along |mm| {out.get('dot_along_mm')}, table |h| mm {out['table_h_mm']}")
    print(f"-> {od}")
    return out


if __name__ == "__main__":
    import fire
    fire.Fire(lambda run="1310-calib-red": main(run) and None)
