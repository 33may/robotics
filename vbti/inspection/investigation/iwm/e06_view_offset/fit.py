"""e06 fit: does one depth scale s (depth_true = s * depth_measured) remove the per-view offset?

FIT set: the table planes of FIT_RUNS only -> s (median over views of r_truth / r_measured
along the optical axis; views looking >= 20 deg down, axis hit < 0.7 m).
VALIDATION (never used to fit s):
  V1 the table planes of the other runs (VAL_RUNS), error along the axis before / after;
  V2 every SIFT feature pair of every run (pairs.py), re-lifted with depth * s;
  V3 the named parts: compute demo5 c5.2 / c5.6 (e05 truth pairs, e02 masks, own depth),
     plus unmasked SIFT pairs for compute demo5 steps 2/6/7 and box demo1 steps 1/4.

Run:  p vbti/inspection/investigation/iwm/e06_view_offset/fit.py
Out:  evidence/fit.json, evidence/fit.png
"""
import itertools
import json

import cv2
import numpy as np

from inspection.investigation.iwm.e06_view_offset import common as C
from inspection.investigation.iwm.e06_view_offset import pairs as PR

FIT_RUNS = ["2209-bigcam-ray-demo5", "2209-box-ray-demo2", "2209-box-ray-demo4", "2209-compute-ray-demo6",
            "2209-compute-ray-demo8", "2209-compute-ray-demo9", "2209-compute-ray-demo12", "2209-compute-ray-demo14",
            "2209-compute-ray-demo16"]
MIN_ELEV, MAX_RANGE = 20.0, 0.7
NAMED_SIFT = {"2209-compute-ray-demo5": [2, 6, 7], "2209-box-ray-demo1": [1, 4]}
TRUTH_SIZE_MIN_PX = 200


def table_split():
    d = json.load(open(C.EVID / "table.json"))
    V = [v for v in d["views"] if v["elev_deg"] >= MIN_ELEV and v["range_hit_m"] < MAX_RANGE]
    for v in V:
        r_t = v["range_hit_m"]
        r_m = r_t + v["e_axis_mm"] / 1000
        v["_rt"], v["_rm"] = r_t, r_m
    fit = [v for v in V if v["run"] in FIT_RUNS]
    val = [v for v in V if v["run"] not in FIT_RUNS]
    return fit, val


def table_err(V, s):
    return np.array([(s * v["_rm"] - v["_rt"]) * 1000 for v in V])


def pair_stats(pci, pcj, Ti, Tj, s):
    D = (C.to_base(s * pci, Tj) if False else None)
    Bi, Bj = C.to_base(s * pci, Ti), C.to_base(s * pcj, Tj)
    D = (Bj - Bi) * 1000
    return float(np.median(np.linalg.norm(D, axis=1))), D.mean(0)


def sift_pairs():
    """(run, i, j, pc_i, pc_j, T_i, T_j) from pairs.py's store."""
    out = []
    for f in sorted((C.DATA / "pairs").glob("*.npz")):
        run = f.stem
        z = np.load(f)
        keys = sorted({k.split("/")[0] for k in z.files})
        Ts = {}
        for k in keys:
            i, j = map(int, k.split("_"))
            for s in (i, j):
                if s not in Ts:
                    Ts[s] = C.load_view(run, s, arrays=False)["T"]
            out.append((run, i, j, z[f"{k}/pc_i"].astype(float), z[f"{k}/pc_j"].astype(float), Ts[i], Ts[j]))
    return out


def unmasked_pairs(run, steps, sift):
    """SIFT over the whole frame, kept only where both lifts land on the object (above the table,
    near the fused cloud); rigid RANSAC drops the rest."""
    fused = np.load(C.RUNS_DIR / run / "fused" / "cloud.npy")
    lo, hi = fused.min(0) - 0.02, fused.max(0) + 0.02
    vs = {s: C.load_view(run, s) for s in steps}
    feats = {}
    for s, v in vs.items():
        gray = cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        feats[s] = (np.float32([k.pt for k in kp]), des)
    out = []
    for a, b in itertools.combinations(steps, 2):
        (ki, di), (kj, dj) = feats[a], feats[b]
        bf = cv2.BFMatcher(cv2.NORM_L2)
        mm = bf.knnMatch(di, dj, k=2)
        back = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(dj, di, k=1) if m}
        good = [m[0] for m in mm if len(m) == 2 and m[0].distance < PR.RATIO * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
        if len(good) < PR.MIN_INL:
            out.append({"run": run, "i": a, "j": b, "n": 0, "why": f"{len(good)} matches"})
            continue
        x1 = ki[[g.queryIdx for g in good]]
        x2 = kj[[g.trainIdx for g in good]]
        Pi, ei = PR.lift(vs[a], x1)
        Pj, ej = PR.lift(vs[b], x2)
        ok = ~np.isnan(Pi[:, 0]) & ~np.isnan(Pj[:, 0]) & ~ei & ~ej
        Bi, Bj = C.to_base(np.nan_to_num(Pi), vs[a]["T"]), C.to_base(np.nan_to_num(Pj), vs[b]["T"])
        inbox = np.all((Bi > lo) & (Bi < hi), 1) & np.all((Bj > lo) & (Bj < hi), 1) & (Bi[:, 2] > 0.005) & (Bj[:, 2] > 0.005)
        keep = PR.rigid_ransac(Bi, Bj, ok & inbox)
        if keep.sum() < PR.MIN_INL:
            out.append({"run": run, "i": a, "j": b, "n": int(keep.sum()), "why": "too few on the object"})
            continue
        out.append({"run": run, "i": a, "j": b, "n": int(keep.sum()), "pc_i": Pi[keep], "pc_j": Pj[keep],
                    "T_i": vs[a]["T"], "T_j": vs[b]["T"], "id_i": vs[a]["id"], "id_j": vs[b]["id"]})
    return out


def named_parts():
    """c5.2 / c5.6 truth pairs from e05: each part = its e02 mask lifted with its view's own depth
    (camera frame), compared by median point. Returns [(name, pc_a, pc_b, T_a, T_b)]."""
    from inspection.investigation.iwm.e05_parts_in_v import common as C5
    truth = json.load(open(C5.EVID / "truth_pairs.json"))
    cache, out = {}, []
    for p in truth["pairs"]:
        if p["a_id"] == 0 or p["b_id"] == 0:
            continue
        for fid in (p["a"], p["b"]):
            if fid not in cache:
                cache[fid] = C5.load_view(fid)
        va, vb = cache[p["a"]], cache[p["b"]]
        pts = []
        for v, nid in ((va, p["a_id"]), (vb, p["b_id"])):
            m = v["masks"][nid]
            vv, uu = np.nonzero(m & (v["depth"] > 0))
            z = v["depth"][vv, uu] * v["scale"]
            ok = (z > 0.1) & (z < 0.8)
            vv, uu, z = vv[ok], uu[ok], z[ok]
            K = v["K"]
            pts.append(np.column_stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]))
        if min(len(pts[0]), len(pts[1])) < TRUTH_SIZE_MIN_PX:
            continue
        out.append({"what": p["what"][:60], "a": p["a"], "b": p["b"], "pc_a": pts[0], "pc_b": pts[1], "T_a": va["T"], "T_b": vb["T"]})
    return out


def part_offset(x, s):
    A = C.to_base(s * x["pc_a"], x["T_a"])
    B = C.to_base(s * x["pc_b"], x["T_b"])
    return (np.median(B, 0) - np.median(A, 0)) * 1000


def main():
    out = {"fit_runs": FIT_RUNS}
    # ---- FIT: s from the FIT runs' table planes
    fit, val = table_split()
    s_view = np.array([v["_rt"] / v["_rm"] for v in fit])
    s = float(np.median(s_view))
    out["fit"] = {"n_views": len(fit), "s": round(s, 4), "s_p10": round(float(np.percentile(s_view, 10)), 4),
                  "s_p90": round(float(np.percentile(s_view, 90)), 4),
                  "table_err_before_mm": {"median": round(float(np.median(np.abs(table_err(fit, 1.0)))), 2)},
                  "table_err_after_mm": {"median": round(float(np.median(np.abs(table_err(fit, s)))), 2),
                                         "p90": round(float(np.percentile(np.abs(table_err(fit, s)), 90)), 2)}}
    print(f"FIT: s = {s:.4f} ({len(fit)} views; p10 {out['fit']['s_p10']}, p90 {out['fit']['s_p90']})")
    # ---- V1: the other runs' table planes
    eb, ea = table_err(val, 1.0), table_err(val, s)
    out["v1_table"] = {"n_views": len(val), "before_mm": {"median": round(float(np.median(np.abs(eb))), 2), "p90": round(float(np.percentile(np.abs(eb), 90)), 2), "max": round(float(np.abs(eb).max()), 2)},
                       "after_mm": {"median": round(float(np.median(np.abs(ea))), 2), "p90": round(float(np.percentile(np.abs(ea), 90)), 2), "max": round(float(np.abs(ea).max()), 2)},
                       "views": [{"run": v["run"], "step": v["step"], "range_m": v["_rt"], "before_mm": round(float(b), 2), "after_mm": round(float(a), 2)} for v, b, a in zip(val, eb, ea)]}
    print(f"V1 table ({len(val)} held-out views): |err| median {np.median(np.abs(eb)):.1f} -> {np.median(np.abs(ea)):.1f} mm, p90 {np.percentile(np.abs(eb), 90):.1f} -> {np.percentile(np.abs(ea), 90):.1f}")
    # ---- V2: SIFT pairs
    S = np.linspace(0.98, 1.10, 61)
    sp = sift_pairs()
    rows = []
    for run, i, j, pci, pcj, Ti, Tj in sp:
        b, vb = pair_stats(pci, pcj, Ti, Tj, 1.0)
        a, va = pair_stats(pci, pcj, Ti, Tj, s)
        rows.append({"run": run, "i": i, "j": j, "split": "fit" if run in FIT_RUNS else "val", "n": len(pci),
                     "before_mm": round(b, 2), "after_mm": round(a, 2), "vec_before_mm": vb.round(2).tolist(), "vec_after_mm": va.round(2).tolist()})
    curve = [float(np.median([pair_stats(pci, pcj, Ti, Tj, x)[0] for _, _, _, pci, pcj, Ti, Tj in sp])) for x in S]
    out["v2_pairs"] = {"n_pairs": len(rows), "pairs": rows,
                       "median_before_mm": round(float(np.median([r["before_mm"] for r in rows])), 2),
                       "median_after_mm": round(float(np.median([r["after_mm"] for r in rows])), 2),
                       "p90_before_mm": round(float(np.percentile([r["before_mm"] for r in rows], 90)), 2),
                       "p90_after_mm": round(float(np.percentile([r["after_mm"] for r in rows], 90)), 2),
                       "max_before_mm": round(float(max(r["before_mm"] for r in rows)), 2),
                       "max_after_mm": round(float(max(r["after_mm"] for r in rows)), 2),
                       "best_s_on_pairs": round(float(S[int(np.argmin(curve))]), 4),
                       "curve": {"s": S.round(4).tolist(), "median_mm": np.round(curve, 2).tolist()}}
    for r in rows:
        print(f"  V2 {r['split']:3s} {r['run']:26s} {r['i']:2d}-{r['j']:2d} n {r['n']:4d} {r['before_mm']:5.1f} -> {r['after_mm']:5.1f} mm")
    print(f"V2 pairs ({len(rows)}): median {out['v2_pairs']['median_before_mm']} -> {out['v2_pairs']['median_after_mm']} mm, "
          f"p90 {out['v2_pairs']['p90_before_mm']} -> {out['v2_pairs']['p90_after_mm']}, max {out['v2_pairs']['max_before_mm']} -> {out['v2_pairs']['max_after_mm']}; "
          f"s that the pairs alone would pick: {out['v2_pairs']['best_s_on_pairs']}")
    # ---- V3: the named parts and the named steps
    sift = cv2.SIFT_create(nfeatures=PR.MAX_FEAT, contrastThreshold=0.02)
    named = []
    for x in named_parts():
        b, a = part_offset(x, 1.0), part_offset(x, s)
        named.append({"kind": "part", "pair": f"{x['a']}-{x['b']}", "what": x["what"], "before_mm": round(float(np.linalg.norm(b)), 2),
                      "after_mm": round(float(np.linalg.norm(a)), 2), "vec_before_mm": b.round(2).tolist(), "vec_after_mm": a.round(2).tolist()})
    for run, steps in NAMED_SIFT.items():
        for p in unmasked_pairs(run, steps, sift):
            if "pc_i" not in p:
                named.append({"kind": "sift", "pair": f"{run} {p['i']}-{p['j']}", "n": p["n"], "skipped": p["why"]})
                continue
            b, vb = pair_stats(p["pc_i"], p["pc_j"], p["T_i"], p["T_j"], 1.0)
            a, va = pair_stats(p["pc_i"], p["pc_j"], p["T_i"], p["T_j"], s)
            named.append({"kind": "sift", "pair": f"{run} {p['i']}-{p['j']} ({p['id_i']}-{p['id_j']})", "n": p["n"],
                          "before_mm": round(b, 2), "after_mm": round(a, 2), "vec_before_mm": vb.round(2).tolist(), "vec_after_mm": va.round(2).tolist()})
    out["v3_named"] = named
    for r in named:
        print(f"  V3 {r['kind']:4s} {r['pair']:42s} {r.get('what', '')[:34]:34s} " +
              (f"{r['before_mm']:5.1f} -> {r['after_mm']:5.1f} mm  vec {r['vec_before_mm']} -> {r['vec_after_mm']}" if "before_mm" in r else f"skipped: {r['skipped']}"))
    json.dump(out, open(C.EVID / "fit.json", "w"), indent=1, cls=C.NpEncoder)
    C.run_record(C.DATA / "fit", [{"fit_runs": FIT_RUNS}], {"min_elev": MIN_ELEV, "max_range": MAX_RANGE, "s": s})
    figure(out)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    v1 = out["v1_table"]["views"]
    ax[0].scatter([v["range_m"] for v in v1], [v["before_mm"] for v in v1], c="#d62728", s=18, label="before")
    ax[0].scatter([v["range_m"] for v in v1], [v["after_mm"] for v in v1], c="#2ca02c", s=18, label=f"after (s = {out['fit']['s']})")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("camera range to the table (m)")
    ax[0].set_ylabel("table range error along the axis (mm)")
    ax[0].set_title("V1: held-out runs' table planes")
    ax[0].legend()
    p = out["v2_pairs"]["pairs"]
    for r in p:
        ax[1].plot([0, 1], [r["before_mm"], r["after_mm"]], "-o", color="#1f77b4" if r["split"] == "val" else "#999999", ms=3, lw=1, alpha=0.8)
    ax[1].set_xticks([0, 1], ["before", "after"])
    ax[1].set_ylabel("same point from two views: median distance (mm)")
    ax[1].set_title("V2: feature pairs (blue = held-out runs)")
    c = out["v2_pairs"]["curve"]
    ax[2].plot(c["s"], c["median_mm"], "-")
    ax[2].axvline(out["fit"]["s"], color="#2ca02c", ls="--", label="s fitted on the table")
    ax[2].set_xlabel("depth scale s")
    ax[2].set_ylabel("median pair distance (mm)")
    ax[2].set_title("V2: pair distance vs the scale")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(C.EVID / "fit.png", dpi=130)


if __name__ == "__main__":
    main()
