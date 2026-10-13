"""e06 I1 + I4: the table plane seen by every view, against the probed slab.

Per captured step of every 2209 run, from the view's OWN depth and the RECORDED pose:
  - table pixels = outside the dilated object mask, 0.13-0.90 m, and (prior gate) within
    40 mm of the probed slab and inside the slab footprint once lifted with the recorded pose;
  - RANSAC + least-squares plane in the CAMERA frame; then in base through T_base_cam;
  - h       height of the fitted plane above the probed slab at the optical-axis hit (mm);
  - e_axis  range error along the optical axis at that hit (fitted - truth, mm; < 0 = short);
  - tilt    the fitted normal minus the slab normal, in base and in the camera frame (mrad);
  - profile per-pixel range residual dr = r_measured - r_truth binned by r_truth (mm);
  - I4      the same from depth_raw + depth intrinsics; the difference in h and tilt.

Run:  p inspection/investigation/iwm/e06_view_offset/table.py [--runs ...] [--no-fig]
Out:  evidence/table.json, evidence/table_profile.png, evidence/table_height.png,
      data/iwm/e06_view_offset/table/<run>_<step>_<src>.npz (inlier points, camera frame)
"""
import argparse
import json

import cv2
import numpy as np
import open3d as o3d

from inspection.investigation.iwm.e06_view_offset import common as C

MIN_R, MAX_R = 0.13, 0.90      # m; fingers live at 0.07-0.12 m
GATE_M = 0.040                 # prior gate around the probed slab (m) - wide vs the 10-25 mm errors
INSET_M = 0.02                 # stay this far inside the slab's footprint
MASK_DILATE_PX = 25
RANSAC_M = 0.004
REFINE_M = 0.005
MIN_INLIERS = 1500
BINS = np.arange(0.10, 0.95, 0.05)
KEEP = 4000                    # inlier points saved per view for fit.py


def table_pixels(v, src, truth, T_bt, ext):
    """Candidate table points in the camera frame for one depth source."""
    if src == "aligned":
        depth, K = v["depth"], v["K"]
    else:
        depth, K = v["depth_raw"], v["Kd"]
    pc, px = C.deproject(depth, K, v["scale"], stride=2, lo=MIN_R, hi=MAX_R)
    if v["mask"] is not None:
        m = cv2.dilate(v["mask"].astype(np.uint8), np.ones((MASK_DILATE_PX, MASK_DILATE_PX), np.uint8)) > 0
        keep = ~m[px[:, 0], px[:, 1]]
        pc, px = pc[keep], px[keep]
    pb = C.to_base(pc, v["T"])
    d = pb @ truth[:3] + truth[3]
    pt = (pb - T_bt[:3, 3]) @ T_bt[:3, :3]
    gate = (np.abs(d) < GATE_M) & (pt[:, 0] > INSET_M) & (pt[:, 0] < ext[0] - INSET_M) & \
           (pt[:, 1] > INSET_M) & (pt[:, 1] < ext[1] - INSET_M)
    return pc[gate], px[gate]


def fit_plane(pc):
    """RANSAC + LSQ plane n.p + d = 0 in the given frame; (plane, inlier mask, rms m)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    plane, idx = pcd.segment_plane(distance_threshold=RANSAC_M, ransac_n=3, num_iterations=1000)
    inl = np.zeros(len(pc), bool)
    inl[idx] = True
    for _ in range(2):
        P = pc[inl]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[-1]
        plane = np.array([*n, -n @ c])
        r = pc @ plane[:3] + plane[3]
        inl = np.abs(r) < REFINE_M
    rms = float(np.sqrt(np.mean((pc[inl] @ plane[:3] + plane[3]) ** 2)))
    return plane, inl, rms


def measure(v, src, truth, T_bt, ext):
    pc, px = table_pixels(v, src, truth, T_bt, ext)
    if len(pc) < MIN_INLIERS:
        return None, None
    plane_c, inl, rms = fit_plane(pc)
    if inl.sum() < MIN_INLIERS:
        return None, None
    T = v["T"]
    plane_b = C.plane_to_frame(plane_c, T)
    if plane_b[2] < 0:
        plane_b, plane_c = -plane_b, -plane_c
    # the optical-axis hit on the probed slab, and the fitted plane's height there
    hit = C.ray_hit(truth, T[:3, 3], T[:3, 2])
    if hit is None:
        return None, None
    z_fit = -(plane_b[0] * hit[0] + plane_b[1] * hit[1] + plane_b[3]) / plane_b[2]
    h = z_fit - hit[2]
    # range error along the optical axis: fitted plane hit vs truth hit
    hit_fit = C.ray_hit(plane_b, T[:3, 3], T[:3, 2])
    e_axis = (np.linalg.norm(hit_fit - T[:3, 3]) - np.linalg.norm(hit - T[:3, 3])) if hit_fit is not None else np.nan
    # tilt vector: fitted normal minus truth normal (small angle), base and camera frame
    tilt_b = plane_b[:3] - truth[:3]
    tilt_c = T[:3, :3].T @ tilt_b
    # per-pixel range residual against the truth plane, in the camera frame
    truth_c = C.plane_to_frame(truth, np.linalg.inv(T))
    P = pc[inl]
    r_meas = np.linalg.norm(P, axis=1)
    u = P / r_meas[:, None]
    r_truth = -truth_c[3] / (u @ truth_c[:3])
    dr = r_meas - r_truth
    ok = (r_truth > 0.05) & (r_truth < 1.2)
    prof = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = ok & (r_truth >= lo) & (r_truth < hi)
        if s.sum() >= 50:
            prof.append({"r": round(float((lo + hi) / 2), 3), "n": int(s.sum()), "dr_med_mm": round(float(np.median(dr[s]) * 1000), 2),
                         "dr_p25_mm": round(float(np.percentile(dr[s], 25) * 1000), 2), "dr_p75_mm": round(float(np.percentile(dr[s], 75) * 1000), 2)})
    # the image-radial pattern of the residual (H3): residual vs normalised radius
    pxi = px[inl]
    K = v["K"] if src == "aligned" else v["Kd"]
    rad = np.hypot((pxi[:, 1] - K[0, 2]) / K[0, 0], (pxi[:, 0] - K[1, 2]) / K[1, 1])
    radial = []
    for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.5)]:
        s = ok & (rad >= lo) & (rad < hi)
        if s.sum() >= 50:
            radial.append({"rad": [lo, hi], "n": int(s.sum()), "dr_med_mm": round(float(np.median(dr[s]) * 1000), 2)})
    rec = {"run": v["run"], "step": v["step"], "id": v["id"], "src": src, "family": v["family"], "address": v["address"],
           "rot": v["rot"], "elev_deg": round(C.elevation_deg(T), 1), "elev_class": C.elev_class(C.elevation_deg(T)),
           "cam_height_m": round(float(T[2, 3] - hit[2]), 4), "range_hit_m": round(float(np.linalg.norm(hit - T[:3, 3])), 4),
           "hit": hit.round(4).tolist(), "n_candidates": int(len(pc)), "n_inliers": int(inl.sum()), "rms_mm": round(rms * 1000, 2),
           "h_mm": round(float(h * 1000), 2), "e_axis_mm": round(float(e_axis * 1000), 2),
           "tilt_deg": round(float(np.degrees(np.arccos(np.clip(plane_b[:3] @ truth[:3], -1, 1)))), 3),
           "tilt_base_mrad": (tilt_b[:2] * 1000).round(2).tolist(), "tilt_cam_mrad": (tilt_c * 1000).round(2).tolist(),
           "plane_cam": plane_c.round(6).tolist(), "plane_base": plane_b.round(6).tolist(),
           "r_truth_med_m": round(float(np.median(r_truth[ok])), 3), "dr_med_mm": round(float(np.median(dr[ok]) * 1000), 2),
           "profile": prof, "radial": radial}
    rng = np.random.default_rng(0)
    sel = rng.choice(np.flatnonzero(inl), min(KEEP, int(inl.sum())), replace=False)
    return rec, {"pc": pc[sel].astype(np.float32), "px": px[sel].astype(np.int16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    truth, T_bt, ext = C.table_truth()
    out_dir = C.DATA / "table"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, used = [], []
    for v in C.views(a.runs):
        row = {}
        for src in ("aligned", "raw"):
            rec, keep = measure(v, src, truth, T_bt, ext)
            if rec is None:
                print(f"{v['run']} {v['step']:2d} {src:7s}: no table plane")
                continue
            np.savez_compressed(out_dir / f"{v['run']}_{v['step']:03d}_{src}.npz", **keep, T=v["T"], K=(v["K"] if src == "aligned" else v["Kd"]))
            row[src] = rec
        if "aligned" in row:
            r = row["aligned"]
            if "raw" in row:
                r["i4_dh_mm"] = round(row["raw"]["h_mm"] - r["h_mm"], 2)
                r["i4_dtilt_mrad"] = round(float(np.linalg.norm(np.array(row["raw"]["tilt_base_mrad"]) - np.array(r["tilt_base_mrad"]))), 2)
                r["raw"] = {k: row["raw"][k] for k in ("h_mm", "e_axis_mm", "tilt_deg", "tilt_cam_mrad", "n_inliers", "rms_mm", "radial")}
            recs.append(r)
            used.append({"run": v["run"], "step": v["step"]})
            print(f"{v['run']:26s} {v['step']:2d} {v['family']} {str(v['address']):16s} elev {r['elev_deg']:5.1f} range {r['range_hit_m']:.3f} "
                  f"h {r['h_mm']:6.1f} e_axis {r['e_axis_mm']:6.1f} tilt {r['tilt_deg']:.2f} inl {r['n_inliers']:6d} rms {r['rms_mm']:.1f} "
                  f"I4 dh {r.get('i4_dh_mm', float('nan')):5.1f}")
    json.dump({"views": recs, "truth_plane": truth.tolist(),
               "settings": {"min_r": MIN_R, "max_r": MAX_R, "gate_m": GATE_M, "inset_m": INSET_M, "mask_dilate_px": MASK_DILATE_PX,
                            "ransac_m": RANSAC_M, "refine_m": REFINE_M, "min_inliers": MIN_INLIERS}},
              open(C.EVID / "table.json", "w"), indent=1, cls=C.NpEncoder)
    C.run_record(out_dir, used, {"min_r": MIN_R, "max_r": MAX_R, "gate_m": GATE_M, "ransac_m": RANSAC_M, "refine_m": REFINE_M})
    if not a.no_fig:
        figures(recs)


def figures(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"top": "#1f77b4", "oblique": "#ff7f0e", "grazing": "#2ca02c"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for r in recs:
        xs = [p["r"] for p in r["profile"]]
        ys = [p["dr_med_mm"] for p in r["profile"]]
        ax[0].plot(xs, ys, "-", color=col[r["elev_class"]], alpha=0.45, lw=1)
    for k, c in col.items():
        ax[0].plot([], [], color=c, label=k)
    rr = np.linspace(0.1, 0.9, 50)
    for s, ls in ((0.02, ":"), (0.04, "--")):
        ax[0].plot(rr, -s * rr * 1000, "k", ls=ls, lw=1, label=f"range {s*100:.0f} % short")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("range to the probed slab along the pixel ray  r (m)")
    ax[0].set_ylabel("measured range - truth  dr (mm, median per 5 cm bin)")
    ax[0].set_title("I1: range residual on the table, one line per view")
    ax[0].legend(fontsize=8)
    for r in recs:
        ax[1].scatter(r["range_hit_m"], r["e_axis_mm"], color=col[r["elev_class"]], s=22,
                      marker="o" if r["family"] == "A" else "^", alpha=0.8)
    ax[1].plot(rr, -0.02 * rr * 1000, "k:", lw=1)
    ax[1].plot(rr, -0.04 * rr * 1000, "k--", lw=1)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("range along the optical axis to the slab (m)")
    ax[1].set_ylabel("e_axis: fitted - truth along the axis (mm)")
    ax[1].set_title("range error at the optical axis (o family A, ^ family B)")
    for r in recs:
        t = r["tilt_cam_mrad"]
        ax[2].scatter(t[0], t[1], color=col[r["elev_class"]], s=22, marker="o" if r["family"] == "A" else "^", alpha=0.8)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].axvline(0, color="k", lw=0.5)
    ax[2].set_xlabel("tilt about camera x  (mrad)")
    ax[2].set_ylabel("tilt about camera y  (mrad)")
    ax[2].set_title("fitted table normal - slab normal, in the camera frame")
    ax[2].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(C.EVID / "table_profile.png", dpi=130)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for r in recs:
        ax2.scatter(r["cam_height_m"], r["h_mm"], color=col[r["elev_class"]], s=24, marker="o" if r["family"] == "A" else "^")
    hh = np.linspace(0, 0.5, 20)
    ax2.plot(hh, 0.04 * hh * 1000, "k--", lw=1, label="4 % of the camera height")
    ax2.set_xlabel("camera height above the slab (m)")
    ax2.set_ylabel("h: fitted table height - probed slab, at the axis hit (mm)")
    ax2.set_title("I1: table height error vs camera height")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(C.EVID / "table_height.png", dpi=130)


if __name__ == "__main__":
    main()
