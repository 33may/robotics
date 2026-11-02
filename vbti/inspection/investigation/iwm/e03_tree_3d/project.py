"""e03: project the e02 tree nodes of a frame into the run's base frame.

Per node: the mask's pixels through the step's aligned depth (colour intrinsics,
depth scale) and T_base_cam -> points in the base frame, colours from rgb, centroid
and extent in mm. Saved per frame in inspection/data/iwm/e03_tree_3d/<frame>/.

Orientation: rgb / depth_aligned / mask on disk are upright, T_base_cam and the
intrinsics are raw; on rgb_rotation_deg == 180 views the pixels are turned back before
deprojection (a 180-degree turn with raw intrinsics would mirror the geometry).

Run:  p inspection/investigation/iwm/e03_tree_3d/project.py [--frames c5.6 ...] [--variant full]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from inspection.cell.geometry import deproject
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.record.run import Run

HERE = Path(__file__).resolve().parent
E02 = C.ROOT / "inspection" / "data" / "iwm" / "e02_agent_tree"
DATA = C.ROOT / "inspection" / "data" / "iwm" / "e03_tree_3d"


def load_tree(frame, variant="full", tag="final"):
    d = E02 / frame["id"] / variant
    tree = json.load(open(d / f"tree_{tag}.json"))
    masks = C.load_packed(d / f"masks_{tag}.npz")
    ids = json.load(open(d / f"masks_{tag}_ids.json"))
    return tree, {i: m for i, m in zip(ids, masks)}


def session_intrinsics(run_dir):
    s = json.load(open(run_dir / "session.json"))
    return s["intrinsics"]["color"], float(s["depth_scale_m_per_unit"])


def project_frame(frame, variant="full"):
    run_dir = C.RUNS_DIR / frame["run"]
    p = C.frame_paths(frame)
    step = json.load(open(p["step"]))
    rot = int(step.get("rgb_rotation_deg", 0) or 0)
    T = np.array(step["T_base_cam"])
    intr, scale = session_intrinsics(run_dir)
    depth = np.load(p["depth"])
    rgb = C.load_rgb(frame)
    tree, masks = load_tree(frame, variant)
    if rot == 180:                      # back to the raw sensor orientation
        depth = depth[::-1, ::-1]
        rgb = rgb[::-1, ::-1]
        masks = {i: m[::-1, ::-1] for i, m in masks.items()}
    assert depth.shape == rgb.shape[:2], (depth.shape, rgb.shape)
    pts_cam, px = deproject(depth, intr, scale, stride=1, return_px=True)
    pts_base = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
    vv, uu = px[:, 0], px[:, 1]
    cols = rgb[vv, uu]
    run = Run.load(run_dir)
    fused = run.fused()
    tree_kd = cKDTree(fused[0]) if fused is not None else None

    out = DATA / frame["id"] / variant
    out.mkdir(parents=True, exist_ok=True)
    nodes = []
    all_pts, all_cols, all_idx = [], [], []
    for nd in tree["nodes"]:
        i = nd["id"]
        m = masks[i]
        sel = m[vv, uu]
        P, Cc = pts_base[sel], cols[sel]
        rec = {**{k: nd[k] for k in ("id", "parent", "depth", "name", "text", "area", "bbox", "source", "children")},
               "n_points": int(len(P)), "px_with_depth": round(float(len(P)) / max(nd["area"], 1), 3)}
        if len(P):
            lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
            rec["centroid_m"] = [round(float(v), 4) for v in P.mean(0)]
            rec["extent_mm"] = [round(float(v) * 1000, 1) for v in (hi - lo)]
            if tree_kd is not None:
                d, _ = tree_kd.query(P[:: max(1, len(P) // 2000)])
                rec["nn_to_fused_mm"] = round(float(np.median(d)) * 1000, 2)
        nodes.append(rec)
        all_pts.append(P)
        all_cols.append(Cc)
        all_idx.append(np.full(len(P), i, np.int32))
    np.savez_compressed(out / "nodes.npz", points=np.concatenate(all_pts).astype(np.float32),
                        colors=np.concatenate(all_cols).astype(np.uint8),
                        node=np.concatenate(all_idx))
    json.dump({"frame": frame, "variant": variant, "nodes": nodes,
               "run_fused_points": int(len(fused[0])) if fused is not None else 0},
              open(out / "nodes.json", "w"), indent=1)
    C.run_record(out, {"id": frame["id"], "run": frame["run"], "step": frame["step"]},
                 {"name": "deproject depth_aligned + colour intrinsics"},
                 {"variant": variant, "rot": rot, "stride": 1})
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--variant", default="full")
    args = ap.parse_args()
    for f in C.frames(args.frames):
        nodes = project_frame(f, args.variant)
        print(f"== {f['id']}  ({len(nodes)} nodes)")
        for nd in nodes:
            ext = nd.get("extent_mm")
            print(f"  #{nd['id']:<3d} d{nd['depth']} {str(nd['name'])[:28]:28s} {nd['n_points']:6d} pts "
                  f"({100 * nd['px_with_depth']:3.0f}% of px)  extent {ext}  nn {nd.get('nn_to_fused_mm')} mm")


if __name__ == "__main__":
    main()
