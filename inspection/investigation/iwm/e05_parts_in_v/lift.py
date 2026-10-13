"""e05 lift (RQ1): a view's e02 tree into V's surface rows, the STV colouring, the label render.

Two lifts of a node's mask:
  L1  through the view's own depth (e03's way), each point snapped to the nearest surface row
      within SNAP_M; unmatched points counted.
  L2  through V: ray-cast V from the view's pose, the row each mask pixel's ray hits.
The node's rows = its own mask's rows; `colour` assigns them to the node's part, and STV.assign
adds them to every ancestor (containment by construction).

Check (Anton's validation): render the labels from the same pose and compare with the mask per
node: IoU, precision, recall; plus the smear (share of the node's rows > OFF_PLANE_M off the
RANSAC plane of its face) and the hit / depth share (TSDF and depth holes).

Run:  p inspection/investigation/iwm/e05_parts_in_v/lift.py --frames c5.6 b1.4 --sizes 1 2 --lifts L1 L2
Outputs  evidence/lift.json, evidence/lift_<frame>_<mm>mm_<lift>.jpg,
         inspection/data/iwm/e05_parts_in_v/<run>/<mm>mm/single_<frame>_<lift>/ (STV + run.json)
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d
from PIL import Image

from inspection.cell.geometry import deproject
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface

SNAP_M = 0.003          # L1: a depth point snaps to a row within this
NEAR_VOX = 1.5          # L2: a hit point takes the nearest row within this many voxels
GATE_M = 0.010          # L3: V's hit depth must be within this of the view's own depth
OFF_PLANE_M = 0.008     # smear: a row this far off the face plane is "smeared" (e03's number)
L1_MIN_RANGE_M = 0.05   # deproject's default 0.13 m is the gripper mask of live runs; close-ups go below it
SMALL_MM = 20.0         # nodes narrower than this are "small" (ports, labels)
FIT_PTS = 20000         # RANSAC subsample
TILE_W = 560            # figure tile width (frame crop | label render crop)


def node_order(tree):
    return [n["id"] for n in sorted(tree["nodes"], key=lambda n: (n["depth"], n["id"]))]


# ------------------------------------------------------------------ lifts
def lift_L1(view, surface, snap_m=SNAP_M):
    pts_cam, px = deproject(view["depth"], view["intr"], view["scale"], stride=1, return_px=True,
                            min_range=L1_MIN_RANGE_M)
    T = view["T"]
    pts = pts_cam @ T[:3, :3].T + T[:3, 3]
    rows_all = surface.rows_near(pts, max_voxels=snap_m / surface.size)
    vv, uu = px[:, 0], px[:, 1]
    out, stats = {}, {}
    for nid, m in view["masks"].items():
        sel = m[vv, uu]
        r = rows_all[sel]
        ok = r >= 0
        out[nid] = np.unique(r[ok])
        n_px = int(m.sum())
        stats[nid] = {"n_px": n_px, "depth_share": round(float(sel.sum() / max(n_px, 1)), 3),
                      "n_rows": int(len(out[nid])), "miss_share": round(float((~ok).mean()) if len(ok) else 1.0, 3)}
    return out, stats


def lift_L2(view, surface, vbg, size_m):
    depth, _ = C5.cast(vbg, size_m, view["T"], view["K"])
    pts, px = C5.hits_to_points(depth, view["K"], view["T"])
    rows_all = surface.rows_near(pts, NEAR_VOX)
    vv, uu = px[:, 0], px[:, 1]
    out, stats = {}, {}
    for nid, m in view["masks"].items():
        sel = m[vv, uu]
        r = rows_all[sel]
        ok = r >= 0
        out[nid] = np.unique(r[ok])
        n_px = int(m.sum())
        stats[nid] = {"n_px": n_px, "hit_share": round(float(sel.sum() / max(n_px, 1)), 3),
                      "n_rows": int(len(out[nid])), "miss_share": round(float((~ok).mean()) if len(ok) else 1.0, 3)}
    return out, stats


def lift_L3(view, surface, vbg, size_m, gate_m=GATE_M):
    """L2 gated by the view's own depth: a hit counts only if V's depth at that pixel is within
    gate_m of the recorded depth; pixels without recorded depth are gated against the node's
    median gated depth. Removes rays that fly through TSDF holes onto interior surfaces."""
    depth, _ = C5.cast(vbg, size_m, view["T"], view["K"])
    own = view["depth"].astype(np.float64) * view["scale"]
    own[(view["depth"] == 65535) | (own < 0.03) | (own > 1.0)] = 0
    pts, px = C5.hits_to_points(depth, view["K"], view["T"])
    rows_all = surface.rows_near(pts, NEAR_VOX)
    vv, uu = px[:, 0], px[:, 1]
    dv, do = depth[vv, uu], own[vv, uu]
    has_own = do > 0
    close = has_own & (np.abs(dv - do) < gate_m)
    out, stats = {}, {}
    for nid, m in view["masks"].items():
        sel = m[vv, uu]
        ref = np.median(dv[sel & close]) if (sel & close).any() else np.median(dv[sel]) if sel.any() else 0.0
        keep = sel & (close | (~has_own & (np.abs(dv - ref) < gate_m)))
        r = rows_all[keep]
        ok = r >= 0
        out[nid] = np.unique(r[ok])
        n_px = int(m.sum())
        stats[nid] = {"n_px": n_px, "hit_share": round(float(sel.sum() / max(n_px, 1)), 3),
                      "gated_share": round(float(keep.sum() / max(sel.sum(), 1)), 3),
                      "n_rows": int(len(out[nid])), "miss_share": round(float((~ok).mean()) if len(ok) else 1.0, 3)}
    return out, stats


LIFTS = {"L1": lambda view, surf, vbg, size: lift_L1(view, surf),
         "L2": lambda view, surf, vbg, size: lift_L2(view, surf, vbg, size),
         "L3": lambda view, surf, vbg, size: lift_L3(view, surf, vbg, size)}


# ------------------------------------------------------------------ colouring and rendering
def colour(stv, view, rows_by_node):
    """The view's tree into the STV: node -> part (root -> stv.root). Returns {node id: part id}."""
    nodes = view["nodes"]
    m = {}
    for nid in node_order(view["tree"]):
        nd = nodes[nid]
        rows = rows_by_node.get(nid, np.empty(0, np.int64))
        if nd["parent"] is None:
            part = stv.root
            if stv.parts[part].name == "object" and nd["name"]:
                stv.parts[part].name = nd["name"]
            stv.assign(part, rows, detection=(view["id"], nid))
        else:
            part = stv.add(m[nd["parent"]], nd["name"] or f"#{nid}", rows=rows, text=nd.get("text"),
                           detection=(view["id"], nid))
        m[nid] = part
    stv.meta[view["id"]] = m
    return m


def label_render(stv, vbg, size_m, T, K, w=C5.W, h=C5.H, color=False):
    """Labels seen from pose T: HxW int32, -1 no hit, -2 hit but no row; plus depth, normal[, colour]."""
    r = C5.cast(vbg, size_m, T, K, w, h, color=color)
    depth = r[0]
    pts, px = C5.hits_to_points(depth, K, T)
    rows = stv.s.rows_near(pts, NEAR_VOX)
    lab = np.full((h, w), -1, np.int32)
    lab[px[:, 0], px[:, 1]] = np.where(rows >= 0, stv.label[np.maximum(rows, 0)], -2)
    return (lab,) + tuple(r)


def part_pixels(lab, stv, part):
    return np.isin(lab, [part] + stv.descendants(part))


def iou(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return (inter / union if union else 0.0, inter / a.sum() if a.sum() else 0.0, inter / b.sum() if b.sum() else 0.0)


# ------------------------------------------------------------------ geometry of a row set
def fit_plane(P, thr=0.003, seed=0):
    """(normal, d, inlier ratio) of the RANSAC plane n.x + d = 0 over P (subsampled)."""
    if len(P) < 10:
        return None, None, 0.0
    if len(P) > FIT_PTS:
        P = P[np.random.default_rng(seed).choice(len(P), FIT_PTS, replace=False)]
    o3d.utility.random.seed(seed)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    model, inl = pc.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
    n = np.asarray(model[:3])
    return n / np.linalg.norm(n), float(model[3]) / np.linalg.norm(n), len(inl) / len(P)


def smear(surface, rows, ref_rows, off=OFF_PLANE_M):
    if len(rows) < 5 or len(ref_rows) < 50:
        return None
    n, d, _ = fit_plane(surface.centres(ref_rows))
    if n is None:
        return None
    dist = np.abs(surface.centres(rows) @ n + d)
    return round(float((dist > off).mean()), 3)


def extent_mm(surface, rows):
    """PCA box of the rows (2-98 percentile), sorted largest first, in mm."""
    if len(rows) < 5:
        return [0.0, 0.0, 0.0]
    P = surface.centres(rows)
    Q = P - P.mean(0)
    _, vec = np.linalg.eigh(Q.T @ Q)
    proj = Q @ vec
    e = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
    return [round(float(v) * 1000, 1) for v in sorted(e, reverse=True)]


def mask_width_mm(view, nid):
    """The node's size from its own mask: the longer bbox side in px times the pixel footprint at the
    median recorded depth under the mask. The same number for every lift."""
    nd, m = view["nodes"][nid], view["masks"][nid]
    z = view["depth"][m].astype(np.float64) * view["scale"]
    z = z[(z > 0.03) & (z < 1.0)]
    if not len(z):
        return 0.0
    px = max(nd["bbox"][2] - nd["bbox"][0], nd["bbox"][3] - nd["bbox"][1])
    return round(float(px * np.median(z) / view["intr"]["fx"] * 1000), 1)


def face_of(view, nid):
    """The node's ancestor at depth 1 (itself at depth 0 / 1)."""
    nd = view["nodes"][nid]
    while nd["depth"] > 1:
        nd = view["nodes"][nd["parent"]]
    return nd["id"]


# ------------------------------------------------------------------ the figure
def _upright(a, rot):
    return a[::-1, ::-1] if rot == 180 else a


def figure(view, stv, m, lab, col, recs, path, title):
    rot = view["rot"]
    rgb, colv, labv = _upright(view["rgb"], rot), _upright(col, rot), _upright(lab, rot)
    h, w = rgb.shape[:2]
    tiles = []
    for nid in node_order(view["tree"]):
        if nid == 0:
            continue
        nd, rec = view["nodes"][nid], recs[nid]
        mask = _upright(view["masks"][nid], rot)
        pp = _upright(part_pixels(lab, stv, m[nid]), rot)
        x0, y0, x1, y1 = C.expand_box(nd["bbox"], (h, w), 0.15)        # e02 boxes are upright already
        left = C.outlines(rgb[y0:y1, x0:x1], [mask[y0:y1, x0:x1]], [(0, 255, 0)], thick=2, dim=0.1)
        right = C.outlines(colv[y0:y1, x0:x1], [mask[y0:y1, x0:x1], pp[y0:y1, x0:x1]],
                           [(0, 255, 0), (255, 0, 255)], thick=2, dim=0.1)
        tile = C.resize(np.concatenate([left, right], 1), TILE_W)
        txt = (f"#{nid} {nd['name'][:26]}  IoU {rec['iou']:.2f}  smear {100 * (rec['smear'] or 0):.0f}%  "
               f"{'hit' if 'hit_share' in rec else 'depth'} {100 * rec.get('hit_share', rec.get('depth_share', 0)):.0f}%")
        tiles.append(C.label(tile, txt, size=14))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    g = C.grid(tiles, cols=3)
    g = C.title_band(g, title)
    Image.fromarray(g).save(path, quality=88)


# ------------------------------------------------------------------ main
def run_one(view, lift_name, vbg, size_m, surf, mm, out_dir):
    t0 = time.time()
    rows, stats = LIFTS[lift_name](view, surf, vbg, size_m)
    t_lift = time.time() - t0
    stv = STV(surf)
    m = colour(stv, view, rows)
    stv.check()
    lab, depth, normal, col = label_render(stv, vbg, size_m, view["T"], view["K"], color=True)
    recs = {}
    for nid in node_order(view["tree"]):
        nd = view["nodes"][nid]
        pp = part_pixels(lab, stv, m[nid])
        i, p, r = iou(pp, view["masks"][nid])
        ext = extent_mm(surf, stv.parts[m[nid]].rows)
        width = mask_width_mm(view, nid)                       # lift-independent size class
        recs[nid] = {"frame": view["id"], "mm": mm, "lift": lift_name, "node": nid, "name": nd["name"],
                     "depth": nd["depth"], **stats[nid], "rows_with_children": int(len(stv.parts[m[nid]].rows)),
                     "iou": round(i, 3), "precision": round(p, 3), "recall": round(r, 3),
                     "smear": smear(surf, rows[nid], rows[face_of(view, nid)]) if nid != 0 else None,
                     "extent_mm": ext, "mask_width_mm": width, "small": width < SMALL_MM}
    stv.save(out_dir)
    C.run_record(out_dir, {"id": view["id"], "run": view["run"], "step": view["step"]},
                 {"name": f"lift {lift_name}", "V": "e04 scene"},
                 {"mm": mm, "snap_m": SNAP_M, "near_vox": NEAR_VOX, "shell_voxels": Surface.SHELL_VOXELS},
                 {"lift_s": round(t_lift, 2), **stv.summary()})
    figure(view, stv, m, lab, col, recs, C5.EVID / f"lift_{view['id']}_{mm}mm_{lift_name}.jpg",
           f"{view['id']} {lift_name} at {mm} mm: mask (green) vs the label render (magenta) from the same pose")
    return list(recs.values()), stv.summary()


def summarise(records):
    out = {}
    for key in sorted({(r["frame"], r["mm"], r["lift"]) for r in records}):
        sel = [r for r in records if (r["frame"], r["mm"], r["lift"]) == key and r["node"] != 0]
        for cls, name in ((True, "small"), (False, "large")):
            rs = [r for r in sel if r["small"] == cls]
            if not rs:
                continue
            out[f"{key[0]}_{key[1]}mm_{key[2]}_{name}"] = {
                "n": len(rs), "iou_median": round(float(np.median([r["iou"] for r in rs])), 3),
                "iou_min": round(float(min(r["iou"] for r in rs)), 3),
                "recall_median": round(float(np.median([r["recall"] for r in rs])), 3),
                "smear_median": round(float(np.median([r["smear"] or 0 for r in rs])), 3),
                "smear_max": round(float(max(r["smear"] or 0 for r in rs)), 3),
                "share_median": round(float(np.median([r.get("hit_share", r.get("depth_share")) for r in rs])), 3)}
    return out


def choose(summary):
    """The lift with the lowest max smear over all nodes, then the higher median IoU on small nodes."""
    score = {}
    for k, v in summary.items():
        lift = k.split("_")[-2]
        s = score.setdefault(lift, {"smear_max": 0.0, "iou_small": []})
        s["smear_max"] = max(s["smear_max"], v["smear_max"])
        if k.endswith("_small"):
            s["iou_small"].append(v["iou_median"])
    # smear within 0.05 of the best counts as equal (the RJ45 recess shows as ~10 % in every lift); then small-node IoU
    if not score:
        return "L3"
    floor = min(s["smear_max"] for s in score.values())
    cands = [l for l, s in score.items() if s["smear_max"] <= floor + 0.05]
    return max(cands, key=lambda l: (np.mean(score[l]["iou_small"] or [0]), l == "L3"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*", default=["c5.6", "b1.4"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--lifts", nargs="*", default=["L1", "L2", "L3"])
    a = ap.parse_args()
    C5.EVID.mkdir(exist_ok=True)
    records, runs = [], {}
    for mm in a.sizes:
        for fid in a.frames:
            view = C5.load_view(fid)
            vbg, size = C5.V(view["run"], mm)
            surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
            for lift in a.lifts:
                out_dir = C5.DATA / view["run"] / f"{mm}mm" / f"single_{fid}_{lift}"
                recs, summ = run_one(view, lift, vbg, size, surf, mm, out_dir)
                records += recs
                runs[f"{fid}_{mm}mm_{lift}"] = summ
                small = [r for r in recs if r["small"] and r["node"] != 0]
                print(f"{fid} {mm} mm {lift}: parts {summ['parts']} rows {summ['rows_labelled']:,} "
                      f"IoU median {np.median([r['iou'] for r in recs if r['node']]):.3f} "
                      f"small {np.median([r['iou'] for r in small]) if small else float('nan'):.3f} "
                      f"smear max {max(r['smear'] or 0 for r in recs):.3f}")
    summary = summarise(records)
    json.dump({"records": records, "summary": summary, "runs": runs, "chosen": choose(summary),
               "settings": {"snap_m": SNAP_M, "near_vox": NEAR_VOX, "off_plane_m": OFF_PLANE_M, "small_mm": SMALL_MM}},
              open(C5.EVID / "lift.json", "w"), indent=1)
    print("chosen lift:", choose(summary))
    for k, v in summary.items():
        print(f"  {k:28s} n {v['n']:2d} IoU med {v['iou_median']:.3f} min {v['iou_min']:.3f} "
              f"smear med {v['smear_median']:.3f} max {v['smear_max']:.3f} share {v['share_median']:.3f}")


if __name__ == "__main__":
    main()
