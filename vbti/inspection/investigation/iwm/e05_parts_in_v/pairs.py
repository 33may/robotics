"""e05 pairs (RQ2): the same physical part lifted from two views of one run.

V coloured from the FIRST view of the run; the SECOND view's tree lifted separately (not
assigned). Per truth pair (evidence/truth_pairs.json): row-set IoU, centroid offset (mm), the
view's rigid vector (median offset over pairs wider than 20 mm), the IoU after removing it
(diagnostic only; V is never corrected), and the render test: the first view's labels rendered
from the second view's pose against the second view's own masks.

Run:  p vbti/inspection/investigation/iwm/e05_parts_in_v/pairs.py [--sizes 1 2] [--lift L3]
Outputs  evidence/pairs.json, evidence/pairs_<run>_<mm>mm.jpg, evidence/pairs_render_<view>_<mm>mm.jpg
"""
import argparse
import json

import numpy as np
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface

WIDE_MM = 20.0
VERIFY_IOU = 0.3        # an uncertain face pair counts as the same face above this row IoU


def subtree_rows(view, rows_by_node, nid):
    out, todo = [], [nid]
    while todo:
        i = todo.pop()
        out.append(rows_by_node.get(i, np.empty(0, np.int64)))
        todo.extend(view["nodes"][i]["children"])
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.int64)


def row_iou(a, b):
    inter = len(np.intersect1d(a, b, assume_unique=True))
    union = len(np.union1d(a, b))
    return inter / union if union else 0.0


def pixels_of_rows(surface, rows, K, T, w=C5.W, h=C5.H):
    u, v, z = C5.project(surface.centres(rows), K, T)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), bool)
    m[v[ok].astype(int), u[ok].astype(int)] = True
    return m


def chosen_lift(arg):
    if arg:
        return arg
    p = C5.EVID / "lift.json"
    return json.load(open(p))["chosen"] if p.exists() else "L3"


def run_pair(run, mm, lift_name, truth):
    first, second = C5.ORDER[run][0], C5.ORDER[run][1]
    va, vb = C5.load_view(first), C5.load_view(second)
    vbg, size = C5.V(run, mm)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    rows_a, _ = L.LIFTS[lift_name](va, surf, vbg, size)
    stv = STV(surf)
    m = L.colour(stv, va, rows_a)
    rows_b, _ = L.LIFTS[lift_name](vb, surf, vbg, size)
    pairs = [p for p in truth["pairs"] if p["a"] == first and p["b"] == second]
    # the uncertain face pair, decided by geometry
    for u in truth.get("uncertain", []):
        if u.get("a") == first and u.get("b") == second and "b_id" in u:
            A, B = stv.parts[m[u["a_id"]]].rows, subtree_rows(vb, rows_b, u["b_id"])
            share = len(np.intersect1d(A, B, assume_unique=True)) / max(len(B), 1)
            iou_ = row_iou(A, B)
            u["geometry"] = {"iou_rows": round(iou_, 3), "share_of_b_on_a": round(share, 3), "mm": mm}
            if iou_ >= VERIFY_IOU or share >= 0.5:
                pairs.append({"a": first, "a_id": u["a_id"], "b": second, "b_id": u["b_id"],
                              "what": u["what"] + f" -> geometry says SAME (iou {iou_:.2f}, share {share:.2f})",
                              "geometry_verified": True})
    recs = []
    for p in pairs:
        A = stv.parts[m[p["a_id"]]].rows
        B = subtree_rows(vb, rows_b, p["b_id"])
        if not len(A) or not len(B):
            recs.append({**p, "mm": mm, "empty": True})
            continue
        ca, cb = surf.centres(A).mean(0), surf.centres(B).mean(0)
        ext = L.extent_mm(surf, A)
        recs.append({**p, "mm": mm, "rows_a": int(len(A)), "rows_b": int(len(B)), "iou_rows": round(row_iou(A, B), 3),
                     "share_b_on_a": round(len(np.intersect1d(A, B, assume_unique=True)) / len(B), 3),
                     "offset_mm": [round(float(x) * 1000, 2) for x in (cb - ca)],
                     "offset_norm_mm": round(float(np.linalg.norm(cb - ca)) * 1000, 2),
                     "width_mm": ext[0], "wide": ext[0] >= WIDE_MM})
    wide = [r for r in recs if not r.get("empty") and r["wide"] and r["a_id"] != 0]
    vec = np.median([r["offset_mm"] for r in wide], axis=0) / 1000 if wide else np.zeros(3)
    # the overlap after removing the rigid vector (diagnostic)
    for r in recs:
        if r.get("empty"):
            continue
        B = subtree_rows(vb, rows_b, r["b_id"])
        Bs = surf.rows_near(surf.centres(B) - vec, 1.5)
        Bs = np.unique(Bs[Bs >= 0])
        r["iou_rows_shifted"] = round(row_iou(stv.parts[m[r["a_id"]]].rows, Bs), 3)
    # the render test: the first view's labels from the second view's pose vs the second view's masks
    lab, depth, _, col = L.label_render(stv, vbg, size, vb["T"], vb["K"], color=True)
    for r in recs:
        if r.get("empty"):
            continue
        pp = L.part_pixels(lab, stv, m[r["a_id"]])
        mb = vb["masks"][r["b_id"]]
        i, pr, rc = L.iou(pp, mb)
        r["render_iou"] = round(i, 3)
        if pp.any() and mb.any():
            ca_px = np.array(np.nonzero(pp)).mean(1)
            cb_px = np.array(np.nonzero(mb)).mean(1)
            r["render_offset_px"] = [round(float(x), 1) for x in (cb_px - ca_px)]
    out = {"run": run, "mm": mm, "lift": lift_name, "first": first, "second": second,
           "rigid_vector_mm": [round(float(x) * 1000, 2) for x in vec], "n_wide_pairs": len(wide), "pairs": recs}
    figure_pairs(run, mm, surf, stv, m, vb, rows_b, recs, col)
    figure_render(vb, lab, stv, m, recs, col, mm)
    return out, truth


def _upright(a, rot):
    return a[::-1, ::-1] if rot == 180 else a


def figure_pairs(run, mm, surf, stv, m, vb, rows_b, recs, col):
    """Per pair: V's colour render from the second view's pose with the first view's rows (green) and
    the second view's rows (magenta) projected; the overlap is where both land."""
    K, T, rot = vb["K"], vb["T"], vb["rot"]
    tiles = []
    for r in recs:
        if r.get("empty"):
            continue
        A = pixels_of_rows(surf, stv.parts[m[r["a_id"]]].rows, K, T)
        B = pixels_of_rows(surf, subtree_rows(vb, rows_b, r["b_id"]), K, T)
        A, B, img = _upright(A, rot), _upright(B, rot), _upright(col, rot)
        ys, xs = np.nonzero(A | B)
        if not len(ys):
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = C.expand_box([xs.min(), ys.min(), xs.max(), ys.max()], (h, w), 0.2)
        tile = C.outlines(img[y0:y1, x0:x1], [A[y0:y1, x0:x1], B[y0:y1, x0:x1]], [(0, 255, 0), (255, 0, 255)],
                          thick=1, dim=0.0)
        tile = C.resize(tile, 420)
        tiles.append(C.label(tile, f"{r['a']}#{r['a_id']} vs {r['b']}#{r['b_id']} {stv.parts[m[r['a_id']]].name[:18]}: "
                                   f"IoU {r['iou_rows']:.2f} off {r['offset_norm_mm']:.1f} mm", size=13))
    if not tiles:
        return
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    g = C.title_band(C.grid(tiles, cols=4), f"{run} {mm} mm: the same part from {recs[0]['a']} (green) and "
                                            f"{recs[0]['b']} (magenta), seen from {recs[0]['b']}'s pose")
    Image.fromarray(g).save(C5.EVID / f"pairs_{run}_{mm}mm.jpg", quality=88)


def figure_render(vb, lab, stv, m, recs, col, mm):
    """The second view's frame with its own masks (magenta) and the first view's parts rendered into it (green)."""
    rot = vb["rot"]
    rgb = _upright(vb["rgb"], rot)
    ga = np.zeros(lab.shape, bool)
    gb = np.zeros(lab.shape, bool)
    for r in recs:
        if r.get("empty") or r["a_id"] == 0:
            continue
        ga |= L.part_pixels(lab, stv, m[r["a_id"]])
        gb |= vb["masks"][r["b_id"]]
    ga, gb = _upright(ga, rot), _upright(gb, rot)
    left = C.outlines(rgb, [gb, ga], [(255, 0, 255), (0, 255, 0)], thick=2, dim=0.0)
    right = C.outlines(_upright(col, rot), [gb, ga], [(255, 0, 255), (0, 255, 0)], thick=2, dim=0.0)
    g = C.title_band(np.concatenate([C.resize(left, 900), C.resize(right, 900)], 1),
                     f"{vb['id']} at {mm} mm: its own masks (magenta) vs the earlier view's parts rendered from V (green)")
    Image.fromarray(g).save(C5.EVID / f"pairs_render_{vb['id']}_{mm}mm.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--lift")
    ap.add_argument("--runs", nargs="*", default=list(C5.ORDER))
    a = ap.parse_args()
    lift_name = chosen_lift(a.lift)
    truth = json.load(open(C5.EVID / "truth_pairs.json"))
    results = []
    for mm in a.sizes:
        for run in a.runs:
            out, truth = run_pair(run, mm, lift_name, truth)
            results.append(out)
            small = [p for p in out["pairs"] if not p.get("empty") and not p["wide"]]
            wide = [p for p in out["pairs"] if not p.get("empty") and p["wide"] and p["a_id"] != 0]
            print(f"{run} {mm} mm {lift_name}: rigid vector {out['rigid_vector_mm']} mm over {len(wide)} wide pairs; "
                  f"IoU wide {np.median([p['iou_rows'] for p in wide]) if wide else float('nan'):.2f} "
                  f"small {np.median([p['iou_rows'] for p in small]) if small else float('nan'):.2f}; "
                  f"shifted wide {np.median([p['iou_rows_shifted'] for p in wide]) if wide else float('nan'):.2f} "
                  f"small {np.median([p['iou_rows_shifted'] for p in small]) if small else float('nan'):.2f}")
            for p in out["pairs"]:
                if not p.get("empty"):
                    print(f"   #{p['a_id']:<3}->#{p['b_id']:<3} w {p['width_mm']:6.1f} IoU {p['iou_rows']:.2f} "
                          f"shifted {p['iou_rows_shifted']:.2f} off {p['offset_norm_mm']:5.1f} mm render IoU {p['render_iou']:.2f}")
    json.dump({"results": results, "settings": {"wide_mm": WIDE_MM, "verify_iou": VERIFY_IOU, "lift": lift_name}},
              open(C5.EVID / "pairs.json", "w"), indent=1)
    # geometry-verified pairs go back into the truth file, marked
    verified = [p for r in results for p in r["pairs"] if p.get("geometry_verified") and r["mm"] == a.sizes[0]]
    if verified:
        have = {(p["a"], p["a_id"], p["b"], p["b_id"]) for p in truth["pairs"]}
        for p in verified:
            if (p["a"], p["a_id"], p["b"], p["b_id"]) not in have:
                truth["pairs"].append({k: p[k] for k in ("a", "a_id", "b", "b_id", "what", "geometry_verified")})
    json.dump(truth, open(C5.EVID / "truth_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
