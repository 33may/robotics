"""e03 turn 2: two frames of the SAME run, both trees in 3D together.

Same run = same base frame, so no alignment: whatever gap shows between the two views'
copies of one part is the projection + pose error that a merge would have to absorb.

  evidence/quality/pair_<A>_<B>.jpg
    row 1: all nodes of A (orange) and B (cyan) on the dimmed fused cloud, from A's
           camera, B's camera, the side and the top
    rows 2+: matched parts (nearest B node to each A node in 3D, similar size), zoomed:
           facing the part and grazing along it, with the centroid gap in mm
  evidence/pairs_3d.json: every matched pair, its gap, and the gap along the face normal

Run:  p inspection/investigation/iwm/e03_tree_3d/pair3d.py [--pairs c5.6:c5.2 b1.1:b1.4]
"""
import argparse
import json

import numpy as np

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e03_tree_3d.quality import (EVID, HERE, F, fit_f, letterbox, look_at,
                                                              main_normal, proj_bbox, splat)

COL_A, COL_B = np.array((255, 140, 0), np.float32), np.array((0, 200, 255), np.float32)
DARK = (18, 20, 26)
MAX_GAP_M = 0.03


def painted(f, col, side_k=0.35, part_k=0.9):
    cols = f.Pc.astype(np.float32)
    k = np.array([side_k if f.nodes[n]["depth"] == 1 else part_k for n in f.Pn])[:, None]
    out = cols * (1 - k) + col * k
    keep = f.Pn != 0
    return f.P[keep], out[keep].astype(np.uint8)


def render(points, colors, T, focus, w=420, h=330, r=2, fill=0.8):
    fz, cx, cy = fit_f(focus, T, fill=fill)
    img = splat(points, colors, T, f=fz, r=r, c=(cx, cy))
    x0, y0, x1, y1 = proj_bbox(focus, T, f=fz, c=(cx, cy), margin=0.15)
    return letterbox(img[y0:y1, x0:x1], w, h, DARK)


def rot_about(center, eye, deg, axis=(0, 0, 1)):
    a = np.deg2rad(deg)
    k = np.array(axis, float) / np.linalg.norm(axis)
    v = eye - center
    v = v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k @ v) * (1 - np.cos(a))
    return center + v


def upright(f):
    return f.T @ np.diag([-1, -1, 1, 1]) if f.rot == 180 else f.T


def node_pts(f, i):
    return f.P[f.Pn == i]


def match(fa, fb):
    """Each A part (depth >= 2) -> nearest B node of similar size, centroid within 3 cm."""
    na, _, _ = main_normal(fa)
    out = []
    for i, nd in fa.nodes.items():
        if i == 0 or nd["n_points"] < 30:
            continue
        Pa = node_pts(fa, i)
        ca, ea = Pa.mean(0), np.ptp(Pa, 0).max()
        best = None
        for j, ndb in fb.nodes.items():
            if j == 0 or ndb["n_points"] < 30:
                continue
            Pb = node_pts(fb, j)
            eb = np.ptp(Pb, 0).max()
            if not (1 / 2.5 < eb / max(ea, 1e-4) < 2.5):
                continue
            d = np.linalg.norm(Pb.mean(0) - ca)
            if d < MAX_GAP_M and (best is None or d < best[0]):
                best = (d, j, Pb.mean(0))
        if best:
            gap = best[2] - ca
            out.append({"a": i, "a_name": nd["name"], "a_depth": nd["depth"],
                        "b": best[1], "b_name": fb.nodes[best[1]]["name"],
                        "gap_mm": round(float(best[0]) * 1000, 1),
                        "gap_along_normal_mm": round(float(gap @ na) * 1000, 1),
                        "gap_in_plane_mm": round(float(np.linalg.norm(gap - (gap @ na) * na)) * 1000, 1),
                        "size_a_mm": round(float(ea) * 1000, 1)})
    return out


def pair_figure(fa, fb, max_parts=4):
    Pa, Ca = painted(fa, COL_A)
    Pb, Cb = painted(fb, COL_B)
    center = np.concatenate([Pa, Pb]).mean(0)
    near = np.linalg.norm(fa.cloud - center, axis=1) < 0.25
    cloud = fa.cloud[near]
    ccol = ((fa.cloud_cols[near] if fa.cloud_cols is not None else np.full((near.sum(), 3), 150)) * 0.4
            ).astype(np.uint8)
    # interleave A and B so neither paints over the other systematically
    P = np.concatenate([cloud, Pa, Pb])
    Cc = np.concatenate([ccol, Ca, Cb])
    both = np.concatenate([Pa, Pb])
    mid_eye = (fa.T[:3, 3] + fb.T[:3, 3]) / 2
    views = [(f"{fa.id} camera", upright(fa)), (f"{fb.id} camera", upright(fb)),
             ("side, 65° around the vertical", look_at(rot_about(center, mid_eye, 65), center)),
             ("top", look_at(center + np.array([0, 0, np.linalg.norm(mid_eye - center)]), center, up=(1, 0, 0)))]
    row1 = [C.label(render(P, Cc, T, both), t, size=14) for t, T in views]
    rows = [C.grid(row1, cols=4, pad=6, bg=(255, 255, 255))]

    matches = match(fa, fb)
    parts = sorted([m for m in matches if m["a_depth"] >= 2], key=lambda m: m["gap_mm"])
    na, _, _ = main_normal(fa)
    for m in parts[:max_parts]:
        A, B = node_pts(fa, m["a"]), node_pts(fb, m["b"])
        c = np.concatenate([A, B]).mean(0)
        size = max(np.ptp(np.concatenate([A, B]), 0).max(), 0.01)
        sel_a = np.linalg.norm(fa.P - c, axis=1) < 2.0 * size
        sel_b = np.linalg.norm(fb.P - c, axis=1) < 2.0 * size
        ctx = np.concatenate([fa.P[sel_a], fb.P[sel_b]])
        ctx_c = (np.concatenate([fa.Pc[sel_a], fb.Pc[sel_b]]) * 0.45).astype(np.uint8)
        ka = np.tile(COL_A.astype(np.uint8), (len(A), 1))
        kb = np.tile(COL_B.astype(np.uint8), (len(B), 1))
        pts = np.concatenate([ctx, A, B])
        cols = np.concatenate([ctx_c, ka, kb])
        eye_front = c + na * 0.12
        up = np.array([0, 0, 1.0]) if abs(na[2]) < 0.9 else np.array([1.0, 0, 0])
        front = look_at(eye_front, c, up=up)
        graze = look_at(rot_about(c, eye_front, 80, axis=up), c, up=up)
        focus = np.concatenate([A, B])
        p1 = C.label(render(pts, cols, front, focus, fill=0.5, r=4), "facing the part", size=13)
        p2 = C.label(render(pts, cols, graze, focus, fill=0.5, r=4), "grazing (80°)", size=13)
        info = np.full((330, 420 * 2 + 6, 3), 255, np.uint8)
        info = C.label(info, f"{fa.id}: {m['a_name']}", pos=(14, 40), size=18, fill=(200, 100, 0), bg="white")
        info = C.label(info, f"{fb.id}: {m['b_name']}", pos=(14, 80), size=18, fill=(0, 140, 200), bg="white")
        info = C.label(info, f"gap between the two copies: {m['gap_mm']:.1f} mm", pos=(14, 150), size=18,
                       fill="black", bg="white")
        info = C.label(info, f"  along the face normal {m['gap_along_normal_mm']:+.1f} mm,"
                             f" in the face {m['gap_in_plane_mm']:.1f} mm", pos=(14, 185), size=16,
                       fill="black", bg="white")
        info = C.label(info, f"  part size {m['size_a_mm']:.0f} mm", pos=(14, 215), size=16, fill="black", bg="white")
        rows.append(np.pad(np.concatenate([p1, np.full((330, 6, 3), 255, np.uint8), p2,
                                           np.full((330, 6, 3), 255, np.uint8), info], 1),
                           ((6, 0), (6, 0), (0, 0)), constant_values=255))
    w = max(r.shape[1] for r in rows)
    img = np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows], 0)
    title = (f"{fa.id} (orange) and {fb.id} (cyan), same run {fa.frame['run']}: both trees in one base frame, "
             f"no alignment")
    return C.title_band(img, title), matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["c5.6:c5.2", "b1.1:b1.4"])
    args = ap.parse_args()
    from PIL import Image
    path = HERE / "evidence" / "pairs_3d.json"
    res = json.load(open(path)) if path.exists() else {}
    for pr in args.pairs:
        a, b = pr.split(":")
        fa, fb = F(C.frames([a])[0]), F(C.frames([b])[0])
        assert fa.frame["run"] == fb.frame["run"], "pairs must come from one run (one base frame)"
        img, matches = pair_figure(fa, fb)
        Image.fromarray(img).save(EVID / f"pair_{a}_{b}.jpg", quality=90)
        V = np.array([node_pts(fb, m["b"]).mean(0) - node_pts(fa, m["a"]).mean(0) for m in matches
                      if m["a_depth"] >= 2 and m["gap_mm"] < 10 and m["a_name"] and "antenna" not in m["a_name"]])
        shift = None
        if len(V) >= 3:
            mu = V.mean(0)
            r = np.linalg.norm(V - mu, axis=1)
            shift = {"n_parts": int(len(V)), "mean_gap_vector_mm": [round(float(x) * 1000, 1) for x in mu],
                     "mean_gap_mm": round(float(np.linalg.norm(mu)) * 1000, 1),
                     "residual_after_shift_median_mm": round(float(np.median(r)) * 1000, 1),
                     "residual_after_shift_max_mm": round(float(r.max()) * 1000, 1),
                     "consistency": round(float(np.linalg.norm(mu) / np.linalg.norm(V, axis=1).mean()), 2),
                     "note": "parts of depth >= 2 with gap < 10 mm; antenna matches excluded (no antenna in B's tree)"}
        res[pr] = {"matches": matches, "rigid_shift": shift}
        print(pr, f"{len(matches)} matches:")
        for m in matches:
            print(f"  {m['a_name'][:26]:26s} ↔ {m['b_name'][:22]:22s} gap {m['gap_mm']:5.1f} mm "
                  f"(normal {m['gap_along_normal_mm']:+5.1f}, in-face {m['gap_in_plane_mm']:4.1f})  size {m['size_a_mm']:.0f}")
    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
    C.run_record(HERE.parents[3] / "inspection" / "data" / "iwm" / "e03_tree_3d" / "pairs",
                 [{"pair": p} for p in args.pairs], {"name": "both trees projected, nearest-centroid match"},
                 {"max_gap_m": MAX_GAP_M})


if __name__ == "__main__":
    main()
