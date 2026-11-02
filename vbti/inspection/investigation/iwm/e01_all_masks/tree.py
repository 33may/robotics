"""e01 part 2: a tree from a set of masks, by containment.

Rule (turn 1):
  1. root = the object mask (recorded, else the largest mask of the set)
  2. keep masks that lie inside the object (>= 50 % of their area), area >= min_area
  3. dedupe: IoU >= 0.85 -> keep the higher score
  4. A contains B  <=>  |A & B| / |B| >= 0.90 and area(A) > area(B)
     parent(B) = the smallest A containing B; root if none
  5. conflicts = pairs that overlap (>= 30 % of the smaller) without containment

Outputs  inspection/data/iwm/e01_all_masks/trees/<frame>/<name>/tree.json, tree.txt
         evidence/<frame>_<name>_levels.png   one panel per depth, masks at that depth
         evidence/<frame>_<name>_nested.png   outlines coloured by depth
         evidence/<frame>_<name>_gallery.png  every node as a crop, by depth

Run:  p inspection/investigation/iwm/e01_all_masks/tree.py --frame c5.6 --sets sam3_obj3_32 sam3_text
      (several --sets are merged; the tree is named a+b)
"""
import argparse
import json

import numpy as np
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C

DEPTH_COLORS = [(230, 30, 30), (255, 140, 0), (240, 210, 0), (40, 180, 60),
                (30, 120, 230), (150, 60, 220), (220, 60, 180), (90, 90, 90)]


def build(masks, recs, obj, contain=0.9, dedupe_iou=0.8, inside=0.5, min_area=40,
          conflict=0.3, ratio=1.25):
    H, W = obj.shape
    # 2. inside the object, big enough
    keep = []
    for i, m in enumerate(masks):
        a = m.sum()
        if a < min_area:
            continue
        if (m & obj).sum() / a < inside:
            continue
        keep.append(i)
    masks = [masks[i] & obj for i in keep]           # clip to the object
    recs = [dict(recs[i]) for i in keep]
    areas = np.array([m.sum() for m in masks])
    scores = np.array([r.get("score", 0.0) for r in recs])
    # 3. dedupe
    order = np.argsort(-scores)
    small = [m[::4, ::4].ravel() for m in masks]
    kept = []
    for i in order:
        dup = False
        for j in kept:
            inter = (small[i] & small[j]).sum()
            union = (small[i] | small[j]).sum()
            if union and inter / union >= dedupe_iou:
                dup = True
                recs[j].setdefault("merged", []).append(recs[i].get("source", "?"))
                break
        if not dup:
            kept.append(int(i))
    masks = [masks[i] for i in kept]
    recs = [recs[i] for i in kept]
    areas = areas[kept]
    # root node 0 = the object
    masks = [obj] + masks
    recs = [{"area": int(obj.sum()), "bbox": C.mask_record(obj)["bbox"], "source": "object",
             "score": 1.0}] + recs
    areas = np.concatenate([[obj.sum()], areas])
    n = len(masks)
    # 4. containment on 4x downsampled masks (fast enough, same answer within 1-2 %)
    S = np.stack([m[::2, ::2].ravel() for m in masks]).astype(np.float32)
    inter = S @ S.T                                   # |A & B|
    a = S.sum(1)
    frac = inter / np.maximum(a[None, :], 1)          # frac[A, B] = |A&B| / |B|
    parent = [-1] * n
    for b in range(1, n):
        # a parent must be clearly larger (ratio), or near-identical masks that
        # slipped past dedupe chain into a meaningless 10-deep ladder
        cands = [A for A in range(n) if A != b and a[A] >= ratio * a[b] and frac[A, b] >= contain]
        parent[b] = min(cands, key=lambda A: a[A]) if cands else 0
    depth = [0] * n
    for b in range(1, n):
        d, p = 1, parent[b]
        while p != 0:
            d, p = d + 1, parent[p]
        depth[b] = d
    # 5. conflicts
    conflicts = []
    for i in range(1, n):
        for j in range(i + 1, n):
            ov = inter[i, j] / max(min(a[i], a[j]), 1)
            if ov >= conflict and frac[i, j] < contain and frac[j, i] < contain:
                conflicts.append((i, j, round(float(ov), 2)))
    nodes = []
    for i in range(n):
        nodes.append({"id": i, "parent": parent[i], "depth": depth[i],
                      "area": int(areas[i]), **{k: v for k, v in recs[i].items() if k != "area"}})
    return masks, nodes, conflicts


def as_text(nodes):
    kids = {}
    for nd in nodes:
        kids.setdefault(nd["parent"], []).append(nd["id"])
    lines = []

    def walk(i, ind):
        nd = nodes[i]
        tag = nd.get("phrase") or nd.get("source", "")
        lvl = f" L{nd['level']}" if "level" in nd else ""
        g = f" g{nd['g']}" if "g" in nd else ""
        lines.append(f"{'  ' * ind}#{i:<3d} {tag}{lvl}{g}  {nd['area']} px  box {nd['bbox']}")
        for k in sorted(kids.get(i, []), key=lambda k: -nodes[k]["area"]):
            walk(k, ind + 1)
    walk(0, 0)
    return "\n".join(lines)


def crop_region(obj, margin=0.06):
    ys, xs = np.nonzero(obj)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return C.expand_box(box, obj.shape, margin)


def fig_levels(rgb, masks, nodes, region, title):
    x0, y0, x1, y1 = region
    maxd = max(nd["depth"] for nd in nodes)
    panels = []
    for d in range(1, maxd + 1):
        ids = [nd["id"] for nd in nodes if nd["depth"] == d]
        ms = [masks[i][y0:y1, x0:x1] for i in ids]
        ov = C.overlay(rgb[y0:y1, x0:x1], ms, alpha=0.5, seed=d) if ms else rgb[y0:y1, x0:x1].copy()
        panels.append(C.label(ov, f"depth {d}: {len(ids)} nodes", size=22))
    cols = 1 if len(panels) <= 3 else 2
    img = C.grid(panels, cols)
    return C.title_band(img, f"{title}   masks at each depth of the tree")


def fig_nested(rgb, masks, nodes, region, title):
    x0, y0, x1, y1 = region
    ms = [masks[nd["id"]][y0:y1, x0:x1] for nd in nodes if nd["id"] > 0]
    cols = [DEPTH_COLORS[min(nd["depth"] - 1, len(DEPTH_COLORS) - 1)] for nd in nodes if nd["id"] > 0]
    img = C.outlines(rgb[y0:y1, x0:x1], ms, cols, thick=2, dim=0.25)
    maxd = max(nd["depth"] for nd in nodes)
    legend = "  ".join(f"depth {d}" for d in range(1, maxd + 1))
    img = C.label(img, f"{title}   outlines by depth: {legend}", size=18)
    # colour swatches
    from PIL import ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for d in range(1, maxd + 1):
        dr.rectangle([8 + (d - 1) * 26, 34, 28 + (d - 1) * 26, 54],
                     fill=DEPTH_COLORS[min(d - 1, len(DEPTH_COLORS) - 1)])
    return np.array(im)


def fig_gallery(rgb, masks, nodes, max_nodes=72, thumb=150):
    """Each node: its bbox crop with the mask tinted, label = id/depth/parent."""
    order = sorted(nodes, key=lambda nd: (nd["depth"], -nd["area"]))[:max_nodes]
    panels = []
    for nd in order:
        i = nd["id"]
        x0, y0, x1, y1 = C.expand_box(nd["bbox"], rgb.shape[:2], 0.15)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        sub = rgb[y0:y1, x0:x1]
        m = masks[i][y0:y1, x0:x1]
        col = DEPTH_COLORS[min(nd["depth"] - 1, len(DEPTH_COLORS) - 1)] if i else (90, 90, 90)
        ov = C.overlay(sub, [m], alpha=0.45, colors=[col], thick=2)
        # square thumbnail
        h, w = ov.shape[:2]
        s = max(h, w)
        sq = np.full((s, s, 3), 255, np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = ov
        sq = C.resize(sq, thumb)
        tag = nd.get("phrase") or nd.get("source", "")[:10]
        sq = C.label(sq, f"#{i} d{nd['depth']} p{nd['parent']} {tag}", size=13)
        panels.append(sq)
    return C.grid(panels, cols=8, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--name")
    ap.add_argument("--contain", type=float, default=0.9)
    args = ap.parse_args()
    f = C.frames([args.frame])[0]
    rgb = C.load_rgb(f)
    name = args.name or "+".join(args.sets)

    masks, recs = [], []
    for s in args.sets:
        d = C.DATA / "masks" / f["id"] / s
        arr, meta = C.load_masks(d)
        g = meta["settings"].get("granularity")
        for m, r in zip(arr, meta["masks"]):
            masks.append(m)
            rr = {**r, "source": s}
            if g is not None:
                rr["g"] = g
            recs.append(rr)
    obj, how = C.recorded_object(f)
    if obj is None:
        # no recorded mask for this step: the object = the largest mask of the
        # coarsest UnSAMv2 run if we have it, else of the sets themselves
        pool, src = masks, recs
        coarse = C.DATA / "masks" / f["id"] / "unsam_plus_g0.9"
        if coarse.exists():
            arr, meta = C.load_masks(coarse)
            pool = list(arr)
            src = [{"source": "unsam_plus_g0.9"} for _ in pool]
        areas = [m.sum() for m in pool]
        big = max(range(len(pool)), key=lambda i: areas[i] if areas[i] < 0.8 * obj_lim(rgb) else -1)
        obj, how = pool[big], f"largest mask of {src[big]['source']}"
    masks, nodes, conflicts = build(masks, recs, obj, contain=args.contain)

    out = C.DATA / "trees" / f["id"] / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"frame": f, "sets": args.sets, "object_from": how, "contain": args.contain,
               "nodes": nodes, "conflicts": conflicts},
              open(out / "tree.json", "w"), indent=1)
    txt = as_text(nodes)
    (out / "tree.txt").write_text(txt + "\n")
    C.save_packed(out / "node_masks.npz", masks)      # node i <-> masks[i]
    region = crop_region(obj)
    title = f"{f['id']}  {name}"
    Image.fromarray(fig_levels(rgb, masks, nodes, region, title)).save(
        C.EVIDENCE / f"{f['id']}_{name}_levels.png")
    Image.fromarray(fig_nested(rgb, masks, nodes, region, title)).save(
        C.EVIDENCE / f"{f['id']}_{name}_nested.png")
    Image.fromarray(fig_gallery(rgb, masks, nodes)).save(
        C.EVIDENCE / f"{f['id']}_{name}_gallery.png")
    (C.EVIDENCE / f"{f['id']}_{name}_tree.txt").write_text(txt + "\n")
    maxd = max(nd["depth"] for nd in nodes)
    per = [sum(nd["depth"] == d for nd in nodes) for d in range(1, maxd + 1)]
    print(f"{f['id']} {name}: {len(nodes) - 1} nodes, depth {maxd}, per depth {per}, "
          f"{len(conflicts)} conflicts, object from {how}")
    C.run_record(out, {"id": f["id"], "run": f["run"], "step": f["step"]},
                 {"name": "containment tree"}, {"sets": args.sets, "contain": args.contain})


def obj_lim(rgb):
    return rgb.shape[0] * rgb.shape[1]


if __name__ == "__main__":
    main()
