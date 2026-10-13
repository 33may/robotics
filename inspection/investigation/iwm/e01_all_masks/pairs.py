"""e01 pair figures: the same tree method on two frames of one object, side by side.

  far/close   c5.2 (29.7 cm, oblique) vs c5.6 (21.9 cm, head-on)
  multi-view  c5.7 vs c9.6 (compute x- panel, two runs); b1.4 vs b4.5 (box sticker face, two runs)

Each: nested outlines coloured by depth for both frames, and the per-depth counts.
Output: evidence/pair_<a>_<b>_<tree>.png
Run:    p inspection/investigation/iwm/e01_all_masks/pairs.py
"""
import json

import numpy as np
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e01_all_masks.tree import DEPTH_COLORS, crop_region

PAIRS = [("c5.2", "c5.6", "far / close"), ("c5.7", "c9.6", "two runs, x- panel"),
         ("b1.4", "b4.5", "two runs, sticker face")]
TREE = "unsam_stack+sam3_text"
W = 900


def nested_panel(fid, tree):
    f = C.frames([fid])[0]
    rgb = C.load_rgb(f)
    tdir = C.DATA / "trees" / fid / tree
    t = json.load(open(tdir / "tree.json"))
    nodes = t["nodes"]
    masks = C.load_packed(tdir / "node_masks.npz")   # node i <-> masks[i], 0 = object
    region = crop_region(masks[0])
    x0, y0, x1, y1 = region
    ms = [masks[nd["id"]][y0:y1, x0:x1] for nd in nodes if nd["id"] > 0]
    cols = [DEPTH_COLORS[min(nd["depth"] - 1, len(DEPTH_COLORS) - 1)] for nd in nodes if nd["id"] > 0]
    img = C.outlines(rgb[y0:y1, x0:x1], ms, cols, thick=2, dim=0.25)
    maxd = max(n["depth"] for n in nodes)
    per = [sum(n["depth"] == k for n in nodes) for k in range(1, maxd + 1)]
    img = C.resize(img, W)
    return C.label(img, f"{fid}  {f['dist_cm']} cm  {len(nodes) - 1} nodes, depth {maxd}, per depth {per}", size=16)


def main():
    for a, b, why in PAIRS:
        try:
            pa, pb = nested_panel(a, TREE), nested_panel(b, TREE)
        except FileNotFoundError as e:
            print("skip", a, b, e)
            continue
        h = max(pa.shape[0], pb.shape[0])
        pad = lambda p: np.concatenate([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)], 0)
        img = C.grid([pad(pa), pad(pb)], cols=2)
        img = C.title_band(img, f"{a} vs {b}: {why}   tree = {TREE}, node boxes coloured by depth")
        Image.fromarray(img).save(C.EVIDENCE / f"pair_{a}_{b}.png")
        print(f"evidence/pair_{a}_{b}.png")


if __name__ == "__main__":
    main()
