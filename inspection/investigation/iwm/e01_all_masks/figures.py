"""e01 comparison figures (read mask sets, write evidence/).

  ladder   per frame, one row: UnSAMv2 granularity 0.1 -> 0.9 on the object crop
  strategies  per frame, one row: SAM 3 grid32 | SAM 3 obj3 | SAM 3 text | UnSAMv2 g0.3 | UnSAMv2 g0.1 eval
  summary  counts table -> evidence/mask_counts.json

Run:  p inspection/investigation/iwm/e01_all_masks/figures.py [--frames c5.6 ...]
"""
import argparse
import json

import numpy as np
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C

LADDER = ["unsam_plus_g0.1", "unsam_plus_g0.2", "unsam_plus_g0.3", "unsam_plus_g0.5",
          "unsam_plus_g0.7", "unsam_plus_g0.9"]
STRATEGIES = ["sam3_grid32", "sam3_obj3_32", "sam3_text", "unsam_plus_g0.3", "unsam_plus_g0.1_eval"]
PANEL_W = 520


def object_region(f, masks_fallback=None):
    obj, _ = C.recorded_object(f)
    if obj is None and masks_fallback is not None:
        areas = [m.sum() for m in masks_fallback]
        obj = masks_fallback[int(np.argmax(areas))]
    if obj is None:
        return (0, 0, 1280, 720)
    ys, xs = np.nonzero(obj)
    return C.expand_box((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), obj.shape, 0.06)


def panel(rgb, set_dir, region, title, inside_only=True):
    x0, y0, x1, y1 = region
    sub = rgb[y0:y1, x0:x1]
    if not (set_dir / "masks.json").exists():
        return C.label(C.resize(sub, PANEL_W), f"{title}: missing", size=16), None
    arr, meta = C.load_masks(set_dir)
    ms = [m[y0:y1, x0:x1] for m in arr]
    if inside_only:
        ms = [m for m in ms if m.sum() > 0]
    ov = C.overlay(sub, ms, alpha=0.5) if ms else sub.copy()
    ov = C.resize(ov, PANEL_W)
    return C.label(ov, f"{title}: {len(ms)} masks in crop ({meta['n_masks']} total)", size=16), len(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    args = ap.parse_args()
    counts = {}
    for f in C.frames(args.frames):
        rgb = C.load_rgb(f)
        base = C.DATA / "masks" / f["id"]
        fallback = None
        if (base / "sam3_obj3_32" / "masks.json").exists():
            fallback, _ = C.load_masks(base / "sam3_obj3_32")
            fallback = list(fallback)
        region = object_region(f, fallback)
        counts[f["id"]] = {}
        # ladder
        panels = []
        for s in LADDER:
            p, n = panel(rgb, base / s, region, s.replace("unsam_plus_g", "UnSAMv2 g="))
            panels.append(p)
            counts[f["id"]][s] = n
        img = C.grid(panels, cols=3)
        img = C.title_band(img, f"{f['id']}  UnSAMv2 (plus) granularity ladder, object crop")
        Image.fromarray(img).save(C.EVIDENCE / f"{f['id']}_ladder.png")
        # strategies
        panels = []
        for s in STRATEGIES:
            p, n = panel(rgb, base / s, region, s)
            panels.append(p)
            counts[f["id"]][s] = n
        img = C.grid(panels, cols=3)
        img = C.title_band(img, f"{f['id']}  strategies, object crop")
        Image.fromarray(img).save(C.EVIDENCE / f"{f['id']}_strategies.png")
        print(f["id"], counts[f["id"]])
    old = {}
    if (C.EVIDENCE / "mask_counts.json").exists():
        old = json.load(open(C.EVIDENCE / "mask_counts.json"))
    old.update(counts)
    json.dump(old, open(C.EVIDENCE / "mask_counts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
