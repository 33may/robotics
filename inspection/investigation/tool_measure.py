#!/usr/bin/env python3
"""Measure the real RG2 + camera stack off photographs, in the flange frame.

The shipped collision hull is a single 258 x 149 x 170 mm box that unions the
gripper's WORST state over every axis at once (closed length, max-open width,
full height). `block_report.py` shows that hull costs 12 of 36 cells at a
0.226 m shell — so it is worth knowing what the hardware actually occupies.

Photographs are not CAD. This produces numbers with a stated scale reference
and an error bar; a hull built from them must still be inflated, never
trimmed to the measured value. Under-modelling a tool is how you crash one.

    p inspection/investigation/tool_measure.py --grid       # overlay to read off
    p inspection/investigation/tool_measure.py --crop 900 400 1700 1100
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "data/tool"


def grid_overlay(img, step=100, major=500, origin=(0, 0), scale=1.0):
    """Label a pixel grid so features can be read off by eye.

    Labels are ALWAYS in ORIGINAL-image pixels, whatever crop or zoom is in
    play — `origin` is the crop's top-left in the original and `scale` the
    zoom. A grid that silently relabels itself per crop is worse than no
    grid: every number read off it is quietly in a different frame.
    """
    out = img.copy()
    h, w = out.shape[:2]
    ox, oy = origin
    x0 = int(np.ceil(ox / step) * step)
    for xo in range(x0, int(ox + w / scale) + 1, step):
        x = int((xo - ox) * scale)
        hard = xo % major == 0
        cv2.line(out, (x, 0), (x, h), (0, 255, 0) if hard else (0, 150, 0),
                 2 if hard else 1)
        if hard:
            cv2.putText(out, str(xo), (x + 4, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 3)
    y0 = int(np.ceil(oy / step) * step)
    for yo in range(y0, int(oy + h / scale) + 1, step):
        y = int((yo - oy) * scale)
        hard = yo % major == 0
        cv2.line(out, (0, y), (w, y), (0, 255, 0) if hard else (0, 150, 0),
                 2 if hard else 1)
        if hard:
            cv2.putText(out, str(yo), (6, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="both")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--crop", type=int, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--scale", type=float, default=2.0, help="upscale a crop")
    args = ap.parse_args()

    names = ["tggrippertop", "tggripperbot"] if args.name == "both" \
        else [args.name]
    for n in names:
        img = cv2.imread(str(TOOL / f"{n}.jpg"))
        if img is None:
            raise FileNotFoundError(TOOL / f"{n}.jpg")
        tag = n
        if args.crop:
            x0, y0, x1, y1 = args.crop
            img = img[y0:y1, x0:x1]
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_CUBIC)
            tag += f"_crop{x0}-{y0}"
        if args.grid:
            img = grid_overlay(img, step=args.step)
            tag += "_grid"
        path = TOOL / f"{tag}.png"
        cv2.imwrite(str(path), img)
        print(f"{path}  {img.shape[1]}x{img.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
