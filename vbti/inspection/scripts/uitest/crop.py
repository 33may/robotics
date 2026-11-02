#!/usr/bin/env python
"""Cut a region out of a screenshot so it can be read closely.

    python crop.py shot.png --region 0,860,1600,140 --out /tmp/rrd_ui_test/timeline.png
    python crop.py shot.png --region 0,60,250,800 --scale 2 --grid 50

--scale upsamples with NEAREST (no blur) so small text stays legible when the
crop is read back as an image.
--grid overlays a labelled coordinate grid in ORIGINAL image coordinates, which
is how you find pixel targets on a WASM canvas that has no DOM to query.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--region", help="x,y,w,h (default: whole image)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=0,
                    help="draw a grid every N original px, labelled with absolute coords")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: missing {args.image}", file=sys.stderr)
        return 2

    im = Image.open(args.image).convert("RGB")
    ox = oy = 0
    if args.region:
        x, y, w, h = (int(v) for v in args.region.replace(" ", "").split(","))
        x, y = max(0, x), max(0, y)
        w, h = min(w, im.width - x), min(h, im.height - y)
        if w <= 0 or h <= 0:
            print(f"ERROR: region outside image {im.width}x{im.height}", file=sys.stderr)
            return 2
        im = im.crop((x, y, x + w, y + h))
        ox, oy = x, y

    if args.scale != 1.0:
        im = im.resize((int(im.width * args.scale), int(im.height * args.scale)),
                       Image.NEAREST)

    if args.grid > 0:
        d = ImageDraw.Draw(im)
        s, g = args.scale, args.grid
        gx = ox - (ox % g) + g
        while (gx - ox) * s < im.width:
            px = (gx - ox) * s
            d.line([(px, 0), (px, im.height)], fill=(0, 255, 0), width=1)
            d.text((px + 2, 2), str(gx), fill=(0, 255, 0))
            gx += g
        gy = oy - (oy % g) + g
        while (gy - oy) * s < im.height:
            py = (gy - oy) * s
            d.line([(0, py), (im.width, py)], fill=(0, 255, 255), width=1)
            d.text((2, py + 2), str(gy), fill=(0, 255, 255))
            gy += g

    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"{args.out}  {im.width}x{im.height}  origin=({ox},{oy}) scale={args.scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
