#!/usr/bin/env python
"""Capture a screenshot of the app under test.

    python observe.py 01_initial            -> $UITEST_DIR/01_initial.png
    python observe.py /tmp/x/foo.png
    python observe.py 02_after --settle 0.5 --region 0,860,1600,140

A bare name (no '/' and no '.png') is resolved inside $UITEST_DIR, which keeps
derived artifacts out of the repo by construction.
--region additionally writes <name>_crop.png for close reading.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cdp as C  # noqa: E402


def resolve(name: str) -> Path:
    p = Path(name)
    if p.suffix != ".png":
        p = p.with_suffix(".png")
    return p if p.is_absolute() or len(p.parts) > 1 else C.ART / p.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="output name or path")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="sleep this long before capturing (let animations finish)")
    ap.add_argument("--region", help="also write a cropped x,y,w,h copy")
    ap.add_argument("--park", metavar="X,Y",
                    help="move the cursor here first, so hover state does not "
                         "leak into the screenshot (Rerun @1600x1000: 100,700)")
    args = ap.parse_args()

    out = resolve(args.name)

    sess = C.Session.load()
    with C.CDP(sess.ws_url) as c:
        if args.park:
            px, py = (float(v) for v in args.park.replace(" ", "").split(","))
            c.move(px, py)
            time.sleep(0.35)
        if args.settle:
            time.sleep(args.settle)
        c.screenshot(out)

    res = {"path": str(out)}
    if args.region:
        from PIL import Image
        x, y, w, h = (int(v) for v in args.region.replace(" ", "").split(","))
        crop = out.with_name(out.stem + "_crop.png")
        Image.open(out).crop((x, y, x + w, y + h)).save(crop)
        res["crop"] = str(crop)
    from PIL import Image as _I
    im = _I.open(out)
    res["size"] = [im.width, im.height]
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
