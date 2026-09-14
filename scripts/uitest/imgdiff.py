#!/usr/bin/env python
"""Image diff for UI assertions. Standalone CLI + importable.

    python imgdiff.py A.png B.png
    python imgdiff.py A.png B.png --region 0,860,1600,140 --min-frac 0.02
    python imgdiff.py A.png B.png --region 250,60,980,800 --max-frac 0.005   # assert NO change
    python imgdiff.py A.png B.png --heatmap /tmp/rrd_ui_test/d.png

Exit code 0 = assertion passed, 1 = failed, 2 = usage/IO error.
With neither --min-frac nor --max-frac it just reports (always exit 0).

Metrics
  changed_frac : fraction of pixels whose max-channel abs difference exceeds
                 --tol (default 12/255).  A per-pixel threshold, not a mean,
                 because a real UI change is a *localised* change - a mean over
                 the whole frame washes it out.
  mean_abs     : mean absolute difference over all channels, 0-255.
  max_abs      : largest single-channel difference.
  bbox         : bounding box of changed pixels, in FULL-IMAGE coords - very
                 useful for discovering where a widget actually lives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def parse_region(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(v) for v in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def diff(a_path, b_path, region=None, tol: int = 12, heatmap: Path | None = None) -> dict:
    a, b = _load(Path(a_path)), _load(Path(b_path))
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}",
                "changed_frac": 1.0, "mean_abs": 255.0, "max_abs": 255, "bbox": None}

    ox = oy = 0
    if region:
        x, y, w, h = region
        H, W = a.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 0 or h <= 0:
            raise ValueError(f"region {region} outside image {W}x{H}")
        a, b, ox, oy = a[y:y + h, x:x + w], b[y:y + h, x:x + w], x, y

    d = np.abs(a - b)
    per_px = d.max(axis=2)
    mask = per_px > tol

    bbox = None
    if mask.any():
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()) + ox, int(ys.min()) + oy,
                int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1]

    if heatmap:
        hm = np.zeros((*mask.shape, 3), dtype=np.uint8)
        hm[..., 0] = np.clip(per_px, 0, 255).astype(np.uint8)
        hm[..., 1] = (mask * 255).astype(np.uint8)
        Path(heatmap).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hm).save(heatmap)

    return {
        "changed_frac": float(mask.mean()),
        "mean_abs": float(d.mean()),
        "max_abs": int(d.max()),
        "bbox": bbox,
        "region": list(region) if region else None,
        "size": [int(a.shape[1]), int(a.shape[0])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--region", help="x,y,w,h to restrict the comparison to")
    ap.add_argument("--tol", type=int, default=12, help="per-pixel channel tolerance (0-255)")
    ap.add_argument("--min-frac", type=float, help="FAIL if changed_frac < this (expect a change)")
    ap.add_argument("--max-frac", type=float, help="FAIL if changed_frac > this (expect no change)")
    ap.add_argument("--heatmap", type=Path, help="write a diff visualisation here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    try:
        r = diff(args.a, args.b, parse_region(args.region), args.tol, args.heatmap)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    ok, why = True, []
    if args.min_frac is not None and r["changed_frac"] < args.min_frac:
        ok = False
        why.append(f"changed_frac {r['changed_frac']:.5f} < min {args.min_frac}")
    if args.max_frac is not None and r["changed_frac"] > args.max_frac:
        ok = False
        why.append(f"changed_frac {r['changed_frac']:.5f} > max {args.max_frac}")

    if not args.quiet:
        import json
        print(json.dumps({**r, "pass": ok, "why": why}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
