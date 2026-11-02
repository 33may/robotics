#!/usr/bin/env python3
"""Draw what the verbs saw onto the captures — eyeball verification.

A number in a log ("0.94") does not tell you whether the box is around the
cup or around the table leg. This renders detect boxes, segment masks and
OCR reads onto the frames the models actually received (after `upright`), so
a human can check the verbs are looking at the right thing.

Writes into the run's own folder: <run>/eyes/preview/<cell>.png

Run: p inspection/eyes/preview.py <run_dir> [--cells 2,0 3,1] [--phrase cup]
"""
import argparse
import sys

import numpy as np


def draw(img, dets, seg=None, lines=(), phrase="cup"):
    """Overlay boxes/mask/text on an RGB frame. Returns a new RGB array."""
    import cv2
    out = img.copy()
    if seg is not None:
        tint = np.zeros_like(out)
        tint[seg.mask] = (0, 255, 128)                 # mask in green
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d.box
        colour = (0, 255, 0) if i == 0 else (255, 200, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2 if i == 0 else 1)
        cv2.putText(out, f"{phrase} {d.score:.2f}", (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    for t in lines:
        x0, y0, x1, y1 = t.box
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(out, f"{t.text} {t.score:.2f}", (x0, min(470, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    import cv2

    from inspection.eyes.tools import ViewTools
    from inspection.eyes.verbs_local import (LocalVerbs, PaddleOcrBackend,
                                             Sam3Backend)
    from inspection.record.run import Run

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="h,v pairs; default = first 6 visited")
    ap.add_argument("--phrase", default="cup")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--ocr", action="store_true", help="also run read_text")
    args = ap.parse_args()

    tools = ViewTools(Run.load(args.run_dir))
    detector = LocalVerbs(Sam3Backend())
    reader = LocalVerbs(PaddleOcrBackend()) if args.ocr else None

    visited = sorted({tuple(s.record.view.address)
                      for s in tools._run.captured if s.id != 0})
    cells = ([tuple(int(v) for v in c.split(",")) for c in args.cells]
             if args.cells else visited[:6])
    outdir = tools._run.path / "eyes" / "preview"
    outdir.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        img = tools.get_view(cell)
        dets = detector.detect(img, args.phrase, min_score=args.min_score)
        seg = detector.segment(img, dets[0].box) if dets else None
        lines = reader.read_text(img) if reader else []
        vis = draw(img.rgb, dets, seg, lines, args.phrase)
        path = outdir / f"cell_{cell[0]:02d}_{cell[1]}.png"
        cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        area = f", mask {int(seg.mask.sum())} px" if seg is not None else ""
        print(f"{path.name}: {len(dets)} {args.phrase} "
              f"(best {dets[0].score:.3f}){area}" if dets
              else f"{path.name}: no {args.phrase} above {args.min_score}")
    print(f"\nwrote {len(cells)} overlays to {outdir}")


if __name__ == "__main__":
    sys.exit(main())
