"""qa_sheet.py — before/after contact sheet for the dressing QA gate.

Two rows (before on top, after below) x one column per pose, with a label band
on top. Pure python (PIL) — runs in the `hum` env.

    python -m humanoid.logic.simulation.mapping.dressing.qa_sheet \
        --before /tmp/dressing_qa/before_t71.png ... \
        --after /tmp/dressing_qa/after_t71.png ... \
        --labels "t=71.0s" ... --out data/maps/wc_v1_lever10min/dressing_qa_before_after.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

_BAND = 28  # label band height, px


def compose_sheet(before_paths, after_paths, labels, out_path,
                  scale: float = 1.0) -> None:
    """Compose the QA contact sheet. Row 1 = before, row 2 = after."""
    if not (len(before_paths) == len(after_paths) == len(labels)) or not before_paths:
        raise ValueError("before/after/labels must be equal-length and non-empty")
    befores = [Image.open(p) for p in before_paths]
    afters = [Image.open(p) for p in after_paths]
    w = int(befores[0].width * scale)
    h = int(befores[0].height * scale)
    sheet = Image.new("RGB", (w * len(befores), _BAND + 2 * h), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, (b, a, lab) in enumerate(zip(befores, afters, labels)):
        d.text((i * w + 8, 7), f"{lab}   (top: before / bottom: after)",
               fill=(255, 255, 0))
        sheet.paste(b.resize((w, h)), (i * w, _BAND))
        sheet.paste(a.resize((w, h)), (i * w, _BAND + h))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose the dressing QA contact sheet.")
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    compose_sheet(args.before, args.after, args.labels, args.out, scale=args.scale)
    print(f"[qa_sheet] -> {args.out}")


if __name__ == "__main__":
    main()
