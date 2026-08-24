#!/usr/bin/env python3
"""Replay a recorded run through the SHIPPED identity pipeline, offline.

This drives `run.segmenter.object_view` — the exact function the settle leg
calls — so the numbers it prints are a measurement of the loop, not of a
probe that resembles it. Everything a view needs is on disk, so a run can be
re-scored after any change without a robot.

The question it was built to answer (run 2408-cup2, view 003, cell (2,0) on
the 10 deg ring): a cable passed behind the cup within 20 mm, so seeded
growth and the 8 cm jump gate both had to accept it and the object extent
went to 175.8 mm. Masked, the same frames give 89.1 mm.

Frames. `rgb.png` and `depth_aligned.npy` were rotated at capture time to be
stored upright; `depth_raw` and `T_base_cam` are raw. The loop works in RAW
orientation throughout, so this un-rotates the stored pair to reconstruct the
`cap` dict a rig would have returned.

    p inspection/investigation/mask_probe.py --run 2408-cup2
    p inspection/investigation/mask_probe.py --run 2408-cup2 --save /tmp/masks
    p inspection/investigation/mask_probe.py --run 2408-cup2 --no-segment
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inspection.cell.geometry import BOX_PCT, CloudAccumulator, rotate180
from inspection.run.segmenter import ObjectSegmenter, object_view

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"


def captures(run_dir: Path):
    """Rebuild each capture as the rig contract dict, in RAW orientation."""
    import cv2
    for d in sorted(p for p in run_dir.iterdir()
                    if p.is_dir() and p.name.isdigit()):
        meta = json.loads((d / "meta.json").read_text())
        rot = int(meta.get("rgb_rotation_deg", 0))
        rgb = cv2.cvtColor(cv2.imread(str(d / "rgb.png")), cv2.COLOR_BGR2RGB)
        depth_aligned = np.load(d / "depth_aligned.npy")
        if rot == 180:
            rgb, depth_aligned = rotate180(rgb), rotate180(depth_aligned)
        yield meta, {"dir": str(d), "rgb": rgb,
                     "depth_raw": np.load(d / "depth_raw.npy"),
                     "depth_aligned": depth_aligned,
                     "T_base_cam": np.array(meta["T_base_cam"], dtype=float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--save", default=None, help="dir for mask overlays")
    ap.add_argument("--no-segment", action="store_true",
                    help="depth-only, for a side-by-side baseline")
    ap.add_argument("--gate", type=float, default=None,
                    help="override JUMP_GATE_M [m]; 0 disables the jump gate")
    args = ap.parse_args()

    if args.gate is not None:
        import inspection.cell.geometry as geom
        geom.JUMP_GATE_M = args.gate

    run_dir = RUNS / args.run
    session = json.loads((run_dir / "session.json").read_text())
    intr = session["intrinsics"]["ir_left"]
    intr_color = session["intrinsics"]["color"]
    scale = session["depth_scale_m_per_unit"]

    segmenter = None if args.no_segment else ObjectSegmenter()
    acc = CloudAccumulator()
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    print(f"{'view':>4} {'source':>10} {'score':>6} {'mask px':>8} {'kept':>6} "
          f"{'drop':>5}  {'raw extent mm':>21}  {'trimmed mm':>21}")
    for meta, cap in captures(run_dir):
        fallbacks = []
        view, seg = object_view(cap, intr, intr_color, scale, acc.points,
                                segmenter, on_fallback=fallbacks.append)
        pts = view["points"]
        dropped = acc.add(pts) if len(pts) else 0
        mn, mx = acc.aabb()
        tl, th = acc.aabb(pct=BOX_PCT)
        print(f"{meta['pose_id']:>4} {('mask' if seg else 'depth'):>10} "
              f"{(f'{seg.score:.2f}' if seg else '-'):>6} "
              f"{(int(seg.mask.sum()) if seg else 0):>8} {len(pts):>6} "
              f"{dropped:>5}  {str(np.round((mx - mn) * 1000, 1)):>21}  "
              f"{str(np.round((th - tl) * 1000, 1)):>21}")
        for why in fallbacks:
            print(f"       ! {why}")

        if save and seg is not None:
            from PIL import Image
            ov = cap["rgb"].copy()
            ov[seg.mask] = (0.45 * ov[seg.mask]
                            + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
            x0, y0, x1, y1 = seg.box
            ov[y0:y0 + 2, x0:x1] = ov[y1 - 2:y1, x0:x1] = (255, 0, 0)
            ov[y0:y1, x0:x0 + 2] = ov[y0:y1, x1 - 2:x1] = (255, 0, 0)
            rot = int(meta.get("rgb_rotation_deg", 0))
            Image.fromarray(rotate180(ov) if rot == 180 else ov).save(
                save / f"{meta['pose_id']:03d}.png")

    mn, mx = acc.aabb()
    tl, th = acc.aabb(pct=BOX_PCT)
    print(f"\n{len(acc.points)} pts   raw {np.round((mx - mn) * 1000, 1)}"
          f"   trimmed {np.round((th - tl) * 1000, 1)} mm")
    if segmenter is not None:
        print(f"segmenter: {segmenter.calls} calls, {segmenter.misses} misses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
