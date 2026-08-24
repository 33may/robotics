#!/usr/bin/env python3
"""What would the object look like if the shell were tighter?

Answers "do I like how it looks" without re-running the robot. Every recorded
capture was taken at a known distance to the object centroid; a camera at a
SMALLER distance sees a smaller metric window with the same pixel count, so
cropping the recorded frame to that window and upscaling reproduces the
FRAMING exactly.

Honest about what it is: framing is exact, sharpness is pessimistic — a real
capture at the shorter distance resolves more detail than this upscaled crop,
never less. So if a candidate radius looks good here, the real one looks better.

    p inspection/investigation/frame_preview.py --run 2408-seeded
    p inspection/investigation/frame_preview.py --run 2408-seeded --fill 0.60
    p inspection/investigation/frame_preview.py --run 2408-seeded --rule upright
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from inspection.view.grid import (object_extent, radius_for_extent,
                                  FILL_TARGET, EXTENT_RULE)

RUNS = Path(__file__).resolve().parents[1] / "data/runs"
OUT = Path(__file__).resolve().parents[1] / "data/framing"
TILE = 320


def captures(run_dir):
    """Recorded cell captures: (dir, T_base_cam, rgb_rotation_deg).

    The rotation matters here because this module PROJECTS into the colour
    frame: since 2026-08-24 `rgb.png` is stored upright while `T_base_cam`
    still describes the raw camera, so a half-turn capture must be rotated
    back before its pixels line up with the projection. Older runs have no
    key and are raw on disk — hence 0, not a guess.
    """
    out = []
    for d in sorted(run_dir.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists() and (d / "rgb.png").exists():
            m = json.loads(meta.read_text())
            out.append((d, np.array(m["T_base_cam"]),
                        int(m.get("rgb_rotation_deg", 0) or 0)))
    return out


def reframe(rgb, T_base_cam, center, intr, r_new):
    """Crop the frame to the metric window a camera at r_new would see, kept
    centred on the object. Returns None if r_new is FARTHER than the capture
    (that would need pixels outside the recorded frame)."""
    # Recorded T_base_cam is the OPENCV convention (+Z boresight) — it is the
    # pose `deproject` unprojects depth with. NOT the viewsphere's +X-forward
    # convention; `load_T_flange_cam` exists precisely because both are live.
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p = (center - t) @ R                       # object centre, camera frame
    r_now = float(p[2])                        # depth along the boresight
    if r_new > r_now:
        return None, r_now
    H, W = rgb.shape[:2]
    cu = intr["ppx"] + intr["fx"] * (p[0] / p[2])
    cv_ = intr["ppy"] + intr["fy"] * (p[1] / p[2])
    half_h = 0.5 * H * (r_new / r_now)
    half_w = 0.5 * W * (r_new / r_now)
    u0, u1 = int(round(cu - half_w)), int(round(cu + half_w))
    v0, v1 = int(round(cv_ - half_h)), int(round(cv_ + half_h))
    # clamp inside the frame; a centre near the edge just shifts the window
    u0, u1 = max(0, u0), min(W, u1)
    v0, v1 = max(0, v0), min(H, v1)
    if u1 - u0 < 8 or v1 - v0 < 8:
        return None, r_now
    return rgb[v0:v1, u0:u1], r_now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2408-seeded")
    ap.add_argument("--fill", type=float, default=FILL_TARGET)
    ap.add_argument("--rule", default=EXTENT_RULE,
                    choices=["sphere", "upright", "footprint"])
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    cloud = np.load(run_dir / "fused_cloud.npy")
    intr = json.loads((run_dir / "session.json").read_text())["intrinsics"]["color"]
    center = cloud.mean(axis=0)
    mn, mx = cloud.min(0), cloud.max(0)

    extent = object_extent(mn, mx, args.rule)
    r = radius_for_extent(extent, intr["fy"], intr["height"], args.fill)
    print(f"{args.run}: AABB {((mx - mn) * 1000).round(1)} mm")
    print(f"rule={args.rule} extent={extent * 1000:.1f} mm  fill={args.fill:.2f}"
          f"  ->  r = {r:.3f} m   ({r * 1000 / intr['fy']:.2f} mm/px on target)")
    for other in ("sphere", "upright", "footprint"):
        e = object_extent(mn, mx, other)
        print(f"    {other:<10} extent {e * 1000:6.1f} mm -> r "
              f"{radius_for_extent(e, intr['fy'], intr['height'], args.fill):.3f} m")

    tiles = []
    for d, T, rot in captures(run_dir):
        img = cv2.imread(str(d / "rgb.png"))
        if rot == 180:                      # back into the raw camera frame
            img = np.ascontiguousarray(img[::-1, ::-1])
        crop, r_now = reframe(img, T, center, intr, r)
        if crop is None:
            continue
        h = int(TILE * crop.shape[0] / crop.shape[1])
        tile = cv2.resize(crop, (TILE, h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(tile, f"{d.name}  {r_now:.2f}->{r:.2f}m", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(tile)

    if not tiles:
        print("no capture was farther than the requested radius — nothing to show")
        return 1
    hgt = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hgt - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=0) for t in tiles]
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + args.cols])
                       for i in range(0, len(tiles), args.cols)])

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.run}_{args.rule}_fill{args.fill:.2f}_r{r:.3f}.png"
    cv2.imwrite(str(path), sheet)
    print(f"\n{len(tiles)} tiles -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
