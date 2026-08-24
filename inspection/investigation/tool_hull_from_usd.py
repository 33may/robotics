#!/usr/bin/env python3
"""Per-PART bounds of the RG2 in the flange frame — the shape of a real hull.

`cell.yaml` models the tool as ONE box that unions the worst value of every
axis independently: closed length AND max-open width AND full bracket height.
The result is a 149 x 170 mm slab carried all the way out to the fingertips,
when in reality the far end is two thin fingers and the bulk sits back near
the mount. Under tilt that phantom slab is what sweeps into the arm.

This prints each CAD part's AABB in the flange frame, then bins the parts
along the tool axis so a two/three-box hull can be cut where the geometry
actually steps rather than where someone guessed.

Does NOT include the camera bracket / blue plate — those are Anton's
additions and not in Antonio's USD. Measure those separately and add them as
their own box; a hull is only safe if it covers what is physically bolted on.

    p inspection/investigation/tool_hull_from_usd.py
    p inspection/investigation/tool_hull_from_usd.py --bins 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspection.cell.usd_preview import (DEFAULT_USD, FLANGE_PRIM,
                                         GRIPPER_PRIMS, collect_meshes)


def part_bounds(stage, slide_mm=-22.0):
    """{part path: (min, max)} in the flange frame, metres."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    T_fw = np.linalg.inv(np.array(
        cache.GetLocalToWorldTransform(stage.GetPrimAtPath(FLANGE_PRIM))).T)
    out = {}
    for path, world, _ in collect_meshes(stage, skip_collisions=True,
                                         slide_mm=slide_mm):
        if not any(path.startswith(r) for r in GRIPPER_PRIMS):
            continue
        homo = np.hstack([world / scale, np.ones((len(world), 1))])
        p = (T_fw @ homo.T).T[:, :3] * scale
        out[path] = (p.min(axis=0), p.max(axis=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, default=DEFAULT_USD)
    ap.add_argument("--slide-mm", type=float, default=-22.0)
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    parts = part_bounds(stage, args.slide_mm)
    if not parts:
        print("no gripper parts found")
        return 1

    print(f"{len(parts)} parts, flange frame, mm\n")
    print(f"{'part':<34}{'X0':>7}{'X1':>7}{'|Y|max':>8}{'Z0':>7}{'Z1':>7}")
    rows = sorted(parts.items(), key=lambda kv: kv[1][0][0])
    for path, (lo, hi) in rows:
        name = path.rsplit("/", 1)[-1][:33]
        print(f"{name:<34}{lo[0]*1000:>7.1f}{hi[0]*1000:>7.1f}"
              f"{max(abs(lo[1]), abs(hi[1]))*1000:>8.1f}"
              f"{lo[2]*1000:>7.1f}{hi[2]*1000:>7.1f}")

    allp = np.array([v for lohi in parts.values() for v in lohi])
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    edges = np.linspace(x0, x1, args.bins + 1)
    print(f"\nenvelope PROFILE along the tool axis "
          f"({args.bins} slices of {(x1-x0)*1000/args.bins:.0f} mm):")
    print(f"{'X span [mm]':<20}{'half-width':>12}{'Z min':>9}{'Z max':>9}"
          f"{'height':>9}")
    for i in range(args.bins):
        a, b = edges[i], edges[i + 1]
        sel = [(lo, hi) for lo, hi in parts.values()
               if hi[0] >= a and lo[0] <= b]
        if not sel:
            print(f"{a*1000:>7.0f} ..{b*1000:>7.0f}        (empty)")
            continue
        hw = max(max(abs(lo[1]), abs(hi[1])) for lo, hi in sel) * 1000
        zl = min(lo[2] for lo, _ in sel) * 1000
        zh = max(hi[2] for _, hi in sel) * 1000
        print(f"{a*1000:>7.0f} ..{b*1000:>7.0f}{hw:>12.1f}{zl:>9.1f}"
              f"{zh:>9.1f}{zh-zl:>9.1f}")
    print("\nshipped single box: X -18..240  |Y| 74.5  Z -100..70  (height 170)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
