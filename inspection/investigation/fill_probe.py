#!/usr/bin/env python3
"""How big does the object actually look, per viewsphere cell?

Offline probe on a recorded run: take the fused object cloud, place a virtual
camera at every {h, v} cell of a shell of radius r, project the cloud with the
REAL D405 intrinsics, and measure the pixel bounding box. Answers the question
the design formula hand-waves: "object fills f of the frame" — which extent,
measured how, and how much does it swing between cells?

    p inspection/investigation/fill_probe.py --run 2408-seeded --r 0.35
    p inspection/investigation/fill_probe.py --run 2408-seeded --fill 0.45
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inspection.view.viewsphere import ViewSphere
from inspection.view.grid import H_BINS, V_ELEVATIONS

RUNS = Path(__file__).resolve().parents[1] / "data/runs"


def load_run(name):
    d = RUNS / name
    cloud = np.load(d / "fused_cloud.npy")
    intr = json.loads((d / "session.json").read_text())["intrinsics"]["color"]
    return cloud, intr


def project(cloud, T_base_cam, intr):
    """Cloud -> pixel bbox in the camera at T_base_cam. Camera convention is
    the viewsphere's (+X boresight, +Y image-right, +Z image-up), so pixels are
    u = ppx + fx * (y/x), v = ppy - fy * (z/x)."""
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p = (cloud - t) @ R                      # base -> camera frame
    p = p[p[:, 0] > 1e-3]                    # in front of the lens
    if len(p) == 0:
        return None
    u = intr["ppx"] + intr["fx"] * (p[:, 1] / p[:, 0])
    v = intr["ppy"] - intr["fy"] * (p[:, 2] / p[:, 0])
    return u.min(), u.max(), v.min(), v.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2408-seeded")
    ap.add_argument("--r", type=float, default=0.35, help="shell radius [m]")
    ap.add_argument("--fill", type=float, default=None,
                    help="instead: per-cell radius that hits this height fill")
    args = ap.parse_args()

    cloud, intr = load_run(args.run)
    W, H = intr["width"], intr["height"]
    center = cloud.mean(axis=0)
    mn, mx = cloud.min(0), cloud.max(0)
    print(f"{args.run}: {len(cloud)} pts, AABB {((mx - mn) * 1000).round(1)} mm, "
          f"bounding-sphere dia {np.linalg.norm(mx - mn) * 1000:.1f} mm")
    print(f"intrinsics {W}x{H} fx={intr['fx']:.1f} fy={intr['fy']:.1f}\n")

    sphere = ViewSphere(center, r=args.r)
    rows = []
    for v in range(len(V_ELEVATIONS)):
        row = []
        for h in range(H_BINS):
            bb = project(cloud, sphere.cam_pose(h, v), intr)
            if bb is None:
                row.append(float("nan"))
                continue
            u0, u1, v0, v1 = bb
            row.append((v1 - v0) / H)
        rows.append(row)

    print(f"height fill per cell at r={args.r:.2f} m  (fraction of {H} px)")
    for v in reversed(range(len(V_ELEVATIONS))):
        cells = "  ".join(f"{x:.2f}" for x in rows[v])
        print(f"  v{v} ({V_ELEVATIONS[v]:.0f}deg)  {cells}")
    flat = np.array(rows).ravel()
    print(f"  spread: min {flat.min():.2f}  max {flat.max():.2f}  "
          f"mean {flat.mean():.2f}  max/min {flat.max() / flat.min():.2f}x")

    if args.fill:
        # fill scales as 1/r, so r_target = r_probe * fill_at_probe / fill_target
        need = np.array(rows) * args.r / args.fill
        print(f"\nradius that hits {args.fill:.2f} height fill, per cell [m]")
        for v in reversed(range(len(V_ELEVATIONS))):
            cells = "  ".join(f"{x:.2f}" for x in need[v])
            print(f"  v{v} ({V_ELEVATIONS[v]:.0f}deg)  {cells}")
        print(f"  one-shell choices: min {need.min():.3f} (closest cell) "
              f"max {need.max():.3f} (safe for every cell) "
              f"mean {need.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
