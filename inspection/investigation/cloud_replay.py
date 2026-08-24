#!/usr/bin/env python3
"""Replay a run's cloud accumulation view by view, offline.

Answers "which capture put THAT in the object cloud, and which guard should
have stopped it" without a robot: every input the loop used is on disk
(`depth_raw` + `meta.json:T_base_cam` + `session.json` intrinsics), and the
rgb rotation never touched depth, so the replay is exact.

Reports per view: points offered by `object_in_base`, points the jump gate
rejected, the resulting cloud extent (raw and trimmed), and how far the new
points reached from the cloud that existed before them — the quantity
`JUMP_GATE_M` is thresholding.

    p inspection/investigation/cloud_replay.py --run 2408-cup2
    p inspection/investigation/cloud_replay.py --run 2408-cup2 --gate 0.04
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inspection.cell.geometry import (BOX_PCT, CloudAccumulator, GROW_EPS_M,
                                      object_in_base)

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"


def views(run_dir: Path):
    """(pose_id, depth_raw, T_base_cam) per capture, in capture order."""
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        meta = json.loads((d / "meta.json").read_text())
        yield (meta["pose_id"], np.load(d / "depth_raw.npy"),
               np.array(meta["T_base_cam"], dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gate", type=float, default=None,
                    help="override JUMP_GATE_M [m]; 0 disables the gate")
    ap.add_argument("--eps", type=float, default=GROW_EPS_M,
                    help="override GROW_EPS_M [m]")
    ap.add_argument("--pct", type=float, default=BOX_PCT)
    args = ap.parse_args()

    run_dir = RUNS / args.run
    session = json.loads((run_dir / "session.json").read_text())
    intr = session["intrinsics"]["ir_left"]
    scale = session["depth_scale_m_per_unit"]

    if args.gate is not None:
        import inspection.cell.geometry as geom
        geom.JUMP_GATE_M = args.gate

    acc = CloudAccumulator()
    print(f"{'view':>4} {'offered':>8} {'rejected':>9} {'reach':>8} "
          f"{'N':>6}  {'raw extent mm':>20}  {'trimmed mm':>20}")
    for pose_id, depth, T_bc in views(run_dir):
        before = acc.points.copy()
        view = object_in_base(depth, intr, scale, T_bc,
                              seed=before, eps=args.eps)
        pts = view["points"]
        offered = len(pts)
        # How far this view's points sit from the cloud that already existed —
        # the max is exactly what the jump gate thresholds.
        reach = float("nan")
        if len(before) and offered:
            from scipy.spatial import cKDTree
            reach = float(cKDTree(before).query(pts)[0].max())
        rejected = acc.add(pts) if offered else 0
        mn, mx = acc.aabb()
        tlo, thi = acc.aabb(pct=args.pct)
        print(f"{pose_id:>4} {offered:>8} {rejected:>9} "
              f"{reach * 1000:>7.0f}m {len(acc.points):>6}  "
              f"{str(np.round((mx - mn) * 1000, 1)):>20}  "
              f"{str(np.round((thi - tlo) * 1000, 1)):>20}")

    P = acc.points
    print(f"\nfinal cloud {len(P)} pts, centroid {np.round(P.mean(0), 3)}")
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    for eps in (0.01, 0.02, 0.03):
        n, lab = connected_components(
            cKDTree(P).sparse_distance_matrix(cKDTree(P), eps), directed=False)
        sizes = np.sort(np.bincount(lab))[::-1][:4]
        print(f"  linkage {eps*1000:.0f} mm -> {n} components, largest {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
