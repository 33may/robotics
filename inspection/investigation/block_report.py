#!/usr/bin/env python3
"""Why is a viewsphere cell blocked — and by WHAT, exactly?

`reachability()` answers yes/no. That conflates three very different things:

  * no IK solution at all (the pose is outside the arm's reach or singular)
  * IK exists, but every branch collides — and then WHICH pair collides
    decides what to do about it (shrink the tool hull? move the object?
    raise the ring?)

This walks every cell, enumerates IK branches for each allowed roll, and
attributes each rejection to a named geometry pair via `first_collision`.

    p inspection/investigation/block_report.py --run 2408-cup1
    p inspection/investigation/block_report.py --run 2408-cup1 --r 0.30
    p inspection/investigation/block_report.py --run 2408-cup1 --raw-cloud
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import inspection.view.viewsphere as vs
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.view.grid import H_BINS, V_ELEVATIONS, object_extent, radius_for_extent

RUNS = Path(__file__).resolve().parents[1] / "data/runs"


def largest_component(pts, eps):
    """Biggest single-linkage blob — drops cable/leak tails that the run's own
    growth threshold could not separate."""
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if not len(pairs):
        return pts
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(len(pts), len(pts)))
    _, lab = connected_components(g, directed=False)
    return pts[lab == np.bincount(lab).argmax()]


def probe(world, ik, sphere):
    """{cell: (state, detail)} — state in {ok, no-ik, blocked}."""
    out = {}
    for h, v in sphere.cells():
        n_branch, culprits, ok = 0, Counter(), False
        for roll in vs.ROLLS:
            T = sphere.flange_pose(h, v, roll)
            branches = ik.branches(T)
            n_branch += len(branches)
            for q in branches:
                if not world.is_colliding(q):
                    ok = True
                    break
                hit = world.first_collision(q)
                if hit is not None:
                    culprits[tuple(sorted(hit[1]))] += 1
            if ok:
                break
        if ok:
            out[(h, v)] = ("ok", None)
        elif n_branch == 0:
            out[(h, v)] = ("no-ik", None)
        else:
            out[(h, v)] = ("blocked", culprits.most_common(1)[0][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2408-cup1")
    ap.add_argument("--r", type=float, default=None, help="override shell radius")
    ap.add_argument("--eps", type=float, default=0.010,
                    help="clean the cloud at this linkage distance [m]")
    ap.add_argument("--raw-cloud", action="store_true",
                    help="use the cloud as the run left it, tails and all")
    args = ap.parse_args()

    cloud = np.load(RUNS / args.run / "fused_cloud.npy")
    pts = cloud if args.raw_cloud else largest_component(cloud, args.eps)
    mn, mx = pts.min(0), pts.max(0)
    r = args.r or radius_for_extent(object_extent(mn, mx), 431.3, 480)
    print(f"{args.run}: {len(pts)}/{len(cloud)} pts, AABB "
          f"{((mx - mn) * 1000).round(1)} mm, shell r={r:.3f} m, "
          f"rolls {np.rad2deg(vs.ROLLS).round(0)}")

    world, ik = RobotCell(), UR5eIK()
    dims = np.maximum(mx - mn + 0.04, 0.05)
    world.set_object("object", dims.tolist(),
                     [*((mn + mx) / 2).tolist(), 0, 0, 0], parent="base")
    res = probe(world, ik, vs.ViewSphere(pts.mean(axis=0), r=r))

    sym = {"ok": "#", "no-ik": "x", "blocked": "o"}
    print("\n  # reachable   o blocked by collision   x no IK solution")
    for v in reversed(range(len(V_ELEVATIONS))):
        row = "".join(sym[res[(h, v)][0]] for h in range(H_BINS))
        print(f"  v{v} {V_ELEVATIONS[v]:>2.0f}deg |{row}|")
    print("            h" + "".join(str(h % 10) for h in range(H_BINS)))

    tally = Counter(s for s, _ in res.values())
    print(f"\n  {tally['ok']}/36 reachable · {tally['blocked']} blocked · "
          f"{tally['no-ik']} no IK")
    blame = Counter(d for s, d in res.values() if s == "blocked")
    if blame:
        print("\n  what blocks the blocked cells:")
        for pair, n in blame.most_common():
            print(f"    {n:>2} cells   {pair[0]}  vs  {pair[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
