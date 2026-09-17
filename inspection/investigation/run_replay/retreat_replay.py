#!/usr/bin/env python3
"""Replay the retreat decision from a recorded stance — did it, could it?

Born from 1509-cam-ai_bench1: an operator-commanded move to [10,0] captured
step 3, the refit swallowed the stance, and 14 straight refusals followed in
a 3+3+3+3+2 cadence — the signature of FOUR silent failed retreats (a failed
retreat only pub.logs; the record keeps nothing). This probe rebuilds both
worlds offline and answers, stage by stage:

  1. is the stance actually in collision under the POST-refit box? (the trap)
  2. does `_retreat_worker`'s exact math find a move under the STANCE box —
     free IK at +RETREAT_M radial, then tier-1 `direct_move`?

    p inspection/investigation/run_replay/retreat_replay.py \
        --run 1509-cam-ai_bench1 --stance-step 3 --approved-step 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from inspection.cell.geometry import object_box
from inspection.cell.world import RobotCell
from inspection.motion.direct import direct_move
from inspection.motion.ik import UR5eIK
from inspection.run.machine import RETREAT_M

RUNS = Path(__file__).resolve().parents[2] / "data/runs"


def load_step(run_dir, sid):
    d = run_dir / "steps" / f"{sid:03d}"
    step = json.loads((d / "step.json").read_text())
    cloud = np.load(d / "fused" / "cloud.npy")
    return step, cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="1509-cam-ai_bench1")
    ap.add_argument("--stance-step", type=int, default=3,
                    help="step whose capture refit swallowed the stance")
    ap.add_argument("--approved-step", type=int, default=2,
                    help="last step BEFORE the stance's move — its refit is "
                         "the box the stance was approved under")
    args = ap.parse_args()
    run_dir = RUNS / args.run

    stance, cloud_after = load_step(run_dir, args.stance_step)
    _, cloud_before = load_step(run_dir, args.approved_step)
    q = np.asarray(stance["joints_rad"], dtype=float)
    ik = UR5eIK()

    def world_with(points, tag):
        dims, pose = object_box(points)
        w = RobotCell()
        w.set_object("object", dims.tolist(), pose, parent="base")
        print(f"  {tag}: box {np.round(np.asarray(dims) * 1000, 1)} mm")
        return w

    print(f"{args.run} step {args.stance_step} at "
          f"{stance['view']['address']}, q recorded at capture")

    # -- 1. the trap: stance vs the box its own capture installed
    w_after = world_with(cloud_after, "post-refit box (what the live world became)")
    hit = w_after.first_collision(q) if w_after.is_colliding(q) else None
    print(f"  stance under post-refit box: "
          f"{'IN COLLISION  ' + str(hit[1]) if hit else 'valid — no trap'}")

    # -- 2. the escape: _retreat_worker's exact math under the stance box
    w_stance = world_with(cloud_before, "stance box (what approved the move)")
    if w_stance.is_colliding(q):
        print(f"  !! stance invalid even under its own approval box: "
              f"{w_stance.first_collision(q)[1]} — retreat validation can "
              f"never accept a start here")
    T = np.asarray(ik.fk(q))
    center = cloud_after.mean(axis=0)          # sphere center at retreat time
    out = T[:3, 3] - center
    out /= np.linalg.norm(out)
    T_r = T.copy()
    T_r[:3, 3] = T[:3, 3] + out * RETREAT_M
    branches = ik.branches(T_r, q_near=q)
    free = [qq for qq in branches if not w_stance.is_colliding(qq)]
    print(f"  retreat target +{RETREAT_M * 100:.0f} cm radial: "
          f"{len(branches)} IK branches, {len(free)} collision-free")
    path = None
    for qg in sorted(free, key=lambda qq: float(np.abs(qq - q).sum())):
        path = direct_move(w_stance, q, qg)
        if path is not None:
            break
    verdict = ("FOUND — retreat should have worked" if path is not None
               else "NONE — retreat honestly fails here")
    print(f"  tier-1 direct_move: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
