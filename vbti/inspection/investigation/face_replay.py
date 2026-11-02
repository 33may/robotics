#!/usr/bin/env python3
"""Replay boxguard face-evidence derivation over a recorded run.

Answers, offline and before any live wiring: when does the stability gate
open, and what does the resolution ladder achieve per face?

Usage (repo root, robo env):
    p vbti/inspection/investigation/face_replay.py run 1409-lan-ai_bench1
"""
import sys
from pathlib import Path

import numpy as np

from inspection.cell.world import RobotCell
from inspection.fit.box import boxguard_fit
from inspection.motion.ik import UR5eIK
from inspection.view.faceview import (FACES, FACE_STABLE_TOL_DEG_DEFAULT,
                                      face_frame, face_radius,
                                      normals_delta_deg, resolve_face_view)
from inspection.view.viewsphere import T_FLANGE_CAM

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"
FY_FALLBACK, HPX_FALLBACK = 430.0, 480    # D405-ish; stamped when used


def run(run_id: str):
    rdir = RUNS / run_id
    steps = sorted((rdir / "steps").iterdir())
    world, ik = RobotCell(), UR5eIK()
    print(f"== {run_id}: {len(steps)} steps  "
          f"(intrinsics fallback fy={FY_FALLBACK} h={HPX_FALLBACK})")
    prev, stable = None, 0
    for sdir in steps:
        cloud = np.load(sdir / "fused" / "cloud.npy")
        d = boxguard_fit(cloud)
        prim = d.primitives[0]
        delta = None if prev is None else normals_delta_deg(prev, prim)
        stable = (stable + 1 if delta is not None
                  and delta <= FACE_STABLE_TOL_DEG_DEFAULT else 0)
        gate = "OPEN" if stable >= 2 else "closed"
        print(f"step {sdir.name}: delta="
              f"{'-' if delta is None else f'{delta:5.1f}deg'} gate={gate}")
        if gate != "OPEN":
            prev = prim
            continue
        yaw = float(np.arctan2(prim.axes[1, 0], prim.axes[0, 0]))
        world.set_object("object", [float(v) for v in prim.params["extents"]],
                         list(map(float, prim.center)) + [0.0, 0.0, yaw])
        for face in FACES:
            _, _, ext = face_frame(prim, face)
            r = face_radius(ext, FY_FALLBACK, HPX_FALLBACK)
            res = resolve_face_view(world, ik, T_FLANGE_CAM, prim, face, r)
            if res is None:
                print(f"    {face:3s}: UNREACHABLE (r={r:.2f})")
            else:
                print(f"    {face:3s}: {res.rung:5s} theta={res.theta_deg:4.1f} "
                      f"roll={np.degrees(res.roll):.0f} r={r:.2f}")
        prev = prim


def main():
    import fire
    from fire import core
    core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire({"run": run})


if __name__ == "__main__":
    main()
