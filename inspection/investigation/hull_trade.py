#!/usr/bin/env python3
"""What does a more honest tool hull buy, in reachable viewsphere cells?

`cell.yaml` models the RG2 + camera stack as ONE box that takes the worst
value of every axis independently — closed length AND max-open width AND full
bracket height — so it describes a gripper that never exists: 149 x 170 mm of
slab carried all the way to the fingertips. Under tilt that phantom slab
sweeps into the robot's own upper arm.

This scores candidate hulls by counting reachable cells against a real
recorded object, so a hull change is justified by a number rather than by how
sensible it looks in a viewer.

SAFETY: every hull here must COVER the hardware. Boxes are built from CAD
(`tool_hull_from_usd.py`), the pendant TCP (240 mm) and the calibrated lens
position (X=+142.4, Z=-47.5 mm), then inflated. A hull that scores well by
under-modelling the tool is how you crash one — score is not the objective,
it only breaks ties between hulls that are all truthful.

    p inspection/investigation/hull_trade.py --run 2408-cup1
"""
import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import inspection.view.viewsphere as vs
from inspection.cell.world import CELL_YAML, RobotCell
from inspection.investigation.block_report import largest_component, probe
from inspection.motion.ik import UR5eIK

RUNS = Path(__file__).resolve().parents[1] / "data/runs"

#: (name, [(x0, x1, half_width, z0, z1), ...]) in millimetres, flange frame.
#: The cross comes from `hull_blueprint`, which is also what verifies it covers
#: every CAD part — one definition, checked in one place.
from inspection.investigation.hull_blueprint import (CURRENT, PROPOSED,
                                                     coverage_gaps, usd_parts,
                                                     MOUNT_STACK_MM)

HULLS = [
    ("current: one box -18..240 x +-74.5 x -100..70", CURRENT),
    ("proposed CROSS: base | slender body | camera slab", PROPOSED),
]


def world_with(boxes, obj_dims, obj_mid):
    """RobotCell whose tool is `boxes`, holding the recorded object."""
    cfg = yaml.safe_load(open(CELL_YAML))
    cfg["tool"] = {}
    for i, (x0, x1, hw, z0, z1) in enumerate(boxes):
        cfg["tool"][f"box{i}"] = {
            "dims": [(x1 - x0) / 1000.0, 2 * hw / 1000.0, (z1 - z0) / 1000.0],
            "pose": [(x1 + x0) / 2000.0, 0.0, (z1 + z0) / 2000.0, 0, 0, 0]}
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, fh)
    fh.close()
    world = RobotCell(yaml_path=fh.name)
    world.set_object("object", obj_dims, [*obj_mid, 0, 0, 0], parent="base")
    Path(fh.name).unlink()
    return world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2408-cup1")
    ap.add_argument("--eps", type=float, default=0.010)
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0.22, 0.24, 0.26, 0.28])
    args = ap.parse_args()

    cloud = np.load(RUNS / args.run / "fused_cloud.npy")
    pts = largest_component(cloud, args.eps)
    mn, mx = pts.min(0), pts.max(0)
    dims = np.maximum(mx - mn + 0.04, 0.05).tolist()
    mid = ((mn + mx) / 2).tolist()
    centre = pts.mean(axis=0)
    ik = UR5eIK()
    print(f"{args.run}: {len(pts)}/{len(cloud)} pts (eps {args.eps*1000:.0f} mm), "
          f"object AABB {((mx-mn)*1000).round(1)} mm\n")

    head = "".join(f"{r:>8.2f}" for r in args.radii)
    print(f"{'hull':<58}{head}   reachable /36")
    for name, boxes in HULLS:
        counts, blames = [], []
        for r in args.radii:
            world = world_with(boxes, dims, mid)
            res = probe(world, ik, vs.ViewSphere(centre, r=r))
            counts.append(sum(1 for s, _ in res.values() if s == "ok"))
            blames.append(Counter(d for s, d in res.values() if s == "blocked"))
        print(f"{name:<58}" + "".join(f"{c:>8}" for c in counts))
        worst = blames[0].most_common(1)
        if worst:
            print(f"{'':<58}  at r={args.radii[0]:.2f}: {worst[0][1]}x "
                  f"{worst[0][0][0]} vs {worst[0][0][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
