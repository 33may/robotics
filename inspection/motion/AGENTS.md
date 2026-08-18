# motion — motion solving (MAY-184)

## Purpose
Get the arm from viewpoint to viewpoint without hitting anything.
Empty package for now; will replace top-level `ik.py` and `safety.py`.

## Files
None yet. Planned: IK solver, 3-tier planner, ur_rtde executor.

## Contracts & decisions
- Decided design: 3-tier ladder over the boolean collision world from
  `cell/`: (1) validated straight joint move, (2) radial retract via the
  outer viewsphere shell, (3) pyroboplan RRT-Connect + shortcut, seeded;
  on timeout, skip the viewpoint.
- IK: EAIK all-8-branches on nominal DH, then Newton refine on the
  calibrated model — closed form breaks on calibrated DH.
- Never moveL between viewpoints: the chord passes through the object.
- cuRobo ruled out.
- Execution via ur_rtde `moveJ`, blends off.

## Does NOT belong here
- World modeling (cell/), viewpoint choice (view/), camera IO
  (perception/), loop orchestration (run/).
