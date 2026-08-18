# cell — the workcell model (MAY-183)

## Purpose
Single source of truth for what the world looks like: robot, table,
shelf, and the object primitive. Everything else queries it; it imports
nothing from sibling packages.

## Files
- `cell.yaml` — declared cuboids with the frame tree: base→table probed
  once (robot is welded to the table, transform is permanent);
  table→shelf tape-measured.
- `geometry.py` — depth-cloud geometry: deproject, table fit/rectify,
  cluster, `CloudAccumulator` (SO-101 era, base-frame clouds).
- `world.py` (planned) — `RobotCell`: pinocchio GeometryModel,
  `is_colliding(q)` / `path_valid(path)`, explicit collision-pair whitelist.
- `viewer.py` — meshcat live preview of the cell.

## Contracts & decisions
- Shelf is modeled as a single keep-out envelope box, not per-board.
- Margin contract: discretization step × reach < distance_padding.

## Does NOT belong here
- Motion planning or IK (motion/), camera IO (perception/),
  orchestration (run/).
