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
- `world.py` — `RobotCell`: the boolean collision world. UR5e meshes +
  tool envelope on the flange + cell boxes + keep-in volume. Queries:
  `is_colliding(q)`, `path_valid(path)`, `min_distance(q)`,
  `first_collision(q)` (diagnosis). Visuals: `show(q)` (verdict-tinted),
  `replay(path)` (freeze at impact). Smoke: `p inspection/cell/world.py`.
- `viewer.py` — meshcat live preview of the cell (boxes + keep-in ghost).
- `usd_preview.py` — measurement rig for Antonio's Isaac USD (done its job).

## Contracts & decisions
- Shelf is modeled as a single keep-out envelope box, not per-board.
- Margin contract: discretization step × reach < padding — ASSERTED in
  `path_valid`. Env pairs 20 mm margin; self/tool-vs-robot pairs 5 mm
  (UR5e design clearances are 17-19 mm at park — 20 mm false-positives).
- Tool pairs stop at the forearm: wrist_1..3 are kinematically welded to
  the tool; the eyeballed cable-loop box overlaps wrist_1 by design.
- Keep-in is a bounds check on geometry AABBs, not a collision pair.
- The inspected object is NEVER a hard obstacle (`add_object` = display/
  standoff only) — a hard cup is unapproachable by construction.
- pinocchio internals are public API: `.model .data .geom_model .geom_data`
  — pyroboplan (MAY-184 tier 3) consumes them directly.
- GENERIC UR home grazes the top-right post in this cell (-0.3 mm) — use
  `Q_PARK` (searched, tool-down over the table) as the park pose.

## Does NOT belong here
- Motion planning or IK (motion/), camera IO (perception/),
  orchestration (run/).
