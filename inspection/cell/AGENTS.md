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
  `object_in_base(..., mask=)` is the masked lift; `project_to_pixels` /
  `prompt_box` turn a known cloud into a box in a new view. Pure numpy —
  it must NEVER import `eyes` (`eyes/tools.py` imports this module, so it
  would be a cycle); the mask arrives as data from `run/`.
- `world.py` — `RobotCell`: the boolean collision world. UR5e meshes +
  tool envelope on the flange + cell boxes + keep-in volume. Queries:
  `is_colliding(q)`, `path_valid(path)`, `min_distance(q)`,
  `first_collision(q)` (diagnosis). Visuals: `show(q)` (verdict-tinted),
  `replay(path)` (freeze at impact). Smoke: `p inspection/cell/world.py`.
- `viewer.py` — meshcat live preview of the cell (boxes + keep-in ghost).
- `usd_preview.py` — measurement rig for Antonio's Isaac USD (done its job).

## Contracts & decisions
- **Object identity is settled in IMAGE space, not by distance** (Anton
  2026-08-24). When `object_in_base` gets a `mask` it IS the answer and
  `seed`/`eps` are ignored. Every geometric guard we built — seeded growth at
  20 mm, the 8 cm jump gate, a percentile extent — asks a question about
  DISTANCE, and on run 2408-cup2 view 003 a cable passed behind the cup
  within 20 mm: genuinely adjacent, so distance had nothing to say and the
  object extent went to 175.8 mm. Masked, the same frames give 91.2 mm.
  Verify with `p inspection/investigation/mask_probe.py --run <run>`, which
  drives the shipped `run.segmenter.object_view`, not a copy of it.
- **`WORKSPACE` filters the SCENE, never the OBJECT** (Anton 2026-08-24).
  It exists to cut the far room so the table-plane fit and the depth fallback
  see the workcell only; a masked pixel is by definition not scene clutter, so
  `object_in_base(mask=)` does NOT `crop_workspace` its points. Cropping them
  made the cloud a hard axis-aligned slab whenever an object sat near a face:
  on run 2408-cup4 the mug straddled `x=0.05`, 22% of every masked view was
  amputated, and it measured 92.8 mm instead of 126.9 — with the missing half
  on the side FACING THE ROBOT, so the collision box was ~32 mm short exactly
  where the arm reaches across. The signature is a suspiciously flat, straight
  cloud edge; the check is whether the extent lands on a `WORKSPACE` bound.
  Masked points stay bounded by `deproject`'s range clip, `above_table`, and
  the accumulator's jump gate.
- **The prompt box does not crop anything.** It is a SAM prompt and nothing
  else — measured on 2408-cup4, mask pixels outside the box contribute real
  points (446 in view 012), and sweeping `PROMPT_MARGIN_PX` 12→60 changes the
  final extent by 0.1 mm. If a cloud looks clipped, it is not this.
- Masks come from rgb, so a masked lift must use `depth_aligned` + the
  COLOUR intrinsics, never `depth_raw` + ir_left. Measured agreement between
  the two reconstructions is 2.6 mm, under the hand-eye residual, so this
  costs nothing and removes all mask warping.
- The table plane is always fitted on the FULL view even when a mask is
  given — the plane needs the table and the mask deliberately deletes it.
- `JUMP_GATE_M` and `BOX_PCT` STAY even though masking subsumes them
  (Anton 2026-08-24). Measured, the gate now changes the result by ~1 mm,
  but it is the only guard independent of the model: a prompt box that drifts
  onto the second cup in the background would produce a clean, confident,
  completely wrong mask, and nothing appearance-based can catch that. The
  box-growth gate that was prototyped alongside it was NOT shipped —
  segmentation beats it outright (91 vs 134 mm).
- Shelf is modeled as a single keep-out envelope box, not per-board.
- Margin contract: discretization step × reach < padding — ASSERTED in
  `path_valid`. Env pairs 20 mm margin; self/tool-vs-robot pairs 5 mm
  (UR5e design clearances are 17-19 mm at park — 20 mm false-positives).
- Tool pairs stop at the forearm: wrist_1..3 are kinematically welded to
  the tool; the eyeballed cable-loop box overlaps wrist_1 by design.
- Bounds are modeled as BOXES (e.g. the wide `floor` slab) — the keep-in
  volume mechanism was removed (Anton, 2026-08-18).
- The inspected object IS a hard obstacle (Anton 2026-08-18, reverses the
  earlier soft-object rule): `set_object` swaps the reconstructed primitive
  in every perception step; we look, never touch — endpoint validity at env
  padding doubles as the standoff. NOTE: the START config must be valid too;
  a park pose hovering over the object dies at step 0.
- pinocchio internals are public API: `.model .data .geom_model .geom_data`.
  The planner no longer consumes them (OMPL queries `is_colliding` through a
  validity callback), but viewer/bench/diagnostics still do.
- GENERIC UR home grazes the top-right post in this cell (-0.3 mm) — use
  `Q_PARK` (searched, tool-down over the table) as the park pose.

## Does NOT belong here
- Motion planning or IK (motion/), camera IO (perception/),
  orchestration (run/).
