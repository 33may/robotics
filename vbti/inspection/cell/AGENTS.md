# cell — the workcell model

## Purpose
Single source of truth for what the world looks like: robot, tool, table,
safety keep-outs, and the object primitive. Everything else queries it; it
imports nothing from sibling modules.

## Files
- `cell.yaml`: frame tree + geometry: `frames` (base→table probed once; the
  robot is welded to the table), `corners`/`planes` (safety keep-outs from
  corner taps, `calib/plane_fit.py`), `boxes` (`floor`, `table_slab`),
  `tool` (flange-mounted boxes: RG2, camera stack, cable loop).
- `geometry.py`: depth-cloud geometry: `deproject`, `fit_table`,
  clustering/seeded growth, `object_in_base(..., mask=)`, `CloudAccumulator`
  (base-frame clouds), `project_to_pixels`/`prompt_box`, `min_yaw_aabb`,
  `load_T_flange_cam`, `is_half_turn`/`rotate180`. Pure numpy/scipy.
- `world.py`: `RobotCell`: the boolean collision world (pinocchio + coal).
  UR5e meshes + tool envelope on the flange + cell boxes + keep-out panels
  and wedge prisms (`build_keepouts`). Queries: `is_colliding(q)`,
  `path_valid(path)`, `min_distance(q)`, `first_collision(q)`; `set_object`
  for the inspected object; visuals `show(q)`, `replay(path)`. `Q_PARK`.
  Smoke: `python world.py` (truth table + timings).
- `viewer.py`: meshcat live preview of `cell.yaml`, re-renders on save.
- `topview.py`: matplotlib top view of what the yaml says (`topview.png`).
- `usd_preview.py`: one-off measurement of the Isaac USD gripper (needs `pxr`).

## Contracts & decisions
- **Object identity is settled in IMAGE space, not by distance.** When
  `object_in_base` gets a `mask` it IS the answer and `seed`/`eps` are
  ignored. Distance guards (seeded growth at 20 mm, the 8 cm jump gate,
  percentile extent) fail when clutter is genuinely adjacent: on run
  2408-cup2 a cable behind the cup took the extent to 175.8 mm; masked, 91.2 mm.
  Check with `investigation/run_replay/mask_probe.py --run <run>` (drives the
  shipped `run.segmenter.object_view`).
- **`WORKSPACE` filters the SCENE, never the OBJECT.** It cuts the far room
  for the table fit and the unmasked fallback. Masked points are not cropped:
  cropping amputated 22% of a mug straddling `x=0.05` (92.8 vs 126.9 mm), on
  the side facing the robot. Signature: a flat, straight cloud edge on a
  `WORKSPACE` bound. Masked points stay bounded by `deproject`'s range clip,
  `above_table`, and the accumulator's jump gate.
- **The prompt box crops nothing.** It is a SAM prompt only; mask pixels
  outside it contribute points. `PROMPT_MARGIN_PX` 12→60 moves the extent 0.1 mm.
- A masked lift uses `depth_aligned` + COLOUR intrinsics (masks come from
  rgb), never `depth_raw` + ir_left.
- The table plane is always fitted on the FULL view, even when a mask is
  given: the mask deletes the table.
- `JUMP_GATE_M` and `BOX_PCT` stay although masking subsumes them: the gate
  is the only guard independent of the segmenter (a prompt drifting onto a
  second object yields a clean, confident, wrong mask).
- Margins: env pairs 20 mm (`DEFAULT_PADDING`), self/tool-vs-robot 5 mm
  (`SELF_PADDING`; UR5e park clearances are 17-19 mm). `path_valid` asserts
  step × lever < padding.
- Tool pairs stop at the forearm: wrist_1..3 are welded to the tool; the
  cable-loop box overlaps wrist_1 by design.
- Bounds are boxes (e.g. `floor`); there is no keep-in volume. Keep-outs are
  drawn by the viewer from the same `build_keepouts` the solver checks.
- The inspected object is a hard obstacle: `set_object` swaps the
  reconstructed primitive in each perception step. Endpoint validity at env
  padding doubles as the standoff. The START config must be valid too.
- Frames: `cell.yaml` holds UR controller `base` readings; the URDF
  `base_link` is `base` rotated pi about Z (`_mount_in_controller_base`).
- pinocchio internals are public (`.model .data .geom_model .geom_data`) for
  viewer/bench/diagnostics. The planner only calls `is_colliding` via OMPL's
  validity callback.
- The generic UR home grazes a post in this cell; park at `Q_PARK`.
- `geometry.py` must never import `eyes/` (`eyes/tools.py` imports it).

## Does NOT belong here
- Motion planning or IK (`motion/`), camera IO (`perception/`), segmentation
  (`run/segmenter.py`), orchestration (`run/`).
