# view — viewsphere & coverage

## Purpose
Enumerate candidate viewpoints on a sphere around the object primitive
and track which have been seen. Later, the VLM picks cells from this
enumeration — it replaces the picker, nothing else.

## Files
- `grid.py` — the LIGHT half: cell addressing and the words for it.
  `H_BINS`/`V_ELEVATIONS`/`DEFAULT_R`, `object_extent` + `radius_for_extent`
  (the derived shell radius and its constants), `step_delta` (h wraps), `neighbors`
  (4-connected, v clamps), `cell_gloss` ("two steps right, higher"),
  `coverage_map` (ASCII bitmap), `moves_from` (order + name an allowed set).
  numpy only — `eyes/` and the future `AgentDecider` both import it, so it
  must never grow a motion/IK dependency. `p inspection/tests/test_grid.py`.
- `viewsphere.py` — v2, UR5e stack. `ViewSphere(center, r)`: {h, v} cells
  (12 azimuth × 3 elevation bins, ONE shell of configurable radius, h0
  faces the base — azimuth computed from the actual center), cell →
  camera pose → flange pose via nominal `T_FLANGE_CAM` (replace after
  hand-eye), `reachability()` map, `plan_to_cell()` = roll search + the
  motion ladder. `p inspection/view/viewsphere.py [--r R] [--demo]`.

## Contracts & decisions
- The grid vocabulary is shared, the geometry is not (Anton 2026-08-24):
  cell indices, gloss and coverage rendering live in `grid.py` for both AI
  tiers; poses, reachability and planning stay in `viewsphere.py`. Move
  legality comes from motion, so `moves_from` receives the feasible set.
- Camera ROLL about the boresight is restricted to `{0, 180}` (Anton
  2026-08-24), upright preferred. The old twelve-roll ladder traded image
  orientation for reachability — it bought NOTHING: measured on the demo cup
  at r=0.244, `{0,180}` reaches the same 26/36 cells as all twelve rolls,
  while `{0}` alone reaches 20/36 and loses the ENTIRE 10 deg ring (0/12),
  which is the only ring that sees the object's side straight on.
  Both allowed rolls are lossless to undo and keep the frame landscape, so
  every capture can be shown to the VLM in one orientation. Raising the low
  ring does not help — 10/12/15/20 deg give identical upright maps, because
  the blocker is the arm's own column, not the elevation.
- The tilt of a capture is recoverable from its pose alone —
  `atan2(up . X_cam, -(up . Y_cam))` on `T_base_cam` — so no roll needs to be
  stored and already-recorded runs can be corrected retroactively.
  De-rotation must NOT touch the geometry path: `object_in_base()` deprojects
  raw depth against `T_base_cam`, so rotated pixels would corrupt the cloud.
- The object is a HARD obstacle in the world (2026-08-18 decision) —
  no separate standoff; endpoint validity at env padding is the standoff.
  Radius must respect the tool, but the old "closed tip rides ~19 cm ahead
  of the camera, r < 0.30 pierces the object" note was WRONG: the measured
  TCP is +240 mm in flange X (`cell/cell.yaml`) and the calibrated lens sits
  at +142 mm, so only **98 mm** of tool leads the camera.
- **`R_MIN` is a property of the OBJECT and its position, not of the tool.**
  An early sweep on the demo cup at the demo position said 0.22 was fine;
  with the real object where it actually sits (`2408-cup1`) that shell
  reaches 19/36 cells against 28/36 at 0.24. So `R_MIN=0.24`, `R_MAX=0.40`,
  and a shell is only trustworthy once `investigation/block_report.py` has
  been run for that object — the band does not transfer between scenes.
  The 0.22 cliff is neither the tool nor the cup: the object is inflated to
  AABB+40 mm and coal adds 20 mm per side, so a 94x115x111 mm cup is seen as
  ~174x195x191 mm. Thinning that recovers 0.22 entirely (+20 mm box → 29/36),
  but the padding IS the standoff (2026-08-18), so the floor moves instead.
- The shell radius is DERIVED from the object and FROZEN, not declared
  (Anton 2026-08-24). `_recenter()` computes it on the survey cloud the
  first time and never again; `--r` is an override for experiments.
  Frozen because cells key coverage and evidence — a shell that tightened
  as the cloud grew would re-point addresses already visited.
  Constants live in `grid.py`: `EXTENT_RULE="sphere"` (AABB diagonal — the
  only rule that never under-shoots; `upright` slams flat/wide objects into
  R_MIN, `footprint` does the same to tall ones) and `FILL_TARGET=0.62`
  nominal ≈ 0.45 measured. Tools to re-fit them without a robot:
  `investigation/fill_probe.py` (projected fill per cell) and
  `investigation/frame_preview.py` (re-frames recorded captures).
- Coverage = a bitmap over the enumerated cells, not surface patches.
- Blocked cells stay visible as facts, invisible as actions.
- Ring elevations 10/40/70 deg above the table (Anton 2026-08-18). Higher
  rings crowd near the pole (at 70 deg the ring radius is cos70 ≈ 0.34 r), so
  a top-heavy set gives far less view diversity than the cell count suggests.
- Measured (r=0.35, demo cup, 10/40/70): 31/36 cells reachable. The 10 deg
  ring loses the four cells nearest the base — a near-horizontal view there is
  blocked by the arm's own column.

## Does NOT belong here
- IK / reachability solving internals (motion/), object modeling
  (cell/), capture (perception/), the loop that walks the cells (run/).
