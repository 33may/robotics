# view — cell and face rays, coverage

## Purpose
Say where the camera may stand. A view is a RAY (object-anchored direction)
and its distance is the closest provably-safe sample on it. The VLM picks
addresses from this enumeration — it replaces the picker, nothing else.
Design: `docs/blueprints/ray-views.md`.

## Files
- `grid.py` — the LIGHT half: cell addressing and the words for it.
  `H_BINS`/`V_ELEVATIONS`/`DEFAULT_R`/`R_MAX`, `step_delta` (h wraps),
  `neighbors` (4-connected, v clamps), `cell_gloss` ("two steps right,
  higher"), `coverage_map` (ASCII bitmap), `moves_from` (order + name an
  allowed set). numpy only: `logic/eyes/`, `logic/brain/`, `logic/viewplan/` and `logic/run/` import
  it, so it must never grow a motion/IK dependency. Tests: `tests/test_grid.py`.
- `rays.py` — the sampler and the vocabulary built on it.
  `sample_ray` walks depths in `RAY_STEP_M` steps from the object surface
  out to `R_MAX`, cheapest check first: sensor min range (`MIN_RANGE_M`,
  D405 datasheet), IK branches, collision — and returns the valid samples of
  the roll whose closest one is closest. `cell_ray(center, h, v)` is a cell's
  ray (centroid anchor, grid direction, same h=0 convention as `ViewSphere`);
  `resolve_cell` takes its closest sample that still keeps the whole belief
  box in frame; `cell_map` resolves the WHOLE grid and writes `None` for an
  inaccessible cell — never a missing key, because that dict is the machine's
  accessibility gate. `face_family` is a face's candidate rays: parallel +z
  shifts of the fronto-parallel ray first (a plane parallel to the sensor
  projects as a pure similarity — no keystone at any shift), then the 5..60
  deg cone. `resolve_face` scans the WHOLE shift rung and takes the member
  whose closest sample is shallowest (tie: smaller shift); tilt is consulted
  only when no shift has a sample. Tests: `tests/test_rays.py`.
- `faceview.py` — boxguard face GEOMETRY: `FACES`, `face_frame` (center,
  outward normal, wall extents), `face_corners`, `face_cam_pose` (camera on a
  normal ray), `top_align_roll` (lay the footprint's wider side along the
  image width), `normals_delta_deg` + `FACE_STABLE_TOL_DEG_DEFAULT` (gate).
  It resolves nothing — `rays.py` builds the family out of these frames.
- `viewsphere.py` — the cell FRAME: `ViewSphere(center, r)` gives `cells()`,
  `cell_dir()` and `cam_pose()`, and `r` is only the radius those markers are
  drawn at (`machine._recenter` feeds it the mean resolved depth). Also holds
  `T_FLANGE_CAM` (calibrated hand-eye, x-forward convention) and `ROLLS`;
  `rays.py` imports `ROLLS`, `logic/run/collect.py` and `logic/run/machine.py` import
  `T_FLANGE_CAM`.

## Contracts & decisions
- **Distance is emergent, not computed.** No fill target, no extent rule,
  no closeness slider, no radius floor: the sampler's stop conditions replace
  them all, and the closest passing depth wins. The earlier fixed-radius shell
  and its resolution ladder were deleted.
- The grid vocabulary is shared, the geometry is not: cell indices, gloss
  and coverage rendering live in `grid.py`; poses and ray resolution stay in `viewsphere.py`/`rays.py`. Move
  legality comes from motion, so `moves_from` receives the feasible set.
- Camera ROLL about the boresight is restricted to `{0, 180}`, upright preferred. The old twelve-roll ladder traded image
  orientation for reachability — it bought NOTHING: measured on the demo cup
  at r=0.244, `{0, 180}` reaches the same 26/36 cells as all twelve rolls,
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
- The object is a HARD obstacle in the world: no separate standoff;
  endpoint validity at env padding is the standoff.
  Depth must respect the tool, but the old "closed tip rides ~19 cm ahead
  of the camera" note was WRONG: the measured TCP is +240 mm in flange X
  (`logic/cell/cell.yaml`) and the calibrated lens sits at +142 mm, so only
  **98 mm** of tool leads the camera.
- Coverage = a bitmap over the enumerated cells, not surface patches.
- Blocked cells stay visible as facts, invisible as actions.
- Ring elevations 10/40/70 deg above the table. Higher
  rings crowd near the pole (at 70 deg the ring radius is cos70 ≈ 0.34 r), so
  a top-heavy set gives far less view diversity than the cell count suggests.

## Does NOT belong here
- IK / reachability solving internals (logic/motion/), object modeling
  (logic/cell/), capture (logic/perception/), the loop that walks the cells (logic/run/).
