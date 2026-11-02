# viewplan — view planning on the reconstructed surface (research)

## Purpose
Offline research path that plans views directly on geometry: step cloud →
mesh → surface samples → normal-offset candidate cameras → measurability
matrix → orthogonality-weighted greedy set cover → ordered camera poses.
Not wired into the live loop (the loop uses `view/` rays). Also hosts the
surface-completion benchmark that produced `fit/box.py:boxguard`.
Research notes: `docs/research/view-planning/` (`poc/`, `completion/`),
framing: `docs/research/geometry-aware-view-planning.md`.

## Files
- `contract.py`: `PlannedView`, `StepPlan`, `abstained`, `look_at_cv`,
  lossless JSON (`plan_to_json`/`plan_from_json`), `stable_json`, `fingerprint`.
- `mesh.py`: cloud → mesh, deterministic: normals oriented toward recorded
  cameras (centroid fallback), Poisson depth 8 single-threaded, trimmed to
  6 mm from the cloud; `table_plane` (used by `fit/box.py`), `height_above`.
- `planner.py`: `plan_step` and its pieces (`sample_surface`,
  `candidate_views`, `visibility`, `quality`, `greedy`, `score_views`);
  `VARIANTS` (`base`, `binary`, `cos2`, `cos75`, `altmesh`, `continuity`);
  run helpers (`run_dir`, `session_intrinsics`, `recorded_cams`).
- `render.py`: offscreen raycast renders of a planned view with the claimed
  covered patches splatted in (`render_view`, `framed_frac`).
- `cli.py`: `run`, `show`, `variants`, `crosseval`, `budget`, `curves` over
  runs' step clouds; multiprocessing `spawn`.
- `workspace.py`: `<run>-viewplan.rrd` Rerun review (via `record/rr/log.py`).
- `regions.py`: prototype: per-cell most-orthogonal pose inside each
  viewsphere cell's region of the shell.
- `complete.py`: surface-completion algorithm registry (`ALGOS`) for the bench.
- `truthbox.py`: meshcat GT cuboid annotator (`derived/<run>/truth/box.json`)
  + completion benchmark.
- `tests/`: synthetic clouds + contract/mesh/planner/render/cli/workspace tests.

## Contracts & invariants
- A planned pose is stored exactly like a recorded one: 4x4 `T_base_cam`,
  OpenCV convention (+Z boresight, +X right, +Y down), base frame, metres. The
  viewsphere +X-forward convention never appears here.
- Poses only: no IK, no reachability, no robot. The only physical guard is the
  table (no camera below it).
- Abstain is a first-class answer (no views, `reason` set).
- Deterministic: one seeded RNG in the surface sampler, ties break on lowest
  index, no Open3D samplers. `plan_step` twice with the same seed gives
  byte-identical JSON (tested).
- The objective is orthogonality, not coverage: gates (standoff, incidence,
  FOV, occlusion, table) only decide admissibility; greedy maximises
  cos-weighted gain.
- Completion code must never read `truth/box.json`: priors come from the cloud.
- Outputs go under `data/derived/<run>/viewplan/<variant>/` (gitignored).

## Dependencies
`fit/common.py` (spacing, normals radius, maturity gate), `fit/box.py`
(complete), `cell/geometry.py:min_yaw_aabb`, `view/grid.py` (regions),
`record/run.py`, `record/rr/log.py`. Open3D required.

## Does NOT belong here
- Anything the live loop needs at runtime other than `mesh.table_plane`.
- Reachability or motion (`view/`, `motion/`).
