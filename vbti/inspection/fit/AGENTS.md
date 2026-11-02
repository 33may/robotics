# fit — object primitive from a step cloud

## Purpose
Turns the accumulated object-only cloud of a step into a `Decomposition`
(one or more posed primitives). Fitters use what the scene already knows
(table under the object, gravity axis, object class) instead of fitting bare
points. `boxguard` is the only fitter; it won the completion bench.

## Files
- `contract.py`: `Primitive`, `Decomposition`, `KINDS`, lossless JSON round
  trip, `abstained`, `pose_from`, `frame_from_axis`.
- `box.py`: `boxguard_core` (yaw + per-side offsets + confidence),
  `boxguard` (open3d mesh), `boxguard_fit` (one `box` primitive), `boxmesh`. CLI `fit <ply>`.
- `apply.py`: `METHODS` registry (name → fitter), `record_of`, and the
  backfill CLI (`run`, `sweep`, `methods`) over recorded runs.
- `common.py`: cloud helpers (`cloud_spacing`, `estimate_normals`,
  `maturity_reason`, naming, fingerprints); also used by `viewplan/`.
- `tests/`: synthetic clouds (`synth.py`) + contract/box/apply tests.

## Contracts & invariants
- Kinds are closed: `box · cylinder · plane_patch · superquadric · mesh`.
  A fitter that cannot express its answer degrades to `mesh`. Required
  `params` keys are enforced at construction (`PARAM_KEYS`).
- Pose: 4x4 row-major, robot base frame, Z up; rotation columns are the
  primitive's axes, translation its centre. Box `extents` are full edge lengths.
- `abstain=True` is a valid answer (no primitives, `reason` set), not an error.
- `METHODS` is the only place a method name binds to code. The live loop
  (`run/machine.py`) and the backfill both call `METHODS[name]` and
  `record_of`, then `record/writer.py`, so live and backfilled
  `decomposition.json` are byte-compatible. The loop records a fitter
  exception or a method-name mismatch as an abstain.
- `boxguard_fit` `params.side_conf` {x+, x-, y+, y-} is per-wall evidence;
  the loop's face gate reads it.
- Extents are the outer envelope (q99.5-style), never a density mode.
- Nothing here may read `truth/` (bench ground truth). Priors come from the
  cloud and the scene only.
- Backfill input is `steps/NNN/fused/cloud.npy` (StepRecord v2); v1 steps are
  reported missing, never guessed.

## Dependencies
`box.py` lazily imports `cell/geometry.py:min_yaw_aabb` and
`viewplan/mesh.py:table_plane` (table height from the cloud's low tail, no
RANSAC, so the fit is deterministic). `apply.py` imports `record/`.

## Does NOT belong here
- Segmentation or cloud accumulation (`run/segmenter.py`, `cell/geometry.py`).
- Rendering of primitives (`record/rr/`), view choice (`view/`, `viewplan/`).
