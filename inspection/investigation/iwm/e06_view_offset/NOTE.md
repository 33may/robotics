# e06: what causes the per-view offset

**Opened:** 2026-10-12. **Rung:** side question under rungs 3–6 (nodes in 3D, two views one
tree, voxels under the tree). Read-only on the 2209 recordings. Serves the decision
"recalibrate, or correct per view in the IWM".

## Question

The same physical point seen from two views of one run lands ~10 mm apart (p90 19, worst
30), and the disagreement is ONE RIGID TRANSLATION PER VIEW (1.6 mm left after removing it;
`cross-view-registration-error-2209-runs`, e03 `pairs_3d.json`, e05 `pairs.json`). Which
error source produces these per-view translations, what single correction removes them, and
how much is left after it?

## What is already measured (inputs, not results)

- compute demo5, c5.2 (step 2, survey cell [9,1], camera 0.267 m up, 40° down, ~0.3 m
  from the ports) vs c5.6 (step 6, face x+, camera 0.071 m up, horizontal, ~0.12 m):
  c5.2's ports sit (−0.4, −2.5, +8.2) mm from c5.6's in base (e05, 1 mm V; e03 read the
  same pair as (−0.2, −0.4, +6.4)). Nearly pure +z.
- 2026-10-06 SIFT probe (22 pairs, 4 runs; scripts in a session scratchpad, ported here as
  `pairs.py`): the survey step 0 is the worst view in every run (10–14 mm), the pairs
  without it are 2–7 mm; matches sit 2–35 px off the epipolar lines the recorded poses
  predict (0.2–3° of rotational disagreement), so the poses disagree with the images
  independently of depth.
- Per-step `geometry.plane` already in every `step.json` (RANSAC table on the
  workspace-cropped view), read this morning: the fitted table sits **+10–12 mm above the
  probed slab (z = −4.9 mm, `calib/table_probe_2026-08-18.yaml`) in the 40–90° views at
  0.25–0.35 m and +20–25 mm in the survey view at 0.5 m**. The table is bare plywood,
  the object stands on it (`steps/000/rgb.png`).

## What the recordings give us (the instruments)

Every captured step stores `joints_rad`, `T_base_flange` (nominal-DH FK of the joints),
`T_base_cam` (= flange · `calib/T_flange_cam_2026-08-19.npy`), `depth_raw` (depth
intrinsics, rectified, zero distortion), `depth_aligned` (colour intrinsics, distortion
ignored by `cell.geometry.deproject`), `mask.png`, `t_arrived`, `t_captured`. The capture
order is frame first, then joints (`run/rigs.py:capture`), robot stopped.

Four instruments, all per view, all offline:

- **I1 table plane.** Pixels outside the dilated object mask, depth 0.10–0.90 m, deprojected
  in the CAMERA frame; RANSAC + least-squares plane. Then in base through the recorded
  pose. Observables: the plane's height error `h` at the optical-axis hit on the probed slab;
  the normal's tilt `τ` from +z, expressed in the camera frame; and the per-pixel range
  residual `Δr(r)` = measured range − range to the probed slab along that pixel's ray.
- **I2 feature pairs.** SIFT inside the object masks between every two views of a run, each
  match lifted with its view's own depth and pose; the per-view translation `v_i` solved by
  least squares over the run's pairs (mean zero per run). Also e05's part pairs
  (c5.2/c5.6) and e03's.
- **I3 the joint configuration** of each view: elbow sign / wrist-2 sign = the IK family.
- **I4 the same view from two depth sources:** `depth_raw` + depth intrinsics vs
  `depth_aligned` + colour intrinsics.

Two natural experiments sit in the data:

- **NE1.** The survey pose is identical in all 17 runs (0.04 mm / 0.006°), but reached by
  two joint families: A = elbow −116°, wrist-2 +91° (8 runs) and B = elbow +104°,
  wrist-2 −69° (9 runs). All 31 hand-eye calibration poses lie in family A's region.
  Same camera pose, different arm: any difference is FK, not camera.
- **NE2.** Top-down views (face top, 70° cells) and horizontal close views (face x±/y±)
  bracket the oblique ones. A camera pitch error moves a top-down view's table sideways
  only (no height change); a range error moves it in height only. A hand-eye translation
  moves every view by the same camera-frame vector regardless of range.

## Hypotheses

Each states its prediction on the instruments and what refutes it.

**H1 — range-dependent depth bias of the D405** (scale `r' = s·r`, or a disparity offset
`Δr = c·r²`).
- Predicts: `h` grows with the camera's height above the slab for top-down views and with
  range along the ray for oblique ones, with ONE `s` (or `c`) for all views and runs;
  `v_i` in the camera frame points along ±z_cam with |v| ∝ range; close face views
  (0.10–0.15 m) carry ≤ 2 mm; NE1's two families agree.
- Refuted if the top-down views at 0.23–0.26 m show a different `h` than the oblique views
  at the same range, if `h` does not grow with range, or if the lateral camera-frame
  components of `v_i` are as large as the along-ray ones in the far views.
- Preliminary (this morning): `h` ≈ 10–12 mm at 0.25–0.35 m, 20–25 mm at 0.5 m, i.e.
  a ~4–5 % shortfall; consistent with the survey being the worst view everywhere.

**H2 — hand-eye error** (`T_flange_cam` off by δt, δθ).
- Predicts: the tilt `τ` is the same vector in the camera frame for every view (δθ);
  `v_i` share one camera-frame constant (δt) independent of range and of the joint
  family; one 6-DoF correction to `T_flange_cam` lowers ALL pair offsets, in every run.
- Refuted if `τ` in the camera frame scatters more than its mean, or if the correction
  that fixes the close views worsens the far ones (then the far-view error is range, not
  pose).

**H3 — depth-to-colour alignment / intrinsics** (distortion ignored on `depth_aligned`).
- Predicts: I4 differs by more than 2 mm in `h`/`τ`, and the plane residual is radial in
  the image.
- Refuted if I4 agrees within 1–2 mm (bounds H3 at that level; the memory already says
  2.6 mm).

**H4 — pose/frame timing.**
- Predicts: `v_i` correlates with the direction the arm came from.
- Refuted if no correlation (expected: frame before joints, 0.2–0.6 s after arrival, arm
  stopped).

**H5 — FK error from nominal DH** (not in the brief; added because the hand-eye absorbs
whatever FK error is constant over the calibration family, and views in the other family
expose it).
- Predicts: NE1's families differ in `h`/`τ` at the identical survey pose; after H1+H2 are
  removed, the remaining `v_i` cluster by family.
- Refuted if the families agree within 3 mm at the survey pose (preliminary: A 24.5 ± 3,
  B 22.1 ± 1.7 mm — ≤ 3 mm, so H5 is at most a secondary term).

The hypotheses are not exclusive. The expected answer is a mixture: a range term (H1) that
dominates the far views and a rotation/translation term (H2, maybe H5) that dominates the
close ones. The fit below separates them.

## What we do

1. `table.py` — I1 and I4 on every captured step of every 2209 run (≈90 views). Writes
   `evidence/table.json` (per view: `h`, `τ` base and camera frame, inlier count, range at
   the hit, residual-vs-range bins, I4 difference) and `evidence/table_profile.png`
   (`Δr` vs `r`, pooled, coloured by elevation; top-down views marked).
2. `pairs.py` — I2 on the runs with features (box4, bigcam5, smallcam5, compute5/11/13/7,
   plus whichever of the others yield ≥ 15 clean matches). Writes `evidence/pairs.json`
   (per pair: mean vector, consistency, epipolar misfit; per view: `v_i` in base and
   camera frame, joint family).
3. `fit.py` — the models on I1 + I2 together, each as a least-squares problem over all
   views of all runs: (a) H1 alone (`s`, and separately `c`), (b) H2 alone (δt, δθ),
   (c) H1 + H2, (d) H1 + H2 + per-family constant (H5 proxy). Reports the per-view residual
   `h` and the per-pair residual offset before and after each model; leave-one-run-out:
   fit on all runs but one, score the held-out run's pairs. Applies the winning correction
   to the named pairs: compute demo5 steps 2/6/7, box demo1 steps 1/4. Writes
   `evidence/fit.json`, `evidence/residuals.png`, `evidence/named_pairs.json`.
4. `numbers.py` — the tables for the report from the three JSON files.

No recording is written to or copied. Heavy intermediates (lifted points, kept matches)
go to `inspection/data/iwm/e06_view_offset/`.

## What we look at

- The `Δr(r)` cloud: a line through zero (scale), a parabola (disparity offset), a flat
  band (translation), or per-view bands (pose).
- `τ` in the camera frame: one blob (hand-eye rotation) or scatter.
- NE1: survey `h`/`τ` family A vs B.
- The residual table per view and per named pair after each model.

## What would count as a no

- No single model (nor H1 + H2) takes the per-view residual below ~3 mm on most views
  while a per-view free translation does: the error is per view and belongs in the IWM.
- The leave-one-run-out residual is much worse than the in-sample one: we fitted noise.

## Done when

`fit.json` names the model, its parameters, the per-view residual before/after on all
runs, the named pairs' offsets before/after, and the REPORT says whether the correction is
one-off (camera depth calibration and/or hand-eye re-solve, and what to do on the robot)
or per view.
