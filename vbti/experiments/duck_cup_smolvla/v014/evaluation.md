# v013 vs v014 — Detection State A/B Evaluation

The clean A/B for the detection-augmentation Test 1: same SmolVLA, same dataset base, same hyperparams. Only the input state changes. v013 is plain joint state; v014 adds 16 detection coords from the distilled student detector. Goal is to see whether explicit `(cx, cy)` per (camera, object) makes the policy's actions sharper, or doesn't.

## Setup

| | v013 | v014 |
|---|---|---|
| Dataset | `eternalmay33/01_02_03_merged_may-sim` | same base + detection columns (`_detection` variant) |
| State dim | **6** (joints in degrees) | **22** (6 joints + 16 detection cx/cy) |
| Detection source | none | StudentDetector (m1_baseline, MobileNetV3-Small, per-camera) |
| Conf-hold | n/a | `apply_confidence_hold`, threshold 0.08 |
| Model | SmolVLA | SmolVLA |
| Training | 50k steps, batch 32, lr 1e-4, cosine schedule | identical |

The 16 detection dims in v014's state are organized as `[left, right, top, gripper] × [duck, cup] × [cx, cy]`, normalized to `[0, 1]`. When the detector returns `conf < 0.08` during dataset prep, the cx/cy slots hold the previous good frame's values — same hold strategy used in inference.

No phase labels, no masking, no action noise. One thing changed at a time.

## Eval pipeline

The eval engine is structured. A protocol file defines duck/cup positions per trial, the operator places objects, presses `s` for success or `f` for failure, the engine logs per-trial outcomes plus checkpoint, action horizon, and protocol metadata into a session JSON. Same pipeline runs against any checkpoint of any version — that's what makes the A/B clean.

Two protocols used:

- **`checkpoint_sweep_no_ood`** — 20 ID trials, used for the per-checkpoint sweep on v013
- **`id_scale_60`** — 60 ID trials, halton-spread duck positions across 6 cup groups, deterministic seed=13. Built later for finer-grained measurement (±20pp CI per cup group → ±13pp).

Action horizon was fixed at 10 across all runs (after fixing an earlier inference bug that was running ah=1 silently).

## v013 — Checkpoint sweep

First run, `checkpoint_sweep_no_ood`, 20 trials each:

| checkpoint | success | rate |
|---|---|---:|
| 015000 | 13/20 | 65% |
| **020000** | **16/20** | **80%** |
| 025000 | 16/20 | 80% |
| 030000 | 13/20 | 65% |
| 035000 | 12/20 | 60% |

Best at chkpt 20k and 25k, both 80%. Re-ran the next day on the same protocol to check variance — 20k held at 75% (15/20), 25k dropped to 60% (12/20). Across the two runs, **20k averages 31/40 = 77.5%**, the most stable checkpoint and the one used for the A/B comparison and the `id_scale_60` protocol.

### v013 success and failure modes

Sample from `id_scale_60` on chkpt 20k (60 trials, halton-spread):

![[vbti/experiments/duck_cup_smolvla/v013/eval_sessions/chkpt_step_020000_ah_10_pr_id_scale_60_20260429_144553/videos/trial_39_success.mp4]]
_v013, trial 39 — clean approach, grasp, drop into cup. ID conditions._

![[vbti/experiments/duck_cup_smolvla/v013/eval_sessions/chkpt_step_020000_ah_10_pr_id_scale_60_20260429_144553/videos/trial_45_failure.mp4]]
_v013, trial 45 — typical failure mode: imprecise grasp, gripper closes near the duck but slightly off, the duck slips out._

The failure mode is exactly the precision problem that motivated detection state in the first place — the gripper gets close but not close enough, closes a centimeter off, duck doesn't lock in. If detection coords help precision, v014 should narrow that gap.

## v014 — Real-robot outcome

**v014 fails on the real robot.** With perfectly aligned cameras, identical training to v013, and a working student detector firing live, the policy jitters and drifts toward the mean action even at 50k steps. Approach is uneven, grasp doesn't lock in, the trajectory looks like the policy is reading a slightly-wrong signal and trying to compensate.

Before getting to the failure analysis, a few inference-pipeline gotchas were necessary just to get v014 running without crashing:

- **RGB vs BGR.** StudentDetector runs on RGB frames from the capture loop. There was a stray `cv2.cvtColor(BGR2RGB)` doing a channel-swap on already-RGB inputs. Fixed.
- **State padding.** Feed the full 22-d state to the preprocessor. The `config.json` reports `[6]` but `max_state_dim=32` pads internally — preprocessor expects 22.
- **`DetectionStateHolder`.** Live inference must mirror `apply_confidence_hold` exactly: when conf < threshold, re-emit the last known cx/cy, never zeros. Reset between trials.
- **Camera order is canonical.** The 16-d aug vector follows `[left, right, top, gripper]`, NOT user-config order. Re-ordering silently mis-aligns the normalizer's mean/std subtraction.
- **`--detection=true`** in `run_real_inference` and `eval_engine` handles both state augmentation and the bbox/crosshair overlay.

All of those were prerequisites to running v014 at all — necessary but not sufficient. The real failure mode is below.

## Why v014 failed — sensitivity ablation

Ran a controlled input-sensitivity ablation on the v014 chkpt 20k checkpoint to figure out what the policy is actually using. The result is sharp.

### TL;DR

Removing the 16 detection coords drives MAE from **1.23° (control) to 9.71°**. Removing all four camera images only goes to **3.46°**. v014 leans on detections, not pixels — **2.8× more sensitive** to losing detections than to losing all four cameras combined.

### Method

- Checkpoint: `lerobot_output_r1/checkpoints/020000/pretrained_model`
- Dataset: `eternalmay33/01_02_03_merged_may-sim_detection` (135,846 frames)
- N = 30 samples, evenly spaced indices in `[0, 135845]`, seed=0
- Action mean (deg): `[0.66, -42.7, 30.36, 56.6, 10.36, 17.18]`
- Mean GT L2 distance from action_mean = **74.00°**
- `fallback_ratio = mean(||pred − action_mean||) / mean(||GT − action_mean||)`
  - ~0.0 → prediction sits on the action mean (full collapse)
  - ~1.0 → prediction is as far from mean as GT is (right scale)
- Single-frame eval: `policy.reset()` + `predict_action_chunk()`, take `chunk[0]`
- Image normalizer is identity → "zero an image" = literal zeros tensor

### Results

| condition | mae (deg) | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | fallback_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1.23 | 0.94 | 1.74 | 2.35 | 0.92 | 0.68 | 0.78 | 1.00 |
| **no_detection** | **9.71** | 5.81 | 17.24 | 17.90 | 6.19 | 6.16 | 4.97 | **0.63** |
| no_joints | 31.20 | 9.32 | 55.75 | 47.55 | 48.73 | 10.83 | 15.00 | 1.00 |
| no_state | 23.46 | 9.61 | 44.36 | 42.21 | 26.30 | 7.79 | 10.51 | 0.57 |
| **no_images** | **3.46** | 1.11 | 6.18 | 7.52 | 2.19 | 1.03 | 2.70 | **0.94** |
| no_images_no_detection | 13.09 | 5.82 | 23.61 | 25.36 | 7.61 | 6.06 | 10.05 | 0.51 |
| drop_top | 1.39 | 0.84 | 2.24 | 2.27 | 1.20 | 0.84 | 0.93 | 0.99 |
| drop_left | 1.33 | 0.81 | 1.97 | 2.68 | 0.91 | 0.78 | 0.81 | 0.99 |
| drop_right | 1.50 | 0.84 | 2.22 | 2.63 | 1.38 | 0.88 | 1.02 | 0.98 |
| drop_gripper | 1.87 | 0.90 | 2.73 | 3.86 | 1.40 | 0.80 | 1.55 | 0.98 |
| det_small_noise | 2.10 | 1.31 | 3.17 | 3.70 | 1.19 | 1.24 | 2.00 | 1.00 |
| det_big_noise | 5.09 | 4.00 | 7.98 | 8.16 | 3.95 | 2.79 | 3.66 | 0.86 |
| det_shuffled | 5.89 | 2.80 | 12.03 | 11.51 | 4.23 | 2.09 | 2.66 | 0.81 |
| wrong_task | 5.44 | 2.16 | 10.11 | 10.54 | 4.37 | 2.42 | 3.03 | 0.79 |

### Plots

![[sensitivity_plots/mae_bars.png]]
_Per-condition MAE. `no_detection` (9.71°) is far worse than `no_images` (3.46°) — removing the 16 scalar detection inputs hurts more than blanking all four 3-channel image streams._

![[sensitivity_plots/per_joint_heatmap.png]]
_Per-joint MAE breakdown. Detection loss concentrates damage on shoulder_lift and elbow_flex — the joints with the largest task-relevant range._

![[sensitivity_plots/scatter_pred_vs_gt.png]]
_Pred vs GT scatter for the control condition — predictions hug the diagonal, sanity floor confirmed._

### Reading the numbers

- **control** = 1.23° MAE, fb=1.00 — sanity floor on training-distribution data; predictions hug GT.
- **no_detection** = 9.71° (fb=0.63) vs **no_images** = 3.46° (fb=0.94): zeroing 16 detection scalars is **2.8× more damaging** than blanking all four 3-channel image streams. The fb gap (0.63 vs 0.94) confirms `no_detection` pulls predictions toward the action mean while `no_images` does not.
- **det_shuffled** = 5.89° (fb=0.81) — preserving detection magnitudes but breaking the spatial cx/cy layout is *more* harmful than `det_big_noise` (5.09°) and only ~40% less harmful than zeroing detection (9.71°). The policy uses the *spatial* meaning of (cx, cy), not just the value distribution.
- **det_small_noise** = 2.10° vs **det_big_noise** = 5.09° — smooth degradation as detection drifts; even σ=0.05 (~5% of normalized range) lifts MAE 1.7× over control.
- **drop_one_cam**: top=1.39, left=1.33, right=1.50, gripper=1.87 (fb≈0.98–0.99). Per-camera deltas vs control ≤0.6°. **No single camera is load-bearing** — even dropping the wrist camera barely moves the policy.
- **no_images_no_detection** = 13.09° — joints+task only, lower bound. `no_detection` (9.71°) sits 3.4° below this, so once detection is removed, the four images recover only ~3.4° of the ~7.6° gap from control to floor. Images carry modest residual signal but are clearly underused.
- **no_joints** = 31.20° (fb=1.00) — zeroing joints is the single most damaging input ablation, but fb≈1 means "wrong direction" not "collapse to mean". Joints are the dominant proprioceptive anchor.
- **wrong_task** = 5.44° (fb=0.79) — task token matters but isn't load-bearing the way detection is.

### Conclusion — easy-signal trap confirmed

- Detection-only removal (9.71°, fb=0.63) is **2.8× more damaging** than removing all four image streams (3.46°, fb=0.94), and **7.9× control** (1.23°). Pixels are nearly free vs detection inputs.
- Of the ~7.6° MAE gap between control and the joints-only floor, **6.3° is recovered by detection alone** vs only **3.4° by images alone**. Detection contributes ~2× the visual signal that images do.
- `det_shuffled` and `det_big_noise` confirm the policy uses *spatial* (cx, cy) structure — exactly what an easy-signal learner does.
- Per-joint damage from detection loss concentrates in shoulder_lift (17.2°) and elbow_flex (17.9°) — the joints with the largest task-relevant range.
- **Practical implication**: any drift in the live detection distribution (camera angle, lighting, MobileNetV3 student-detector failure on real robot) pushes `state[6:22]` off the training manifold; the underused image stream cannot compensate. This matches the observed real-robot failure mode: jitter + drift toward mean action.

## Decision

The detection-coords-as-state path doesn't help — and forcing it via training-time fixes may be a dead end. Three threads forward:

- **Drop detection from state** _(preferred)_. v013 already shows that with a sufficiently diverse dataset, SmolVLA's frozen ViT extracts spatial understanding from images on its own. Detection-state may simply not be necessary.
- **Detection dropout** _(fallback)_. Retrain v014 with `state[6:22]` randomly zeroed 30–50% of training steps to force image utilization while keeping detection available at inference.
- **Depth channel** _(next axis to explore)_. Add a depth stream to the image inputs to give the encoder explicit 3D cues — could lift performance beyond what either v013 or detection-augmented v014 reach.

For the next experiment we go with option 1: drop detection from state, build out the dataset further, and let the frozen ViT do its job.
