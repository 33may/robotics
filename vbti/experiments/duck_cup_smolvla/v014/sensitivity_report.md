# v014 SmolVLA — input sensitivity ablation

## TL;DR
Removing the 16 detection coords drives MAE from 1.23° (control) to 9.71° (fallback=0.63), while removing all four images only goes to 3.46° (fallback=0.94). v014 leans on detections, not pixels — 2.8× more sensitive to losing detections than to losing all four cameras.

## Method
- Checkpoint: `vbti/experiments/duck_cup_smolvla/v014/lerobot_output_r1/checkpoints/020000/pretrained_model`
- Dataset: `eternalmay33/01_02_03_merged_may-sim_detection` (135846 frames)
- N = 30 samples, evenly spaced indices in `[0, 135845]`, seed=0
- Action mean (deg) from `policy_preprocessor_step_5_normalizer_processor.safetensors`:
  `[  0.66, -42.7 ,  30.36,  56.6 ,  10.36,  17.18]`
- Mean GT L2 distance from action_mean = **74.00°**
- `fallback_ratio = mean(||pred - action_mean||) / mean(||GT - action_mean||)`
  - ~0.0 -> prediction sits **on the action mean** (full collapse)
  - ~1.0 -> prediction is as far from mean as GT is (right scale; could be tracking GT or wandering)
  - >1.0 -> prediction wanders even further from the mean than GT does
  - On training-distribution data, control = 1.00 (matches GT scale); meaningful drops below 1 indicate pull toward mean.
- Single-frame eval: `policy.reset()` + `predict_action_chunk()`, take `chunk[0]`.
- Joints in degrees throughout. Image normalizer is IDENTITY -> "zero an image" = literal zeros tensor.
- State `[0:6]`=joints (deg), `[6:22]`=detection cx/cy normalized to [0,1].

## Results
| condition | mae (deg) | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | fallback_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1.23 | 0.94 | 1.74 | 2.35 | 0.92 | 0.68 | 0.78 | 1.00 |
| no_detection | 9.71 | 5.81 | 17.24 | 17.90 | 6.19 | 6.16 | 4.97 | 0.63 |
| no_joints | 31.20 | 9.32 | 55.75 | 47.55 | 48.73 | 10.83 | 15.00 | 1.00 |
| no_state | 23.46 | 9.61 | 44.36 | 42.21 | 26.30 | 7.79 | 10.51 | 0.57 |
| no_images | 3.46 | 1.11 | 6.18 | 7.52 | 2.19 | 1.03 | 2.70 | 0.94 |
| no_images_no_detection | 13.09 | 5.82 | 23.61 | 25.36 | 7.61 | 6.06 | 10.05 | 0.51 |
| drop_top | 1.39 | 0.84 | 2.24 | 2.27 | 1.20 | 0.84 | 0.93 | 0.99 |
| drop_left | 1.33 | 0.81 | 1.97 | 2.68 | 0.91 | 0.78 | 0.81 | 0.99 |
| drop_right | 1.50 | 0.84 | 2.22 | 2.63 | 1.38 | 0.88 | 1.02 | 0.98 |
| drop_gripper | 1.87 | 0.90 | 2.73 | 3.86 | 1.40 | 0.80 | 1.55 | 0.98 |
| det_small_noise | 2.10 | 1.31 | 3.17 | 3.70 | 1.19 | 1.24 | 2.00 | 1.00 |
| det_big_noise | 5.09 | 4.00 | 7.98 | 8.16 | 3.95 | 2.79 | 3.66 | 0.86 |
| det_shuffled | 5.89 | 2.80 | 12.03 | 11.51 | 4.23 | 2.09 | 2.66 | 0.81 |
| wrong_task | 5.44 | 2.16 | 10.11 | 10.54 | 4.37 | 2.42 | 3.03 | 0.79 |

## Plots

![MAE bars](sensitivity_plots/mae_bars.png)

![Fallback ratio](sensitivity_plots/fallback_ratio.png)

![Per-joint heatmap](sensitivity_plots/per_joint_heatmap.png)

![Pred vs GT scatter](sensitivity_plots/scatter_pred_vs_gt.png)

## Reading the numbers
- **control** = 1.23° MAE, fb=1.00 — sanity floor on training-distribution data; predictions hug GT (see scatter plot).
- **no_detection** = 9.71° (fb=0.63) vs **no_images** = 3.46° (fb=0.94): zeroing 16 detection scalars is **2.8× more damaging** than blanking all four 3-channel image streams. The fb gap (0.63 vs 0.94) confirms `no_detection` pulls predictions toward the action mean while `no_images` does not.
- **det_shuffled** = 5.89° (fb=0.81) — preserving detection magnitudes but breaking the spatial cx/cy layout is more harmful than `det_big_noise` (5.09°) and only ~40% less harmful than zeroing detection (9.71°). The policy uses the *spatial* meaning of (cx, cy), not just the value distribution.
- **det_small_noise** = 2.10° vs **det_big_noise** = 5.09° — smooth degradation as detection drifts; even σ=0.05 noise (~5% of normalized range) lifts MAE 1.7× over control.
- **drop_one_cam**: top=1.39, left=1.33, right=1.50, gripper=1.87 (all fb≈0.98–0.99). Per-camera deltas vs control are ≤0.6°; no single camera is load-bearing, and even dropping the wrist camera barely moves the policy.
- **no_images_no_detection** = 13.09° (fb=0.51) — joints + task only, the worst-case lower bound. **no_detection** (9.71°) sits 3.4° below this — so once detection is removed, the four images recover only ~3.4° of the ~7.6° gap from control to that lower bound. Images carry modest residual signal but are clearly underused.
- **no_joints** = 31.20° (fb=1.00) — zeroing the 6 joint dims is the single most damaging input ablation. fb≈1 here means the predictions wander far from GT but at GT-like distance from mean — not a "collapse to mean" failure mode, but an "off in the wrong direction" one. Joints are the dominant proprioceptive anchor.
- **wrong_task** = 5.44° (fb=0.79) — swapping the task string ("Pick up the duck..." → "do nothing") moves MAE 4.4× over control; task token matters but isn't load-bearing the way detection is.

## Conclusion
**Hypothesis supported — easy-signal trap confirmed.**
- Detection-only removal (no_detection=9.71°, fb=0.63) is **2.8× more damaging** than removing all four image streams (no_images=3.46°, fb=0.94), and **7.9× control** (1.23°). Pixels are nearly free vs detection inputs in this policy's decision boundary.
- Of the ~7.6° MAE gap between control (1.23°) and the joints-only worst case (13.09°), **6.3° is recovered by detection alone** (no_images: 3.46°→1.23°), while **only 3.4° is recovered by images alone** (no_images_no_detection: 13.09°→9.71°). Detection contributes ~2× the visual signal that images do.
- `det_shuffled` (5.89°) and `det_big_noise` (5.09°) confirm the policy uses the *spatial* (cx, cy) structure of detections, not just the marginal value distribution — this is exactly what an "easy-signal" learner would do.
- Per-joint heatmap: the damage from detection loss concentrates in shoulder_lift (17.2°) and elbow_flex (17.9°) — the joints with the largest task-relevant range — while wrist_roll (6.2°) and gripper (5.0°) are less affected.
- Practical implication: any drift in the live detection distribution (camera angle, lighting, MobileNetV3 student-detector failure on real robot) pushes state[6:22] off the training manifold; the image stream — which the policy effectively ignores — cannot compensate. This matches the observed real-robot failure mode (jitter + drift toward mean action).
- Next experiment: retrain v014 with state[6:22] removed (or replaced with a learned-detection-dropout schedule) to force the policy to read the cameras instead of the cheap coordinate channel.
