# v017 — RGB-only baseline on new depth-collected datasets

## Status
Configured. Awaiting:
1. Aggregate `eternalmay33/04_05_06_07_merged_may-sim_depth` from sources 04/05/06/07.
2. Push merged dataset to remote (5090 training rig).

## Hypothesis
SmolVLA fine-tuned on the **same** 4 newly-collected depth-recorded teleop datasets but **without** the gripper depth feature is the matched control for v018. Same data, same hyperparams, same merge — only difference is the camera list.

## A/B against v018
- **v017** — 4 cams: `[top, left, right, gripper]` (RGB only)
- **v018** — 5 cams: `[top, left, right, gripper, gripper_depth]` (+ packed-depth turbo render)

Anything beyond that — model code, optimizer, schedule, data — must stay byte-identical.

## Dataset
| source | episodes | frames | scene |
|---|---:|---:|---|
| 04_red_cup_depth | 64 | 24 813 | red cup, varied bg |
| 05_black_cup_depth | 51 | 20 072 | black cup, varied bg |
| 06_black_cup_red_bg_depth | 49 | 21 136 | black cup, red bg |
| 07_red_cup_black_bg_depth | 11 | 4 363 | red cup, black bg |
| **merged** | **175** | **70 384** | duck-into-cup, mixed bg/cup |

Aggregated once as `eternalmay33/04_05_06_07_merged_may-sim_depth` (full, with depth), then **stripped** of `observation.images.gripper_depth` via `vbti.logic.dataset.strip_feature` to produce `eternalmay33/04_05_06_07_merged_may-sim` (2.3 GB; depth-included version is 11 GB). v017 trains on the stripped twin, v018 trains on the full version. Both share byte-identical RGB videos, identical episode/frame indexing, identical RGB stats — only difference is whether `gripper_depth` exists.

## Hyperparams (= v013, = v014)
- 50 000 steps, batch 32, lr 1e-4, cosine decay → 2.5e-6, warmup 1000.
- bf16, num_workers 8.
- Frozen SigLIP, expert-only training, state proj trainable.
- ImageNet stats normalization for vision inputs.

## What we expect to see
- Baseline matches v013-style real-robot behavior on the new task variants (red cup, black cup, mixed bg).
- v018 should beat v017 *if* depth carries useful 3D signal that the frozen SigLIP can absorb. Weak hypothesis — SigLIP was pretrained on RGB and the gripper depth is rendered as turbo RGB, so the encoder is being asked to interpret a colormap as spatial structure.

## Outputs
- `lerobot_output_r1/` — training run (matches v014 layout)
- `evaluation.md` — real-robot eval
- Compare with `v018/evaluation.md` after both finish.
