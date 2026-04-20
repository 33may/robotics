# v013 — Detection Coordinates (Clean Interpolated)

**Hypothesis**: Adding object detection coordinates (duck/cup cx/cy from left/right/top/gripper, minus top_duck which is unreliable) as extra state dimensions improves grasping accuracy.

**What changed from v012**:
- Dataset: `eternalmay33/01_02_03_merged_may-sim_detection` (20d state, was 6d)
- State layout: [6 joints] + [16 detection cx/cy: 4 cams x 2 objects]
- Detection: Grounding DINO (TRT, stride=3, linear interpolated) — dense, clean coordinates
- No phase labels (keeping it simple for A/B test)
- `train_state_proj: true`
- Same architecture, hyperparameters, cameras as v012

**A/B pair**: v013 (clean) vs v014 (hold-simulated). Same model, same hyperparams, different detection noise levels.

**Key question**: Does clean detection data give better policy than hold-simulated data that matches inference distribution?

**Baseline**: v012 (no detection, 6d state)

**Inference**: requires live detection (TRT async, ~103ms/cam, hold strategy between updates)
