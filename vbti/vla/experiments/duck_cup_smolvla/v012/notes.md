# v012 — Delta Action Prediction

**Hypothesis**: Step-wise delta actions (`action[t] - state[t]`) improve out-of-distribution spatial generalization compared to absolute joint positions (v010).

**What changed from v010**:
- Dataset: `eternalmay33/01_02_09_00_merged_delta` (step-wise delta, joints 0-4 relative, gripper absolute)
- Everything else identical: same hyperparameters, same architecture, same cameras

**Inference**: requires `--delta_actions` flag on `run_real_inference.py`

**Baseline**: v011 (`eternalmay33/01_02_09_00_merged`, absolute actions)
