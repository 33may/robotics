# v014 — Student Detection Coordinates

**Hypothesis**: Adding student detector coordinates (duck/cup cx/cy/conf from 4 cameras = 24d state) as extra state improves grasping accuracy over pure vision.

**What changed from v013**:
- Dataset: `eternalmay33/01_02_03_merged_may-sim_detection` (detection parquet merged in, 30d state: 6 joints + 24 detection values)
- Detection: StudentDetector (MobileNetV3-Small, m1_baseline, per-camera)

**A/B pair**: v013 (no detection) vs v014 (student detection). Same model, same hyperparams, same dataset base.

**Key question**: Does student detector coordinates improve policy over pure vision baseline (v013)?

**Inference**: requires live StudentDetector (m1_baseline) at inference time
