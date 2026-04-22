# v013 — New Baseline (01_02_03 May-Sim)

**Hypothesis**: Fresh baseline on the latest merged dataset

**What changed from v012**:
- Dataset: `eternalmay33/01_02_03_merged_may-sim` (absolute actions, may-sim = calibration profile)
- Batch size: 32 (was 64)
- No detection state — standard 6d joint state

**A/B pair**: v013 (no detection) vs v014 (student detection). Same model, same hyperparams, same dataset base.

**Key question**: Does adding student detector coordinates improve policy performance over pure vision?
