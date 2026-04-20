# v014 — Detection Coordinates (Hold-Simulated, Variable Stride)

**Hypothesis**: Training with hold-simulated detection data (stride 5-20 frames, uniform random per episode) makes the policy robust to the noisy coordinates it will see at inference time, improving real-world performance vs clean data.

**What changed from v013**:
- Dataset: `eternalmay33/01_02_03_merged_may-sim_detection_hold` (20d state)
- Detection coordinates simulated with hold strategy: random stride 5-20 frames, forward-fill between arrivals
- This matches inference distribution: TRT async detection at ~103ms/cam, 4 cameras round-robin, ~12 frames between updates with variance

**Hold simulation stats** (vs dense ground truth):
- Mean difference: 12.0 px
- Median difference: 0.2 px
- P95 difference: 62.8 px

**A/B pair**: v013 (clean) vs v014 (hold-simulated). Same model, same hyperparams, different detection noise.

**Key question**: Does domain-matched noisy training beat clean training for real-world deployment?

**Baseline**: v013 (clean detection), v012 (no detection)

**Inference**: Same as v013 — live TRT async detection with hold strategy
