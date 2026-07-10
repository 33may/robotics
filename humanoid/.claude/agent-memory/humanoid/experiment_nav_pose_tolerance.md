---
name: experiment-nav-pose-tolerance
description: Closed-loop nav-stack tolerance budget (2026-07-10 experiments) — constant bias is the binding constraint; a fast odometry layer makes reloc-rate cheap
metadata:
  type: project
---

Measured on the REAL unmodified nav stack (`OccupancyGrid` → A* → pure-pursuit) in a deterministic
kinematic closed loop. Harness + raw results + plots at
`docs/research/localization/experiments/` (branch `33may/nav-localization-research`). Reproduced
fresh cell-for-cell by an independent verifier agent at 24 trials/cell.

- **E1 (accuracy):** position Gaussian noise is nearly free — 100% success to 0.1 m σ, still ≥0.92 at
  0.4 m σ. Yaw robust — ~0.83 at 20° σ. **Constant position BIAS is the binding constraint** — fine to
  0.1 m, then collapses to **0.42 success at 0.2 m**.
- **E2 (rate):** `hold-last-fix` (no odometry) cliffs past ~1 Hz (0.5 Hz → 0.21 success / 58% collision;
  0.2 Hz → 0.08 / 92%). `dead-reckon` (perfect-odometry proxy) holds **100% down to 0.2 Hz**.

**Why:** this is the accuracy/rate SPEC for picking a localizer. A few-cm RGBD relocalizer sits
comfortably inside the accuracy budget; the real risk is **systematic bias** (moving-camera FK
extrinsics / time-sync), not sensor noise. And a slow/bursty relocalizer is acceptable **only** if a
fast odometry layer carries the pose between fixes.

**How to apply:** target steady-state < ~0.1–0.2 m position, < ~10° yaw, < ~0.1 m bias; treat the bias
spec as a calibration job, not an algorithm choice; mandate an odometry layer. Caveat: kinematic /
perfect-actuation model = an UPPER bound — real budgets are tighter (adds locomotion tracking error,
control latency). Links: [[research-nav-localization]], [[nav-brain-localizer-seam]].
