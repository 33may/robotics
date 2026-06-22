---
name: oli-head-joint-order-reversed
description: Oli head joints in RobotState are head_yaw (index 15) then head_pitch (index 16), opposite of what LimX's parallel-joint doc lists
metadata:
  type: project
---

Empirically probed during MAY-142 wire-contract capture: the `motor_names` list from `RobotState` orders the head as `[15] head_yaw_joint`, `[16] head_pitch_joint`. LimX's `parallel_joint_mapping_en.md` lists them in the opposite order (pitch, yaw).

**Why:** Vendor doc mismatch. The PR-space joint indexing in the live SDK is the truth; the doc is out of date or written for a different variant.

**How to apply:** When writing anything that addresses head joints by index — Isaac port, controllers, eval scripts — use yaw=15, pitch=16. Treat any vendor-doc joint table with suspicion until re-probed; the probe script lives at `humanoid/logic/simulation/mujoco/probe_contract.py`.
