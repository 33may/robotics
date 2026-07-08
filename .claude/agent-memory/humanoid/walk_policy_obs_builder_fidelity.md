---
name: walk-policy-obs-builder-fidelity
description: HU_D04_01 walk policy obs builder decoded — exact 102→510 layout, scales, projected-gravity identity, the last_actions aliasing quirk, head-order swap; what the walk policy.onnx actually expects.
metadata:
  type: reference
---

The LimX walk policy for HU_D04_01 — source `vendor/humanoid-rl-deploy-python/controllers/HU_D04_01/walk_controller/{walk_controller.py,walk_param.yaml}`, artifact `policy/default/policy.onnx` (obs[1,510] → actions[1,31]). The exact obs the ONNX expects, decoded for the Oli brain (MAY-147, replicated in `logic/oli/action/policy_runner.py:encode_walk_obs`).

**Obs vector (102 = 3+3+3+31+31+31); gait terms are COMMENTED OUT — do not build them:**
`[ base_ang_vel·0.25 | projected_gravity | commands[v_x,v_y,w_z] (UNSCALED) | (q−default_angle)·1.0 | dq·0.05 | last_actions ]`
History: 5-deep, **newest-first**; first obs replicated ×5 → 510. decimation=10 → pace the policy by Δ≥10 ms sim-stamp, not wall-clock.

**projected_gravity = Rᵀ·[0,0,-1]** with the quat reordered wxyz→xyzw (scipy is scalar-last). The deploy's quat→euler('zyx')→quat detour is the algebraic identity of this — verified equal to 1e-5.

**last_actions aliasing quirk (subtle, replicate it):** `walk_controller.py:282` `self.last_actions = self.actions` *aliases* the array, then the per-joint loop mutates it in place, so the `last_actions` fed into the next obs is the **torque-CLAMPED** action, not the raw/clipped ONNX output. The deployed policy walks on the clamped value → set `last_actions = clamped`.

**Resolution (per joint):** clip ONNX to ±100; torque-limit clamp `action ∈ [ q−def + (kd·dq − τlim·0.95)/kp , q−def + (kd·dq + τlim·0.95)/kp ] / action_scale` (bounds |τ| ≤ 0.95·τlim); then `q_des = action·action_scale + default`; dq_des=0, tau_ff=0, Kp/Kd constant from walk_param.yaml.

**Head-order swap:** walk_param.yaml lists head_pitch(15) then head_yaw(16); our `PR_ORDER` is head_yaw(15), head_pitch(16). Numerically harmless — both heads share identical default(0)/kp(15.12)/kd(1.93)/action_scale(0.3141)/τlim(19). Use the yaml params positionally with PR-ordered obs; fix systematically later (design OQ).

**STAND is analytic** (separate `stand_controller.py`), NOT an ONNX. Related: [[project-invariant-oli-interface]], [[isaac-pd-implicit-drive]], [[walk-onnx-io-shape]].
