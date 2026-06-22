---
name: isaac-pd-implicit-drive
description: In Isaac Sim, realize a deploy controller's PD law via PhysX implicit drive (set_gains + position/velocity targets), NOT explicit set_joint_efforts — the latter rings unstably for MuJoCo/real-robot-tuned Kd.
metadata:
  type: reference
---

When applying an external PD controller's `τ = Kp(q_d−q) + Kd(dq_d−dq) + τ_ff` to an Isaac Sim articulation, **push Kp/Kd into the joint drive gains and command position+velocity targets** so PhysX integrates the PD implicitly. Add `τ_ff` separately via `set_joint_efforts` (it is additive on top of the drive). Do **not** compute `τ` yourself and apply it all through `set_joint_efforts`.

## Why

`set_joint_efforts` applies a pure external torque for one step — the `Kd·dq` damping term is computed from the *previous* step's velocity and applied with a one-step lag. For gains tuned to implicit integration (real motor firmware, MuJoCo's implicit damping solver), this explicit application violates the discrete stability bound `Kd·dt/I ≲ 2`. On Oli's HU_D04_01 the knee link has `izz ≈ 0.003`, so the deploy `Kd=17` at `dt=1ms` gives `17·0.001/0.003 ≈ 5.6` → a sustained velocity limit-cycle (measured: ±1.8 rad/s ringing that never decays).

The implicit PhysX drive integrates the *same* formula but applies damping against the velocity it actually sees during the step — stable, no lag. Switching to it on Oli: peak velocity decayed 12.5 → 0.54 rad/s (23×) and the pose held to 0.087 rad. This is also how IsaacLab's own actuator models work.

Critically: the gains come from LimX's controllers over the wire — we cannot retune them. The implicit drive lets us honor the exact gains the policy expects.

## How (isaacsim 5.0 / SingleArticulation)

```python
view = single_articulation._articulation_view   # isaacsim.core.prims Articulation
view.set_gains(kps=kp_isaac.reshape(1, -1), kds=kd_isaac.reshape(1, -1))  # only on change
view.set_joint_position_targets(q_d_isaac.reshape(1, -1))
view.set_joint_velocity_targets(dq_d_isaac.reshape(1, -1))
single_articulation.set_joint_efforts(tau_ff_isaac)   # additive feedforward
```

- Re-call `set_gains` only when Kp/Kd change (deploy controllers send constant gains per mode → rare). Targets + τ_ff every tick.
- The HU_D04_01 USD ships `PhysicsDriveAPI` on every joint with soft default gains; whatever you do, the drive is summed with any explicit effort. Embrace the drive rather than fighting it.

## Steady-state droop caveat

With finite Kp and no gravity feedforward, a held joint settles at `q ≈ q_d − τ_gravity/Kp` (e.g. ~0.12 rad droop on the knee at Kp=139). The shipped LimX controllers compensate via `τ_ff` + tuned per-joint gains. Faithful gravity holding/tracking is a policy-porting concern, not a sim-bridge concern.

## Effort API names (isaacsim 5.0 SingleArticulation)

`get_joint_positions()`, `get_joint_velocities()`, `get_measured_joint_efforts()` (read), `set_joint_efforts()` (write). The `_articulation_view` carries `set_gains` / `set_joint_position_targets` / `set_joint_velocity_targets` / `get_max_efforts` / `get_gains`.

Found during MAY-147 Phase 5 (Isaac limxsdk bridge). Related: [[isaac-oli-smoke-loader]], [[limx-sdk-role-gating]].
