---
name: walkmatch-actuator-id-harness
description: Sim-to-sim system-ID toolkit at logic/simulation/walkmatch/ for any Isaac↔MuJoCo fidelity question — pinned-base actuator step-response ID, plus the foot-trace + gain-scale instrumentation built to debug the Isaac walk.
metadata:
  type: reference
---

Built 2026-06-25 (MAY-147) to find why Oli walks in MuJoCo but not Isaac. Reuse whenever you
need to decide "is an Isaac↔MuJoCo discrepancy the actuator, the body, or the contact."

**`logic/simulation/walkmatch/`** (the lab notebook `NOTEBOOK.md` lives here too):
- `spec.py` — shared, pure-numpy (imports in BOTH `isaac` py3.11 and `limx` py3.8 envs):
  PR order, walk_param kp/kd/default, the open-loop step-reference protocol, leg-only gain
  mask (the bare serial ankle is unstable under explicit gains, so the ID drives only hip/knee).
- `actuator_id_isaac.py` (isaac env) / `actuator_id_mujoco.py` (limx env) — pin the base,
  gravity off, drive an identical joint step reference, record q(t). MuJoCo runs the SERIAL
  model (achilles equality disabled) = the trusted reference. Sweeps {implicit,explicit}×{arm}.
- `compare.py` — overlays + scores (RMS) the Isaac configs vs MuJoCo, saves a PNG.
- `ankle_jacobian.py` (limx env, added 2026-07-01) — measures the achilles Jacobian
  `G=∂(pitch,roll)/∂(A,B)` in MuJoCo and the effective PR-space stiffness `K=kp·J^T J`; `--sweep`
  scans the range. Answered "does the twisted ankle have pitch↔roll coupling?" → NO (≈0 everywhere
  but the singular joint limit). Reuse for any "is there ankle coupling / what's the effective
  joint stiffness" question. See NOTEBOOK F11/F12 + [[isaac-walk-physics-fidelity]] final verdict.
- Result that mattered: Isaac legs match MuJoCo to RMS 0.0004 rad → leg actuator faithful.

**Instrumentation added to the production sim (env-gated / opt-in, defaults unchanged):**
- `OLI_FOOT_TRACE=/path.jsonl` on `sim_world_main` → per-command world positions of the 6
  `contact_foot_{heel,center,tip}_{L,R}` rigid bodies + base. Distinguishes slip (stance x
  drifts) vs launch (both feet z>0) vs roll. Read foot links via `RigidPrim([paths])`.
- `OLI_TRACE=/path.jsonl` (already in `logic/oli/runtime.py`, the BRAIN) → q/dq/qdes/gyro/quat
  per policy step. Same brain drives Isaac AND MuJoCo, so it taps BOTH for command-vs-realized
  diffing. Capped 400.
- `sim_world_main --ankle-kp-scale / --ankle-roll-scale / --waist-kp-scale` → multiply the
  achilles-parallel joints' kp+kd (recover joint-space stiffness the dual-motor linkage gives).
- `/tmp/gear.py` pattern — measure the achilles gear ratio in MuJoCo: drive A/B motors, read
  the serial ankle through the loop. (For HU_D04_01: additive A+B → ROLL r≈0.99, differential
  A−B → PITCH r≈0.93; linkage ~1:1, authority is the dual motors.)

Lesson reinforced: separate the failure axes with measurement before tuning — pinned-base
actuator ID exonerated the legs and isolated the ankle in ~2 Isaac boots, turning "watch it
fall" into quantitative comparisons. Related: [[isaac-walk-physics-fidelity]],
[[isaac_pd_implicit_drive]], [[walk_policy_obs_builder_fidelity]], [[feedback_tests_in_repo_tdd]].
