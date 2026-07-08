---
name: isaac-oli-stand-spawn-height
description: In the Isaac sim_world, --spawn-height 1.1 is the tuned base-z where Oli spawns into the crouch, settles, and stands still under the walk policy at zero command; lower (e.g. 0.95) collapses ~1s after the walk handoff.
metadata:
  type: reference
---

`sim_world_main.py --spawn-height 1.1` (HU_D04_01) is the empirically tuned base-z at
which Oli stands still. Bring-up: spawn into the `default_angle` crouch (== stand_pos;
the policy's nominal pose, see [[walk_policy_obs_builder_fidelity]]) via
`Oli.set_joint_state` (SimComm permutes PR→Isaac), a short stiff settle so the feet
plant + the IMU populates, then the **WALK ONNX run at zero velocity command** holds it
upright. Standing = walk@[0,0,0] (Anton's call) — the analytic position-hold
`StandPolicy` cannot balance a free base.

Found 2026-06-24 (MAY-147), Anton watching the Isaac viewport: at 1.1 the crouch
**settles stably** (~z=0.89) during the stiff settle. Lower spawns (0.95) collapse even
during/just after settle. So 1.1 is the right spawn/settle height — keep it.

**CORRECTION (2026-06-25):** the walk policy does NOT actually hold a stable stand here.
Under the walk ONNX (zero cmd or fwd), Oli topples at ~1.0–2.5 s regardless of spawn
height, loop pacing, or lock-step (latency falsified — see below). The earlier
"stands still" read was a brief windowed-run impression, not robustness. Root-cause
hunt moved to **World-side physics fidelity** — leading suspect is armature=0 in the
USD; see [[isaac_walk_physics_fidelity]].

Related: [[isaac-pd-implicit-drive]], [[walk_policy_obs_builder_fidelity]],
[[project_invariant_oli_interface]], [[isaac_oli_smoke_loader]].
