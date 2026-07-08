---
name: may172-glide-wiring
description: How glide is wired into the deployment-invariant loop — a parallel GLIDE_CMD message type, GlideAction swaps in, same Orchestrator; PolicyOut untouched.
metadata:
  type: project
---

**Decision (Anton, 2026-07-01, MAY-172):** the kinematic glide plugs into the SAME
Orchestrator loop as the walk flow — it is NOT a separate app and does NOT alter the walk
contract. The only per-mode differences: the Action forwards the velocity command instead
of running the walk ONNX, and the Comm sends whatever the Action returns.

**Shape (all built + TDD'd, brain side green):**
- New pure module `logic/oli/glide.py`: `GlideCmd(stamp_ns,v_x,v_y,w_z)` (the wire message),
  `GlideModel` (accel-limited kinematic integrator = the "fake physics"), `GlideAction`
  (`.step(policy_in)→GlideCmd`, a drop-in for `PolicyRunner`).
- Wire: `MsgType.GLIDE_CMD=3` + `pack/unpack_glide_cmd` in `comm/protocol.py`,
  `encode/decode_glide_cmd` in `comm/codec.py` — ADDITIVE (a 4th tag on the shared UDS
  SEQPACKET socket). `CMD`/`STATE_IMU`/`HELLO` frames are byte-identical (CI-pinned by
  `test_walk_frame_sizes_unchanged`).
- Loop generalization: `Comm.send(msg)` dispatches by type (`PolicyOut`→`write_policy_out`,
  `GlideCmd`→`write_glide_cmd`); `BrainComm.write_glide_cmd` added; `Orchestrator` calls
  `comm.send(action_out)` and guards its q_des obs-trace to walk-only.
- `brain_main.py --mode glide` selects `GlideAction`; joystick/Teleop/Comm unchanged.

**Why this shape:** Anton's constraint — "don't alter the real flow that works once we fix
the policy." The velocity must cross to the World and `PolicyOut` has no slot for it and we
won't add one → it MUST be a parallel message, not a mode field. The tagged protocol was
built for exactly this. "Same robot, one more interface": everything up to `Intent` is
shared; only the `Intent` consumer (Action) + the World's apply swap. Real walk drops in =
swap `GlideAction`→`PolicyRunner`, callers unchanged.

**World side (block 4) — BUILT, awaiting live Isaac verification:**
- `WorldComm.receive_glide_latest`/`receive_glide_blocking` (loopback-tested, brain env green).
- `logic/simulation/isaacsim/glide_world_main.py` — boots Isaac + Oli **free base**
  (`pin_root=False`), settles the crouch, serves, loops: publish Observation → receive
  GlideCmd → `GlideModel.step` → per-substep velocity-drive the base + hold legs → step.
- **Collision approach = velocity-driven (V), decided by Anton 2026-07-01** ("we'll need
  wall collision eventually, that's why we build it"): NOT a true kinematic body. Each
  substep sets the ROOT spatial velocity = body→world twist + P height-hold
  (`vz = height_kp·(z0−z)`) + upright lock (`angular=(0,0,wz)` zeroes roll/pitch rate); PhysX
  integrates position and resolves contacts, so walls block Oli (once MAY-171's mesh lands).
  Legs held in the stand crouch via implicit-drive PD (a visual step is a later pass).
- Run: World `conda run -n isaac python humanoid/logic/simulation/isaacsim/glide_world_main.py`;
  brain `... brain_main.py --mode glide --joystick socket|fixed --vx …`.

**LIVE-VERIFIED 2026-07-02 (headless, `--vx 0.3`):** glide works first try — settled stand
z=0.899, glides straight forward, z held dead-flat at 0.90, lateral drift ~2 cm over 3 s, no
topple, clean teardown. `--foot-clearance 0.0` (feet AT floor) showed no drag/jitter at
0.30 m/s → the hover hack looks unnecessary. Default `--height-kp 20`, `--hold-kp 80/kd 4`
were fine untuned. Wall-collision unverified (no obstacle geometry until MAY-171).

**OPEN — ~2× speed/time discrepancy:** base advanced 1.83 m at logged t=3.0 s under commanded
0.30 m/s (expected 0.9 m). Either the glide runs ~2× fast (scale fix) OR `world.step()`
advances more physics than the `tick·1 ms` stamp assumes (a sim-time/pacing bug likely shared
with `sim_world_main`, which stamps the same way). Resolve before trusting glide odometry.
**Deferred (Anton, 2026-07-02):** still open, confirmed during the MAY-150 fused-command
integration; the visual glide-drive demo is unaffected, so the scale fix waits for a focused
sim-time pass before SLAM/odometry work. Glide World now also serves cameras (`--cameras`,
mirrors `sim_world_main`) so `launcher.py --sim isaac --mode glide --dev-app --cameras` shows
live RGBD while driving — see [[launcher-single-entrypoint]], [[devapp_build_and_validation]].

Related: [[may172-glide-scope-defer-fit]], [[project-invariant-oli-interface]],
[[project-joystick-teleop-architecture]], [[isaac-oli-smoke-loader]].
