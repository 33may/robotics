---
name: vendor-patch-sim-freeze-until-cmd
description: vendor simulator.py patched to skip mj_step + ctrl write until first RobotCmd arrives; eliminates the bring-up free-fall window
metadata:
  type: project
---

`vendor/humanoid-mujoco-sim/simulator.py` ships an unconditional `mj_step` from frame 0. With our launcher's bring-up order (sim → deploy → autostart stand) there's a ~1 s gap before any `RobotCmd` publishes; during that gap the sim's PD law sees `Kp = Kd = 0` and Oli free-falls before the stand controller engages.

Patched 2026-06-19 to gate physics integration on a first-cmd flag:
- `SimulatorMujoco.__init__` sets `self._cmd_received = False`
- `robotCmdCallback` flips it to `True`
- `run()` skips `mj_step` while `_cmd_received` is False
- `run()` skips the per-actuator `ctrl[i] = …` assignment while `_cmd_received` is False
- Added one `mujoco.mj_forward(...)` at loop start so the first `RobotState` publish carries the real MJCF rest pose (not stale zeros); `StandController.on_start` uses that for `init_joint_angles`.

**Why:** Visually Oli was loading, collapsing for ~0.5 s, then stand was trying to recover from a partial fall. Now Oli holds rest pose until stand's first cmd, then smoothly interpolates to stand_pos.

**How to apply:** If `vendor/humanoid-mujoco-sim/simulator.py` is reset (it's the parent of the `humanoid-description` submodule, not a submodule itself, so a fresh clone would lose the patch), re-apply by adding the four diffs above. Long-term consider forking and pointing our docs there. Patch is recorded under § 10 of `humanoid/docs/vendor/humanoid-mujoco-sim.md`.

Related: [[vendor-humanoid-mujoco-sim]], [[vendor-patch-stand-autostart]], [[vendor-patch-quat-commas]].
