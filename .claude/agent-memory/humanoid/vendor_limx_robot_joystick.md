---
name: vendor-limx-robot-joystick
description: LimX robot-joystick is a keyboard-driven pygame+limxsdk PyInstaller app that publishes SensorJoy; axis/button/mode mapping.
metadata:
  type: reference
---

`humanoid/vendor/humanoid-mujoco-sim/robot-joystick/robot-joystick` (42 MB ELF, + `.exe`)
is LimX's shipped joystick app. `strings`/`ldd` reveal it is a **PyInstaller-frozen
Python 3.8 app** built from exactly two pieces: `pygame/joystick…so` and
`limxsdk/robot/_robot.so`.

What it actually is (NOT a physical-gamepad reader):
- A **keyboard-driven on-screen virtual gamepad** (pygame window). Bindings from
  `robot-joystick/doc/joystick.png`: Arrow keys = left stick, Numpad 8/5/4/6 = right
  stick, letter keys = buttons (Y:I A:K X:J B:L L1:Q L2:E R1:U R2:O, START:X …).
- It publishes `limxsdk.SensorJoy{stamp, axes[], buttons[]}` on the MROS bus; the
  sim/controller subscribes via `Robot.subscribeSensorJoy`.

Mappings (authoritative — used by our port):
- **Axes** (per `humanoid-rl-deploy-python/.../walk_controller.py` `_update_commands_from_joy`):
  `v_x ← axis1`, `v_y ← axis0`, `w_z ← axis3`, each clipped to max_v*.
- **Buttons / mode** (per `humanoid-rl-deploy-python/main.py`, PlayStation indices):
  `L1(4)+Y(3) → STAND`, `R1(7)+X(2) → WALK`, `L1+B(1) → mimic`, `L1+A(0) → damping`,
  `L1+X → exit`. In keyboard terms: Q+I → stand, U+J → walk.

Consequence: "their joystick app" is trivially portable as ~100 lines of pure pygame —
see [[project-joystick-teleop-architecture]]. The physical-gamepad path on the real
robot is a separate `mrosjoy` ROS node reading `/dev/input/js0`. Related:
[[walk-policy-obs-builder-fidelity]], [[vendor_humanoid_mujoco_sim]].
