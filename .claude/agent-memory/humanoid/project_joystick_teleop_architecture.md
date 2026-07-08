---
name: project-joystick-teleop-architecture
description: Joystick teleop = ported pygame app emitting our own UDP JoyPacket wire into a pure brain; layout, ports, open items.
metadata:
  type: project
---

Joystick teleoperation for Oli (decided & built 2026-06-24, branch
`33may/may-147-wire-limxsdk-loop-in-isaac`, change `may-147-oli-deployment-interface`).

**Decision:** rather than spawn LimX's opaque `robot-joystick` binary (which publishes
`limxsdk.SensorJoy` on the MROS bus, py3.8 — see [[vendor-limx-robot-joystick]]), we
**ported it as code** with **our own communication protocol**, so `limxsdk` leaves the
joystick path entirely and the brain stays pure.

**Why:** the brain must import neither `isaacsim` nor `limxsdk` (the enforceable
deployment invariant). Reading `SensorJoy` would need `limxsdk` in-brain → breaks it.
Owning the wire also lets a future remote/phone joystick just change the UDP host.

**Layout** — `logic/oli/reason/teleoperation/joystick/` (teleoperation = method family
for future glove/mocopi/VR; joystick = method #1; Anton chose "everything under
joystick/"):
- `protocol.py` — `JoyPacket{stamp_ns, axes[], buttons[]}`, UDP datagram, magic `JOY1`,
  default port **9001**, latest-wins, pure stdlib `struct`.
- `source.py` — `SocketJoystickSource` (bind UDP, drain-to-newest, hold-last) +
  `FixedJoystick` (constant, for tests/scripted demo). Both satisfy `JoystickSource`
  protocol (`poll() -> JoyPacket | None` — returns the FULL stick state, axes + buttons,
  not bare axes, so Teleop can read buttons for mode).
- `app.py` — keyboard pygame window → `JoyPacket` over UDP. `keyboard_to_packet` is the
  pure tested core; pygame imported **lazily in `run()`** so the brain imports it
  pygame-free. Axes: axis0=v_y, axis1=v_x, axis3=w_z. Run:
  `python -m humanoid.logic.oli.reason.teleoperation.joystick.app`.
- `teleop.py` — `Teleop` + `JoystickAdapter` (moved here from `reason/teleop.py`).
  `Teleop.to_policy_in(obs, joy)` takes a `JoyPacket`: axes→velocity, **button combos→mode**
  (sticky/latched). Mode-from-buttons lives in Teleop, NOT the Orchestrator (D5: button
  combo→intent is reasoning). Combos = LimX `main.py`: L1+Y (Q+I)→STAND, R1+X (U+J)→WALK.
- Tests mirror under `tests/oli/reason/teleoperation/joystick/` (~37 tests, `brain` env).

`brain_main.py`: `--joystick fixed|socket`, `--joy-port`, `--spawn-app --app-python`.

**Env note:** `pygame` is in the `isaac` env, NOT `brain`. The app is a separate
process (UDP-decoupled) so run it wherever pygame is (isaac env), or `pip install
pygame` into brain to let `--spawn-app` use the brain interpreter.

**Done since:** button→mode wired (2026-06-25) — `poll()` now returns the full
`JoyPacket`, `Teleop` latches mode from button combos. Verified end-to-end (keys →
packet → source → Teleop → PolicyIn) in `test_joystick_integration.py`.

**Open / next:**
- Live run needs the **other agent's** `sim_world_main` (§8.3, already built — Oli STANDS;
  forward walk topples after a few steps, under debugging) — then drive Oli with the stick.
- Possibly add `pygame` to the `brain` env for one-interpreter `--spawn-app` (today the app
  runs in the `isaac` env, which has pygame).

**How to apply:** build on this seam when extending teleop or wiring mode; keep the
brain pure (anything touching `limxsdk`/devices lives in the app process or RealComm,
never the brain).
