---
name: devapp-build-and-validation
description: The Oli robot-brain dev app as built at logic/oli/devapp/ (imgui_bundle) — plugin layers, run/headless-screenshot commands, and the imgui_bundle/immvision gotchas found during the MAY-150 camera+teleop milestone.
metadata:
  type: reference
---

Built 2026-07-01 (MAY-150, first milestone: camera streams + joystick launch). Decision
rationale in [[project_dev_app_stack]]. Lives at `logic/oli/devapp/`, runs in the **brain**
env (py3.11). Does NOT own the World — attaches after the robot is spawned.

**Plugin layers (each swappable):**
- `panel.py` — `Panel` base (title, dock_space, setup/draw(state)/teardown) = the plugin contract.
- `registry.py` — `PanelRegistry.register()`; the shell iterates it (add a panel = 1 line in `__main__.build_registry`).
- `state.py` — `AppState` latest-wins buffer (UI↔brain thread); thin for now (frame counter).
- `app.py` — the shell: Hello ImGui runner, full-screen dock space, splits MainDockSpace|LeftSpace|BottomSpace.
- `capture.py` — `enable_screenshot(rp, path, n_frames)`: render N frames → `final_app_window_screenshot()` → PNG → exit.
- `sources/camera_source.py` — `CameraSource` protocol + local `CameraFrame` (mirrors MAY-149's forthcoming `contracts.CameraFrame`: stamp,name,rgb,depth,intrinsics). Panels depend on the protocol, never Isaac.
- `sources/synthetic_camera_source.py` — `SyntheticCameraSource` (animated RGBD; the current stand-in until MAY-149's SOCK_STREAM source lands).
- `launcher.py` — `ProcessLauncher` (generic start/stop/status for a child process).
- `panels/` — `CameraPanel` (immvision RGBD), `TeleopPanel` (launches `reason.teleoperation.joystick.app`, port 9001), `InfoPanel`.

**Run — ONE command boots World + app (walking, windowed):**
```
p logic/oli/launcher.py --sim isaac --dev-app --vx 0.3   # (run_oli_sim.py is now a shim → this)
```
The entrypoint is now the unified [[launcher-single-entrypoint]] (`--sim`/`--mode`); `run_oli_sim`
is a thin shim. `--dev-app` makes the isaac backend route the brain half to `-m humanoid.logic.oli.devapp`
(instead of `brain_main.py`) with the SAME `--socket`/locomotion flags. The app then runs the
brain itself: `brain_link.BrainLink` builds the same Orchestrator (`BrainComm`→World, `Teleop`,
`PolicyRunner`, joystick) on a **daemon thread**, its `recorder` feeds `AppState`, and panels
render live (StatePanel: mode/stamp/base-tilt). The app does NOT own the World — attaches to
whatever serves the socket. `--mode walk` → launch the joystick from the Teleop panel button.

**Run app standalone (UI only, synthetic cameras, no brain):**
```
/home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp        # windowed
# attach to an already-running World manually:
/home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp \
    --socket /tmp/oli-world.sock --mode walk --joystick fixed --vx 0.3
```
**Headless self-validation screenshot (no monitor — this is how the agent SEES the app):**
```
PYTHONPATH=<repo-root> xvfb-run -a -s "-screen 0 1600x1000x24" \
  /home/may33/miniconda3/envs/brain/bin/python -m humanoid.logic.oli.devapp \
  --screenshot /tmp/app.png --frames 30
```
Then Read the PNG. `-m humanoid...` needs repo root on `PYTHONPATH` (or cwd=repo root).

**Deps installed in the brain env only:** `imgui-bundle` (1.92.801) + `pillow`; system `xorg-x11-server-Xvfb` (Fedora `dnf`). pygame 2.6.1 already in brain (joystick app).

**Glide fusion (2026-07-02, block-2):** dev app can attach as the **glide** brain —
`BrainLink._make_action("glide")` builds `GlideAction` (not `PolicyRunner`); devapp `--mode`
gained `glide`; `StatePanel` shows the `GlideCmd` (vx/vy/wz) when `policy_out` has no `q_des`.
`run_oli_sim.py --glide` boots `glide_world_main.py` (glide tuning flags: `--hold-kp/kd`,
`--height-kp`, `--foot-clearance`, `--lin/yaw-accel`) + brain `--mode glide`; so **`run_oli_sim
--glide --dev-app`** = the fused command (glide World + dev app brain, `--joystick socket` so the
Teleop panel's vendor pad drives). Headless-proven: `test_brainlink_glide_forwards_velocity_to_world`
(fake World receives a `GlideCmd`). Still TODO to fuse: real camera source (blocked on MAY-149
serving frames) + live integration verify (P4). 24 devapp + 19 runner tests green.

**Teleop / joystick (2026-07-01):** the dev app's `TeleopPanel` launches the **EXACT vendor
`robot-joystick` binary** (full gamepad) + `logic/simulation/mujoco/sensorjoy_bridge.py`
(run via the sibling `limx` env python) — NOT our stripped `reason/.../joystick/app.py`.
The bridge is a py3.8 limxsdk POLICY peer: subscribes `SensorJoy` on the loopback MROS bus,
re-emits our `JoyPacket` UDP :9001; the brain's `SocketJoystickSource` consumes it. **Proven
STANDALONE** — vendor pad + bridge alone forward ~100 pkts/s on loopback with NO sim / MROS
edge (the bridge's "is the sim up?" error is pessimistic; `robot.init('127.0.0.1')` succeeds
solo). So the dev app reuses the mujoco joystick path verbatim; no native imgui joystick
rebuild needed. Focus the pad window to drive. Same pair `run_oli_mujoco.py --spawn-app` uses.

**Gotchas / fixes:**
- imgui default font has no `→` glyph (renders `�`) — use ASCII `->` in panel text.
- **immvision requires `immvision.use_rgb_color_order()` once at startup** (breaking change Oct 2024) or it raises at first `image_display`. Called in `CameraPanel.setup()`. Our frames are RGB → use_rgb (not bgr); then pass arrays as-is (no channel flip).
- **Exit segfault** if immvision's GL texture cache outlives the GL context → call `immvision.clear_texture_cache()` in `CameraPanel.teardown()` (before context destroy). Fixed → exit code 0.
- Hello ImGui `RunnerParams.callbacks.*` default to **None** (not a no-op) on a fresh instance → guard chained calls (`if prev is not None: prev()`).
- Don't name non-test helper classes `Test*` — pytest tries to collect them (renamed `TestCameraSource`→`SyntheticCameraSource`).
- **`immvision.image_display` two hard gotchas (cost real time 2026-07-02):** (1) pass
  `refresh_image` POSITIONALLY — `image_display(label, img, disp, True)`; this build rejects it as
  a **kwarg**. (2) the image MUST be a **WRITEABLE** C-contiguous uint8 array. `decode_camera_frame`
  returns a **read-only** `np.frombuffer` array, and `np.ascontiguousarray` KEEPS it read-only when
  it's already contiguous → ImmVision raises "incompatible function arguments". Fix in the panel:
  `np.array(x, dtype=np.uint8, copy=True)` (forces a writeable copy). The synthetic source dodged
  this because it already `ascontiguousarray`'d fresh (writeable) arrays. Labels must be unique (`##uid`);
  frames change each tick so refresh must be on. Root-cause note: a cleaner fix would be to make
  `codec.decode_camera_frame` return writeable arrays, but that adds a copy for all consumers.

Tests: `tests/oli/devapp/` (20, all `@pytest.mark.brain`, pure — GUI validated by screenshot). Related:
[[launcher-single-entrypoint]], [[oli-perception-camera-design]], [[project_joystick_teleop_architecture]], [[feedback_tests_in_repo_tdd]], [[feedback_show_plots_inline]].
