---
name: project_dev_app_stack
description: Oli developer app ("robot OS") is a standalone Python app built on Dear ImGui Bundle (imgui_bundle), booted inside the brain process, with camera/3D/state/reasoning as in-window plugin panels; decided 2026-07-01 (MAY-150).
metadata:
  type: project
---

The Oli **developer app** (aka the "robot OS" — one place to *see and manage everything happening inside the robot brain*) is a **standalone Python app** built on **Dear ImGui Bundle (`imgui_bundle`)**, **booted inside the brain process** (py3.11 `brain` env). Camera streams, the 3D scene, state panels, reasoning-debug, scene loader and input panel are all **plugin modules built inside the app** — a plugin = a Python draw-function registered into a named dock space.

**Why:** Anton wants one *owned* app whose render loop IS (or hosts) the brain's loop — imgui's `immapp.run()` frame callback steps the brain: one process, one thread, deterministic, no thread/GIL dance. Immediate-mode + owning the GL context means camera (GL texture) and 3D (moderngl / pygfx / `pyvista-imgui`) are just draw calls in a panel. Must work **sim AND real** (deployment-invariance is the project thesis), which killed the sim-locked and external-viewer options.

Alternatives rejected (2026-07-01 research):
- **Isaac omni.ui** — 3D viewport free, but Kit-only → sim-locked. Out.
- **rerun.io** — best 3D/observability engine, but it's a *log-to viewer* with no interactive control widgets → can't be the shell (could still be embedded in a plugin later).
- **Dear PyGui** — 3D is only the 2D drawing API (no depth buffer); Dear Py3D still a prototype mid-2026; docking experimental. Out.
- **Qt/PySide6** — runner-up: retained rqt-style plugin framework + VTK dense-cloud 3D, but heavier and a second loop bolted next to the brain rather than hosting it.

**How to apply:** Build all dev-tooling as imgui_bundle plugin panels in this app, not ad-hoc scripts or an external viewer. If dense point clouds (SLAM) later need VTK, embed via `pyvista-imgui` rather than switching frameworks. Design doc lives at `humanoid/docs/dev-app/design.md`. Cameras ([[MAY-149]]) are the first plugin — the "cameras" slice of state observability. Related: [[project_invariant_oli_interface]], [[walkmatch_actuator_id_harness]].
