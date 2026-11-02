"""devapp — the Oli robot-brain developer app.

A standalone Dear ImGui Bundle application, booted inside the brain process, that lets
us *see and manage everything happening inside the robot brain* from one window. It does
NOT own the World: the robot is already spawned in some World (Isaac / MuJoCo / real) and
this app attaches afterwards.

Layers (each independently swappable):
    panel.py      the plugin contract — a dockable window is a `Panel`
    registry.py   register panels in one line; the shell iterates
    state.py      AppState — latest-wins buffer shared UI thread ↔ brain thread
    app.py        the shell — Hello ImGui runner, docking, main loop
    capture.py    headless screenshot for self-validation
    sources/      data sources behind stable protocols (CameraSource, …) — swap sim↔real
    panels/       the concrete windows (camera, teleop, …)

Design rationale: agent memory `project_dev_app_stack`; doc `docs/dev-app/design.md`.
"""
