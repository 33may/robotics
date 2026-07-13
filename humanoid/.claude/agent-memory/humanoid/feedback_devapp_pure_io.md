---
name: feedback-devapp-pure-io
description: dev_app panels are pure I/O (render + forward input via AppState) — planning/reasoning logic lives in the brain module behind clean in/out contracts, never in the panel
metadata:
  type: feedback
---

dev_app panels are **pure I/O**: render state from `AppState` and forward user input into it. They
must contain **no domain logic** — no planning, costmaps, or reasoning. That logic lives in the
brain module (e.g. `reason/nav`) behind clean **in/out contracts**.

**Why:** Anton reacted sharply ("wtf, why does the motion logic live in dev_app, this is bullshit")
when I put `plan_path`/`inflate`/`clearance_cost` in `MapPanel` as a shortcut for a no-motion path
preview. It violates his core architecture: planning has a single place — the brain. A dev_app
panel calling the planner is the wrong side of the boundary, even for a "preview."

**How to apply:** the correct shape is a contract flow, e.g. nav: **goal IN** (`GoalCoordinate` →
`Nav.set_goal`) → Nav plans on its OWN costmap → **path OUT** (`Nav.plan(pose)` / `Nav.path`) →
panel renders it. The `AppState` seam only shuttles contracts across the UI↔brain threads (UI
writes goal, brain writes pose+path). The panel may still un-project a click to world coords
(that's display geometry it owns), but it hands off a world-frame contract and does zero planning.
Name spatial goal contracts `GoalCoordinate`, reserving "goal" for higher-level reasoning/task
goals later. See [[architecture-nav-costmap-layering]] and [[nav-brain-localizer-seam]].
