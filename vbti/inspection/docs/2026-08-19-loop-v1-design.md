# Inspection Loop v1 (POC) — Design

Date: 2026-08-19
Parent: `2026-08-12-ai-inspection-project-definition.md` (D3/D4 govern; this
implements the orchestrator with a human in the model slot)

## Goal

Close the D3 loop on the real UR5e with Anton playing READ and DECIDE. The
harness — context assembly, menu, motion, capture, geometry update, record —
is the deliverable; it must not change when the AI replaces the human.

## Decisions (this brainstorm, 2026-08-19)

- **Action space**: `look(h, v)` and `answer(text)` only — exactly D4. No
  detection map in v1: **one object on the table**, the above-table cluster is
  the object (detection + cluster pick moves to v2).
- **Survey pose**: freedrive-taught. One-time `teach` subcommand reads q via
  RTDE and saves `run/survey_pose.json`. Boot plans current → survey.
- **Approval gate on every motion, boot included**: plan → replay in the 3D
  view → terminal `approve? [y/n]` → only then execute. Interim 3D view is
  meshcat (`world.replay`); Anton's custom frontend (separate project, panel
  toolkit: pywebview + React/dockview + websocket msgpack bus) takes over the
  replay/approve later.
- **Software stop**: armed the whole run. ENTER during motion → `stopJ`
  deceleration. Requires async `moveJ` + progress polling in the executor
  (blocking moveJ cannot be interrupted from software).
- **After a stop**: no recovery state. Arm is at a known q; every plan starts
  from current q, so the loop simply returns to the menu (re-pick same cell,
  pick another, or quit).
- **Interface v1**: terminal decider + small cv2 window showing the newest
  capture (stopgap until the frontend). The printed context IS the future
  model prompt; typed comments become the evidence log.
- **UI decoupling**: `Decider` interface is the AI seam; `Publisher` stub is
  the UI seam. Loop core knows neither terminal nor bus nor model.

## Architecture

```
inspection/run/
  loop.py      orchestrator: boot, turn machine, coverage, run record.
               Wires viewsphere → plan → approve → execute → capture →
               geometry. Nothing imports from run/ (existing contract).
  decider.py   Decider interface:
                 read(capture) -> comment        (READ)
                 decide(ctx)   -> Look(h,v) | Answer(text)
               v1 impl: TerminalDecider. Later: BusDecider, AgentDecider.
  survey_pose.json   taught joints (not committed until taught)
```

`motion/execute.py` gains interruptible motion: per-waypoint
`moveJ(..., asynchronous=True)` + ~50 ms progress poll + stop-event check →
`stopJ` on trigger; existing preflight/start-tolerance/safety checks stay.

## Flows

**Boot**: preflight (NO-GO exits) → load survey pose → plan current→survey →
replay → approve → execute → capture `000` → `object_from_view` → centroid →
`ViewSphere` + reachability.

**Turn** (≤ `--max_turns`, default 12):
1. Show capture (cv2 stopgap); comment = one typed line (READ)
2. Print ctx: coverage counts, feasible unvisited menu with egocentric
   glosses, current cell → type `look H V` or `answer <text>` (DECIDE)
3. `look`: `plan_to_cell` → replay → approve → execute → capture → fuse cloud
   (`CloudAccumulator`) → re-centroid → re-reach (~33 ms). Plan failure →
   cell marked blocked this round → menu.
4. `answer`: write `run.json`, arm stays put, loop ends.

## On disk

Existing bundle format untouched: `outdir/session.json`, `NNN/` per capture.
New: `outdir/run.json` — per turn: comment, decision, cell, plan tier,
approved/stopped flags, timings. This log is the AI's future context, free.

## Error handling (POC floor)

- Plan fail → blocked cell, back to menu. Capture fail → error printed, menu.
- Software stop → menu. Hardware e-stop / protective stop → preflight-style
  message, loop exits (restart resumes from current q by design).
- Deliberately out: budget/termination gates beyond max_turns, radius
  adaptation (fixed `--r`), answer gating, retries, detection map.

## Testing

Live supervised run with the duck: every motion is human-approved, software
stop tested on the first boot move. No sim-only mode in POC — the approve
gate is the safety net, and planning/replay already validate against the
calibrated world model.
