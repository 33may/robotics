# run — orchestration

## Purpose
Top-level entry points: the exploration loop and demo scripts. This is
the only layer allowed to talk to everything — perception, cell, view,
motion — and wire them together.

## Files
- `loop.py` — the v1 inspection loop (POC): boot to the taught survey pose,
  seed the object, then READ -> DECIDE -> plan -> meshcat preview ->
  approve -> move -> capture -> fuse. Every motion is human-approved;
  ENTER during motion is the software stop. Subcommands: `teach`
  (save freedrive survey pose), `run`. Spec:
  `inspection/2026-08-19-loop-v1-design.md`.
- `decider.py` — the AI seam. Decider.read/.decide contract, egocentric
  menu glosses (D4), Console (stdin owner + stop arming), TerminalDecider.
  BusDecider (UI toolkit) and AgentDecider slot in here later.
- `survey_pose.json` — taught on hardware, not committed until it exists.

## Contracts & decisions
- Nothing imports from `run/`. If a sibling package needs something
  defined here, that something is in the wrong place — push it down.
- Orchestration only: sequencing, UI, logging, entry points. Any
  geometry, planning, or camera logic that accretes here gets moved to
  its pipeline stage.
- The AI slot: it replaces exactly the Decider (read + decide), nothing
  else. The orchestrator owns budget, approval, and the software stop.

## Does NOT belong here
- Reusable logic of any kind — this layer is glue, not a library.
