# run — orchestration

## Purpose
Top-level entry points: the command-driven inspection loop and demo
scripts. This is the only layer allowed to talk to everything —
perception, cell, view, motion — and wire them together.

## Files
- `machine.py` — `Supervisor`, the v2 command-driven state machine: one
  dispatcher thread owns every bit of run state; named worker threads
  (plan/preview/exec) talk back only via events on a queue. Commands:
  `view/request`, `view/confirm`, `run/stop`, `run/exit`, all validated
  backend-side (a stale button can never move the arm). Spec:
  `inspection/2026-08-20-ui-driven-loop-design.md`.
- `app.py` — composition root. `run` (bus + window + `Supervisor` +
  `RealRig`, SIGINT-safe shutdown) and `teach` (freedrive-then-save the
  survey pose).
- `rigs.py` — `FakeRig`/`RealRig`/`CameraWorker`/`PoseStreamer`: the rig
  contract (`q`/`move`/`capture`/`frame`/`close`) that both hardware and
  the mock drive identically.
- `decider.py` — parked for v2-AI; not wired into the run path since loop
  v2 (see `inspection/2026-08-20-ui-driven-loop-design.md`). Still holds
  the egocentric menu-gloss helpers (`gloss`, `build_menu`) and the v1
  terminal implementation (`Console`, `TerminalDecider`), kept for when
  an AI decider slots into the request/confirm seam.
- `survey_pose.json` — taught on hardware, not committed until it exists.

## Contracts & decisions
- Nothing imports from `run/`. If a sibling package needs something
  defined here, that something is in the wrong place — push it down.
- Orchestration only: sequencing, UI, logging, entry points. Any
  geometry, planning, or camera logic that accretes here gets moved to
  its pipeline stage.
- Command authority: `Supervisor` is the only thing that mutates run
  state or moves the arm; every command it receives is validated before
  anything happens (design FR11). The AI slot, when it lands, replaces
  exactly the operator's judgment at `view/request`/`view/confirm` —
  not the state machine, the safety checks, or the stop hierarchy.

## Does NOT belong here
- Reusable logic of any kind — this layer is glue, not a library.
