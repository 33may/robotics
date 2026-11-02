# run — orchestration

## Purpose
Entry points and the command-driven state machine. The only layer allowed to
wire perception, cell, view, motion, fit, brain and record together.
Design: `docs/blueprints/ui-driven-loop.md`.

## Files
- `machine.py`: `Supervisor`: one dispatcher thread owns all run state;
  worker threads (plan/preview/exec) talk back only via events on a queue.
  Commands `view/request`, `view/confirm`, `run/stop`, `run/finish`,
  `run/exit`, all validated backend-side. A generation counter (`self.gen`)
  makes a stale button or stale agent request unable to move the arm. Moves
  are `motion/transaction.py` legs; the exec worker records the waypoints it
  actually travelled (`_arrival_path`) for the retreat. Optional live fitting
  (`fit=` name from `fit/apply.py:METHODS`).
- `settle.py`: `settle_capture`: one capture → object identity + table-plane
  gate (`PLANE_Z_TOL_M`, `PLANE_TILT_TOL_DEG`) + fuse, or refused. No
  publishing or disk writes; used by the machine and offline replay.
- `segmenter.py`: `ObjectSegmenter` (box → mask, with the belief policy:
  score floor, minimum pixels, backend exceptions → miss) and `object_view`,
  the per-view identity decision. Lives here because `cell/` may not import
  `eyes/` and `perception/` stays model-free. Backend injected; tests use `StubBackend`.
- `app.py`: composition root (fire CLI): `run` (bus + UI child process +
  `Supervisor` + `RealRig` + cognition; `brain/ask` starts the planner),
  `collect` (Flow B data sweep, no cognition), `teach` (freedrive, save
  `survey_pose.json`). Also `config_snapshot`, `finish_run`. `run` lands in
  `data/runs/`, `collect` in `data/datasets/`; `--object` from `OBJECTS`.
- `collect.py`: `SweepDriver`: requests every open cell through
  `SupervisorMover`, cheapest-first (or `--random`). It keeps its OWN
  `RobotCell` for costing; `sup.world` belongs to the dispatcher.
- `rigs.py`: `FakeRig`/`RealRig`/`CameraWorker`/`PoseStreamer`. Shared rig
  contract: `q()`/`move(path, world)`/`capture(pose_id)`/`close()` plus a
  `stop_event`. Live frames flow through `CameraWorker`
  (`RealRig.start_camera` / `ui/mock.py:start_mock`).
- `decider.py`: v1 terminal decider (`Console`, `TerminalDecider`,
  `build_menu`). Not wired into the run path; only `tests/test_decider.py` uses it.
- `survey_pose.json`: taught survey joint pose (`app.py teach`).

## Contracts & decisions
- Only composition roots import from `run/` (`ui/mock.py`,
  `calib/depth_eval.py`, `investigation/`). A library module needing
  something here means it is in the wrong place: push it down.
- **Threading.** Only the dispatcher mutates run state, with two deliberate
  exceptions (the settle leg's `acc.add`/`_recenter()`, and `_reach` refresh
  in the workers), safe because exactly one action worker is alive at a time.
  That is enforced: OMPL cannot be cancelled, so a `view/request` during
  `planning` is deferred (`_pending_request`, last wins); `_cancel_preview()`
  joins before any spawn. `RobotCell` is not thread-safe.
- `PoseStreamer` handoff is structural: the settle leg waits for `_pose_ack`
  and `pose_quiesce()` before `_recenter()`. Both waits are bounded.
- **Teardown order** (`app.run` finally): `stop_event` → clear `pose_active`
  → `join_workers(5 s)` → stop poses → `stopJ` → `finish_run` (guarded) →
  `rig.close()` → close UI → stop bus.
- **The segmenter is optional.** `Supervisor(segmenter=None)` is the pure
  depth pipeline; every segmenter failure falls back to depth growth with a
  warning. A model failure costs liveness (fuzzier box, still jump-gated),
  never safety. `app.py` warms SAM 3 in a background thread at startup; a
  failed warm-up falls back to lazy load. Never at import.
- **Identity persists through geometry, not text.** The accumulated cloud +
  `T_base_cam` reproject into each new view as SAM's box prompt
  (`prompt_box`). No noun phrase anywhere in the loop; the question stays in
  `eyes/`/`brain/`. Instance-level by construction.
- Masks are written beside their capture (`mask.png`, same orientation as
  `rgb.png`); score/box/px live in `steps/NNN/step.json:segmentation`.
  Best-effort: a disk error there never fails a settle.
- **One writer.** The `Supervisor` writes only through its injected
  `RunWriter` (`begin_step`, `write_capture`, `write_fused`, `mark_step`,
  `event`). The layout is `record/schema.py`. The exec worker writes the
  capture before queueing `settle_done`; the dispatcher writes the fused half.
- **Command authority.** `Supervisor` is the only thing that moves the arm.
  The planner and the sweep press `view/request` like the UI; `view/confirm`
  stays with the operator (except `collect --auto`). STOP outranks everything.
- Orchestration only: geometry, planning or camera logic that accretes here
  moves to its pipeline stage.

## Does NOT belong here
- Reusable logic of any kind: this layer is glue, not a library.
