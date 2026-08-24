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
  survey pose). The native window is a child process (`inspection/ui/app.py
  serve`), not a thread: pywebview needs a main thread and this one belongs
  to the dispatcher. Teardown order is the design's:
  `stop_event` → `sup.join_workers(bounded)` → `stopJ` → save → close
  hardware → close window → stop bus.
- `rigs.py` — `FakeRig`/`RealRig`/`CameraWorker`/`PoseStreamer`. The shared
  rig contract both hardware and the mock implement is `q()`/`move(path)`/
  `capture(pose_id)`/`close()` plus a `stop_event` attribute; `FakeRig`
  additionally has `frame()` for its own capture/grab plumbing — `RealRig`
  has no equivalent. Live camera frames instead flow through
  `CameraWorker`, wired in `RealRig.start_camera` (hardware) /
  `mock.start_mock` (mock).
- `segmenter.py` — object identity for the geometry path. `ObjectSegmenter`
  (box → mask, with the policy that decides when to believe it: score floor,
  minimum pixels, backend exceptions swallowed into a miss) and
  `object_view(cap, ...)`, the whole per-view identity decision the settle
  leg calls. Lives HERE because `cell/geometry.py` may not import `eyes`
  (cycle via `eyes/tools.py`) and `perception/` is barred from it too — the
  run tier is the only one above both. Backend is injected, so tests use
  `StubBackend` and never touch a GPU. `p inspection/tests/test_segmenter.py`.
- `decider.py` — parked for v2-AI; not wired into the run path since loop
  v2 (see `inspection/2026-08-20-ui-driven-loop-design.md`). Still holds
  the egocentric menu-gloss helpers (`gloss`, `build_menu`) and the v1
  terminal implementation (`Console`, `TerminalDecider`), kept for when
  an AI decider slots into the request/confirm seam.
- `survey_pose.json` — taught on hardware, not committed until it exists.

## Contracts & decisions
- Nothing imports from `run/`. If a sibling package needs something
  defined here, that something is in the wrong place — push it down.
- **The segmenter is optional and the loop must survive without it.**
  `Supervisor(segmenter=None)` is the pure depth pipeline; every failure
  mode — no rgb, object out of frame, low score, tiny mask, a backend that
  raises — falls back to depth growth with a warning rather than aborting.
  A model failure is a LIVENESS cost (a fuzzier collision box, still guarded
  by the jump gate and the percentile extent); dropping the view would be a
  safety one. The real segmenter is injected in `app.py` only, and its
  weights load lazily on first capture, never at import.
- **Identity persists through geometry, not through text.** No noun phrase
  is used anywhere in the loop: the accumulated cloud plus `T_base_cam`
  reproject into each new view to give SAM its box (`prompt_box`). That keeps
  the question ("is there a logo on this cup?") entirely in `eyes/`, costs
  no extra inference, and is instance-level by construction — a text prompt
  would re-find *any* cup. Measured across a 180 deg roll and a ring change:
  scores 0.73-0.97, 5 calls, 0 misses.
- Masks are written beside their capture (`mask.png` + `meta.json:mask`) in
  the SAME orientation as `rgb.png`, so a run stays replayable offline and
  the artifact that decided the geometry is on disk. Best-effort: a disk
  error there must never fail a settle that otherwise succeeded.
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
