# Loop v2 — UI-driven inspection (design)

2026-08-20. Supersedes the operator-interaction half of
`2026-08-19-loop-v1-design.md`; the planning/perception/safety core it wired
together is unchanged. **Overturned decision:** v1's "the UI is a monitor, no
command path" is dead — the UI is now the operator surface. Reasoning from the
frontend shaped the backend: the interaction model below was designed first,
the loop restructured to serve it.

## The operator flow

Boot the app → the window shows the 3D cell, live camera, log, sphere. The
grid holds one button: **survey**. Every motion, boot included, is the same
two-press pattern:

1. Press a view button → `pending` (button recolors, text "preview") →
   backend plans.
2. Plan ok → `previewing`: the 3D robot replays the approach **in a loop**;
   the status strip says which of the 3D panel's two motion modes is showing
   (preview replay vs live mirror).
3. Press the same button again → the real arm executes, 3D mirrors it live.
4. Done → capture, fuse, recenter, all visible in the log; the view leaves
   the grid, goes `visited` on the sphere and in 3D, its image clickable.

- Pressing a **different** view mid-preview cancels the current one and plans
  the new one. No dedicated cancel; "like none" = Exit.
- **Stop** halts the arm (software stop, `executing` only). **Exit** ends the
  run cleanly.
- Plan refused → `blocked` (amber), back to idle.

Deferred to v2-AI: decider, question, per-view comments, Answer. The judgment
is in Anton's head; the record keeps poses + images + logs. The AI later slots
into exactly the seam where the mouse clicks are.

## Functional requirements

- FR1 retained-state boot: window opened any time shows the full current world.
- FR2 camera streams live always (~10 Hz), including during planning/motion.
- FR3 the 3D panel has two motion modes — preview replay and live mirror —
  disambiguated by `run/status.phase`, never by timing.
- FR4 actions panel: survey + view grid + Stop + Exit.
- FR5 first press → request command → `pending`.
- FR6 plan ok → looping preview; plan fail → `blocked`. Another view's press
  supersedes the current request/preview.
- FR7 second press on the same button executes on the real arm.
- FR8 Stop halts the arm mid-move (stopJ decel path, same as v1's ENTER).
- FR9 after the move: capture → save → fuse → recenter; view visited
  everywhere; image on the sphere panel.
- FR10 Exit saves record + fused cloud, closes hardware, exits.
- FR11 every command validated backend-side; a stale button can never move
  the arm.
- FR12 Ctrl-C is a safe shutdown: stop the arm, save, close hardware.

Stop hierarchy: pendant e-stop (hardware) > Ctrl-C (process) > UI Stop
(software). The terminal's only remaining role is hosting the process.

## Architecture: command-driven state machine + named worker threads

Chosen over (B) retrofitting the sequential `Loop` with a bus-backed console —
the press-driven flow requires abandoning blocked waits mid-state, which fights
the turn structure — and (C) asyncio, which adds a second concurrency model
while every real dependency (pinocchio, ur_rtde, RealSense) stays blocking.

One process, `inspection/run/app.py`, the composition root:

| Thread | Job |
|---|---|
| main | signals + command dispatch + state machine; never blocks; the only thread that mutates run state |
| action worker | one at a time: plan → preview-loop → execute → capture/fuse; cancellable via events |
| camera | owns the RealSense pipe; publishes `camera/wrist`; serves frames to capture |
| pose | 30 Hz `arm.q()` → `scene/poses`, active only in `executing` |
| bus | porthole server (existing) |

Startup: preflight → bus + `serve_ui` (window alive before the robot does
anything) → camera thread → `publish_world` → idle with the survey button.

Ownership rules (invariants, not conventions):

- Only the dispatcher mutates state (`visited`, `current`, phase).
- Only the action worker touches planning and `arm.execute`.
- `scene/poses` has one producer per phase: worker in `previewing`, pose
  thread in `executing`. The dispatcher flips the flag. (This also makes the
  publisher's cached pinocchio data safe — producers are sequential, never
  concurrent.)
- `stop_event` set by dispatcher (UI Stop) or SIGINT handler; cleared only by
  the dispatcher between actions.

## State machine & commands

Commands (bus command channel, untrusted, validated):

```
view/request {target}     target = "survey" | [h, v]
view/confirm {target}
run/stop {}
run/exit {}
```

```
idle ── request(t) ──▶ planning(t) ── ok ──▶ previewing(t) ⟲ replay loop
                          │ fail                  ├─ request(u) → planning(u)
                          ▼                       └─ confirm(t) → executing(t)
                     idle (t → blocked)
executing(t): done → settling → idle (t → visited)
              stop_event → idle (current unknown; replan from real q)
              executor raise → fault (grid dead, "check pendant", Exit only)
settling: capture → fuse → recenter → republish; capture fail → idle, NOT visited
```

("settling" is the worker's internal leg; it publishes `capturing` then
`fusing` on the status strip — the operator-visible vocabulary.)

- **Generation counter.** Every request bumps it; worker results carry theirs;
  stale results are discarded. `plan_viewpoint` cannot be aborted mid-call
  (C++), so superseding = ignore the result. This is v1's `console.drain()`
  ("stale type-ahead must never approve a motion") rebuilt for the bus:
  consent must be contemporaneous.
- `confirm` accepted only in `previewing` with a matching target; otherwise
  logged and dropped.
- `stop` acts only in `executing`; elsewhere a logged no-op.
- `exit` during `executing` = stop first, then shutdown.
- Defense in depth: `arm.execute` still re-checks start tolerance + live world
  before the first command. After any stop the old plan fails start-mismatch
  by construction.

## Topic amendments (vs `inspection/ui/AGENTS.md`)

No new topics; commands are not topics. Three changes:

- `views/state` cell states become `available | pending | previewing |
  visited | current | blocked | unreachable`. Still one `publish_views()`
  emitting markers + JSON, so 3D, sphere panel, and grid read the same truth.
  Two new `STATE_COLOR` entries.
- `views/state` gains a top-level `survey` entry (same vocabulary) so the
  grid renders the survey button before any sphere exists. After boot:
  `visited`, disabled (no return-home in v1).
- `run/status.phase` ∈ `idle | planning | previewing | executing | capturing |
  fusing | fault | done`, plus `target`. `awaiting_approval` dies with the
  terminal.

All state topics stay retained; a mid-run window reconstructs completely,
half-finished preview included.

## Streamers & hardware access

**Camera thread** — sole owner of the pipe. `wait_for_frames` → align → cache
latest under a lock → JPEG at ~10 Hz. `capture()` asks the thread for a
bundle: ≥3 frames newer than the request (v1's `flush=3, settle_s=0.1`
semantics). Direct `capture_bundle(pipe, …)` access retires. Camera death is
not run death: log + retry; captures fail through the existing
capture-fail path; the arm never depends on the camera.

**Pose thread** — shares the one `RTDEReceiveInterface` with the executor's
safety polls behind a lock (micro-second reads; sidesteps the multi-client
RTDE question). Verify ur_rtde's documented thread-safety at implementation;
the lock is correct either way.

**Preview replay** — the worker loops the dense path at 30 Hz until
confirm/cancel. Freeze-at-impact retained for the degenerate case only; a
colliding preview means the plan was invalid → `blocked`, not a frozen loop.

## Safety & shutdown

- UI Stop reuses v1's proven path unchanged: `stop_event` → `_wait_async` →
  `stopJ(2.0)` → `{"stopped": True}`.
- **The Ctrl-C gap closes structurally.** v1: `arm.execute` catches only
  `Exception`; `KeyboardInterrupt` mid-move would not stop the commanded
  motion. v2: signals land in the main thread, which is never inside
  `arm.execute`; SIGINT sets `stop_event`, the worker decelerates properly.
  No change to `execute.py`.
- Shutdown (Exit and SIGINT alike): stop event → join worker (bounded) →
  belt-and-braces `stopJ` → save → camera stop → `arm.close()` → bus stop.
  Second Ctrl-C = hard exit.
- `run.json` written on every state change, not at run end.
- Dead browser mid-move is safe by construction: motions are finite and
  pre-validated; the system idles. No heartbeat in v1 (deliberate; bus
  exposes `clients` when an indicator is wanted).
- Dead bus = blind but alive: publisher swallows everything; Ctrl-C works.

## Frontend, mock, tests

- **`ActionsPanel.tsx`** (app-owned): survey + grid + Stop + Exit. Pure IO,
  no local state — click sends a command, truth returns via retained
  `views/state`. Grid layout: pending Anton's drawing. Verify the TS
  framework's command-send API shape at implementation.
- CloudInspect + scene: add the two new state colours. Status strip: new
  phases + target.
- **Mock = same brain, fake muscles.** `app.py mock` runs the real
  dispatcher + state machine with a `FakeRig` (real: world, IK, planning,
  viewsphere, replay; fake: slow-interpolating arm so `executing` is
  watchable and stoppable, canned camera frames). Full click-flow with zero
  hardware; no UI-side demo branch exists to rot.
- Python tests (FakeRig + injected commands): happy path, cancel-supersede,
  stop mid-execute, stale confirm rejected, generation counter, capture-fail
  → not visited, SIGINT save path.
- `npm run check`: ActionsPanel renders every state; clicks emit the right
  commands onto a capture bus.

## Retired / kept / deferred

- Retired: `Loop`, `Console` approval flow, `TerminalDecider` in the run
  path, meshcat preview in the run path, `loop.py run()`.
- Kept: `teach()`, all of `motion/`, `cell/`, `view/`, `perception/`
  (capture moves behind the camera thread), `publisher.py` (grows the new
  states), meshcat for debug CLIs, `decider.py` parked for v2-AI.
- Deferred: AI decider + question/comments/answer, return-to-survey,
  UI heartbeat/deadman, multi-object, re-boot without restart.
