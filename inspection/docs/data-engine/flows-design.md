# Live flows through the record layer — design

Status: APPROVED in discussion, 2026-09-07 (Anton + robot agent). Step "wire the
loop" of the data-engine plan (MAY-193), after schema/writer/management landed
on `data-engine/record-layer`.

Two flows, one machine: **A** — the live AI inspection run, recorded through
the schema. **B** — the data-collection sweep: same loop, no AI, visits every
reachable view. Both keep the manual approve+preview gate for now; autonomy
later is one marked line per flow.

---

## 1. Decisions (Anton, 2026-09-07)

1. **Replace, don't dual-write.** `RunWriter` becomes the only writer of live
   runs; the old ad-hoc writes (`run.json` turns, `NNN/meta.json` inside
   `rig.capture`, teardown `fused_cloud.npy`) are deleted, not shadowed.
   Old runs stay readable through the adapter — never rewritten.
2. **Ownership mirrors the trust split.** Supervisor threads own motion truth
   (`run.json`, `steps/**`, `session.json`); the brain thread owns `ai/**`
   through its own `AIRunWriter`. The brain cannot touch motion truth, the
   same way it cannot confirm a motion.
3. **One read door for the whole application: the `Run` object.** Built fresh
   on the schema — not a shim over `eyes/replay`. Replay and benchmarking
   stand on it later; `eyes/replay.py` dies.
4. **Flow B picks the true shortest path.** Measured (2026-09-07, kinematic
   sandbox): `plan_to_cell` over 25 reachable cells = 4.4 s total, median
   30 ms. Planning *every* remaining candidate each step and taking the
   cheapest is affordable — no proxy metrics.
5. **Flow B uses the same UI, reduced**: no AI panel, no Ask box. The
   data-collection flag is `RunRecord.source="data-engine"` (schema.py:271).

## 2. Supervisor decomposition

The state-machine core (dispatch, phases, generation counter, pose handoff,
preview join — the hard-won concurrency in `run/machine.py`) is **untouched**.
The refactor extracts seams around it:

```
run/app.py (composition root)
 ├─ RunWriter(outdir, name, object, source, config)   ← created before the arm
 ├─ Supervisor(rig, pub, writer, ...)                 ← core logic unchanged
 │   ├─ dispatch / phases / gen counter               (as today)
 │   ├─ _plan_worker / _exec_worker                   (motion only)
 │   └─ run/settle.py: capture→segment→plane-gate→fuse   ← extracted from
 │        _exec_worker; returns SettleResult; machine routes to writer + pub
 ├─ InspectionPublisher                               (UI seam, as today)
 └─ driver seat (exactly one):
      A: brain — make_ask_handler(sup, pub, ai_writer, cognition)
      B: sweep — run/collect.py driver over SupervisorMover
```

- `rig.capture` **returns data** and writes nothing; persistence has one owner.
- Deleted write sites: `Supervisor._record_turn`/`_save`, `_save_mask` and
  `save_chain`'s write halves (renders still produced — routed through the
  writer), `app.py` teardown `np.save`.

## 3. Write paths

| moment (today's site) | writer verb | record |
|---|---|---|
| app boot | `RunWriter.create` | `run.json{status:running}`, `config.json` |
| camera open | `writer.session` | `session.json` |
| question asked | `run.json.question` set; `AIRunWriter.open` | `ai/airun.json` |
| mover beats (`brain/live.py on_event`) | `event(kind)` | `events.jsonl` |
| settle: capture ok (`_exec_worker`) | `begin_step` + `write_capture` | `steps/NNN/` + `step.json phase:"captured"` |
| plane gate / capture crash | `mark_step(rejected\|failed, detail)` | pose-less step record |
| fuse ok (`_on_settle_done` path) | `write_fused(geometry, view_state)` | `step.json phase:"fused"` + `view_state.json` |
| stop / fault (`_on_exec_done`) | `event("stopped"\|"fault")` | `events.jsonl` |
| brain inspects | `AIRunWriter.transcript` | `ai/transcripts/tNNN.json` (keyed `step_id`) |
| menu rendered | `AIRunWriter.menu_def` (once/version) + `menu_input` (per turn) | `ai/menu/**` |
| answer() | `AIRunWriter.answer` | `ai/answer.json` |
| shutdown `finally` | fused arrays → `writer.close(status)` | `fused/`, `manifest.json`, final `run.json` |

Concurrency: Supervisor files and `ai/**` are disjoint by ownership.
`events.jsonl` is the one shared file — O_APPEND single-line writes from both
sides (POSIX-atomic at these sizes).

Every write passes `validate_for_write` at the moment it happens; a crash
leaves an unambiguous state (`phase:"captured"` = the crash window,
detect-on-read flags staleness).

`ViewState.decider`: `"ai"` (flow A), `"sweep-shortest"` (flow B),
`"operator"` (a redirect — the human drove).

## 4. The `Run` read object (`record/run.py`)

One scalable interface for the whole application (stories: A2, C1-C3, D1-D5,
E1-E3, F4). Read-only, lazy over files; files stay the truth.

```python
run = Run.load(run_dir)            # THE door; legacy adapted inside
                                   # (run.provenance == "legacy"), same API
run.record  run.session  run.config  run.events
run.steps -> [Step]      run.step(7)      run.survey      # survey = step 0
run.captured             run.at((3,0))                    # latest at address
step.record              step.view_state
step.rgb(upright=True)   step.depth()  step.cloud()  step.mask()
step.T_base_cam          step.dir                         # escape hatch
run.fused() -> (points, colors|None)
run.ai -> [AIRun]        airun.transcripts  airun.menu  airun.answer  airun.story()
run.refresh()            # pick up steps written mid-run (brain's live reads)
run.validate(deep=False) # same Report as the CLI
run.derived(method)      # fits/overlays for D4
```

Properties:
- **Lazy**: records validated eagerly (small JSON), binaries on demand.
- **One rotation policy**: `step.rgb(upright=True)` owns `rgb_rotation_deg` +
  the box-RAW-frame convention — today re-solved in `eyes/tools.upright`,
  `record/rr/workspace._frustum_image`, `record/show.py`.
- **Mid-run safe**: tolerates in-flight `phase:"captured"` steps; `refresh()`
  replaces `brain/live.refresh_views`' store surgery.
- **`run.ai` is plural from day one**: replays (E1) land as new AIRuns on the
  same run, original untouched.
- Not the writer, not the catalog scan (`Card` stays the cheap projection),
  not derivation compute.

Consumers ported: `ViewTools` (stays — it is the brain's *image verbs* — but
consumes a `Run`), `has_survey`, `SupervisorMover._captures`, `record/show.py`,
`record/story.py`. **`eyes/replay.py` is deleted.**

## 5. Flow A — live AI run, disk timeline

```
boot                run.json{status:running, source:"live", name, object}
                    config.json{git_sha, cell_yaml, calib, constants, models}
camera open         session.json
brain/ask "logo?"   question set; ai/airun.json; trace.jsonl as today
── one turn: move([3,0]) ─────────────────────────────────────────────
mover.request       events: requested
plan→preview        events: awaiting_approval
operator confirm    events: approved            (redirected/cancelled likewise)
settle              steps/001/{rgb.png, depth_aligned.npy, mask.png, chain_*.png}
                    step.json phase:"captured"
  plane gate fail → step.json outcome:"rejected", detail — run continues
fuse ok             step.json phase:"fused" + geometry
                    view_state.json{candidates, chosen:[3,0], decider:"ai"}
                    events: captured
brain inspects      ai/transcripts/tNNN.json (step_id:1) + menu_input
──────────────────────────────────────────────────────────────────────
answer()            ai/answer.json (+ evidence hunt records)
shutdown            fused/ → writer.close("completed") → manifest + run.json
Ctrl-C mid-run      close("aborted") — steps on disk are complete records
```

Launch: `p inspection/run/app.py run --outdir=… --name=cup-logo --object=cup`.

## 6. Flow B — data-collection sweep

Launch: `p inspection/run/app.py collect --name=box1 --object=box`
→ `RunWriter(source="data-engine", question=None)`.

Driver `run/collect.py` sits in the brain's seat (`SupervisorMover`) — same
grep-provable property: it may request, only a human confirms.

```
survey first (approved), then repeat:
  candidates = reachable ∧ unvisited ∧ ¬blocked
  plan to EVERY candidate (~4 s, shrinking), cost = joint-L1 path length
  next = argmin cost → mover.request(next) → preview → operator approves → settle
  plan refusal → blocked, next-cheapest
done when candidates empty → close("completed"); early exit → "aborted"
```

- Operator clicks a different cell mid-preview → redirect: their cell runs
  (`decider:"operator"`), sweep resumes from wherever the arm ends up.
- Per-step writes identical to Flow A minus `ai/**`.
- A3 (multi-object sessions): finish → `collect` again with new name/object;
  camera/session reuse is app-level, not writer-level.
- Autonomy later: the driver sends the confirm itself — one marked line,
  the same seam as the brain's future autonomy.

## 7. UI wiring

- Publisher retains a `run/meta` topic at boot: `{source, name, object,
  question}`.
- Frontend: `source == "data-engine"` → hide trace panel + Ask box; scene,
  preview, confirm buttons, chain panel identical.
- No new frontend state: mode is a projection of a retained topic, same as
  every other panel input.

## 8. Testing

1. **TDD per module** (superpowers workflow, as the record layer was built):
   writer verbs at machine hook points (mock rig, no hardware), sweep ordering
   (fake planner costs), `Run` parity old-vs-new run, settle-pipeline
   extraction (same events out, byte-identical step records).
2. **Mock rehearsal**: `ui/mock.py` keeps rehearsing the exact wiring
   (`make_ask_handler`'s one-implementation rule), now writing through
   `RunWriter` — a mock run validates like a real one.
3. **Headless playwright e2e, both flows** (Anton 2026-09-07): launch the
   mock app headless, drive the real frontend —
   - Flow A: ask → preview appears → click confirm → trace beats land →
     `validate_run` clean, `source:"live"`.
   - Flow B: collect mode → AI panel provably absent → approve sweep steps →
     `validate_run` clean, `source:"data-engine"`.
   - Screenshots kept as artifacts.

## 9. Build order

1. `Run` object + port readers (`ViewTools`, `has_survey`, `_captures`,
   show/story) — delete `eyes/replay.py`. Everything after stands on it.
2. Settle-pipeline extraction (`run/settle.py`) — no behavior change, tested
   by event/record parity.
3. Writer wiring into machine/rigs/app (flow A motion side) + `AIRunWriter`
   into brain/eyes write sites.
4. Flow B driver + `collect` entry + UI mode.
5. Playwright e2e over both flows.

## 10. Out of scope (deliberate)

- AI replay harness (E1-E3) — next branch; it consumes `Run`, designed for it.
- Autonomy (auto-confirm) — the seam is marked, not built.
- Migration of the 27 legacy runs — they stay adapter-read (F4).
- Catalog/API server, exports — later consumers per the stories doc.
