# brain — the planner tier

## Purpose
The blind planner. It owns the question, the geometry, the plan and the answer;
it never sees a pixel. All visual knowledge reaches it as text from `eyes/`.
Runtime is the **Claude Agent SDK**: we adopt its loop and build only the
harness (tool bodies, safety gate, context render, evidence ledger).

    survey -> plan() -> { inspect(question, cell) | move(cell) }* -> answer()

## Files
- `loop.py`: `Brain(run_dir, question, vlm, verbs, ..., mover=None)`
  (models injected, never defaulted; `eyes/cognition.py` assembles them): the
  four verbs as in-process MCP tools, the `PostToolUse` state hook
  (`_staple_state`: STATE + MOVES every turn), the `PreToolUse` gate on
  `move` (`_gate`), the survey declaration before the opening, answer-time
  evidence hunts with bounce-on-miss (`_collect_evidence`), the system prompt
  and the turn budget. `mover=None` is replay over captured cells.
  CLI: `python loop.py <run_dir> "<question>"`.
- `live.py`: live binding: `SupervisorMover` sends `view/request` on the same
  queue as the UI buttons and polls the `Supervisor` phase; plus the UI
  handlers (`make_event_sink`, `make_beat_handler`, `make_ask_handler`) and
  `TraceWatcher`.
- `render.py`: everything the planner reads that a model did not write:
  `state_block`, `moves_block`, `survey_bearing_line`, `gloss`, `opening`,
  `menu_def`.
- `trace.py`: `TraceWriter`/`read_trace`: `<run>/ai/<seq>/trace.jsonl`,
  append-only, one per planner session. Image paths are run-relative; the UI
  publisher maps them to URLs.

## Contracts & decisions
- **The planner is blind.** No images, ever. The description is the world
  model: what the vision tier does not say is invisible here. Description
  quality, not planner model choice, is the dominant capability lever.
- **Append-only context.** The transcript accumulates; the SDK loop stays
  intact. Safe because any captured view can be re-inspected for zero motion.
  Re-reading does not fix a reinforced wrong hypothesis, hence adaptive
  thinking on by default.
- **Content accumulates, state is recomputed.** Position/coverage/menu are
  derived every turn and pushed onto tool results by hook: a push cannot be
  forgotten, a pull can.
- **Deltas, never snapshots.** "4/12 -> 5/12" stays true; a coverage table
  is false one view later.
- **Four verbs (`plan`, `inspect`, `move`, `answer`), and that is a budget.**
  Pruning the action set is the largest measured lever on agent loops
  (AgentOccam: WebArena 16.5 -> 25.8%). A fifth verb must earn its place.
- **`move` = deterministic robot code + the inspect call.** It returns exactly
  what `inspect` returns. Position and menu arrive by hook.
- **`inspect` = the inspection model alone** on any held view, no motion.
  Both verbs go through `Brain._inspect_cell`, so every description is a
  `FindingWriter` entry with a transcript on disk.
- **`answer_schema` constrains readout shape by prompt text only**:
  `models.respond`'s schema argument only flips Gemini into JSON mode. Field
  ORDER is fixed (evidence, reasoning, answer).
- **Failures come back categorical, not verbatim** (LLM3 `2403.11552`).
- **The full finding crosses back**, not a one-line summary. Transcripts stay on disk.
- **Answer policy is judgment, not arithmetic.** One clear view settles a yes;
  a no is a coverage argument. No vote counting, no confidence numbers.
- **No `undetermined()` verb.** An offered escape hatch gets over-used.
  Budget exhaustion still yields that verdict, as a harness outcome.
- **Answer is stored before the run ends** (`_commit_answer`): a schema
  rejection fails the tool call and the run continues.
- **Approval gate.** In live mode the brain may press `view/request`; only the
  operator presses `view/confirm` (`Supervisor._on_view_confirm` drops
  confirms unless previewing). Exception: `SupervisorMover(auto_confirm_s=...)`
  for `collect --auto` data sweeps with an operator at the cell. A stale
  request cannot move the arm (`Supervisor.gen`). `_gate` validates the cell /
  face address in code before `move` runs.

## Does NOT belong here
- Pixels, crops, detection, OCR: `eyes/`, behind `inspect`.
- Camera-centric language. Subagents speak in-frame; this tier speaks
  azimuth/elevation because it knows the pose.
- Motion or its own motion path: live moves go through the `Supervisor` queue.
- Post-run validation and reflection: `answer.json` carries the evidence a
  later validator needs to disagree.
