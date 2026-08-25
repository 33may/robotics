# brain — the orchestrator tier (MAY-187)

## Purpose
The blind planner. It owns the question, the geometry, the plan and the answer;
it never sees a pixel. All visual knowledge reaches it as text from the `eyes`
tier. Runtime is the **Claude Agent SDK** — we adopt the loop and build only
the harness (tool bodies, safety gate, context render, evidence ledger).

Loop shape (Anton, 2026-08-25):

    survey -> plan() -> { inspect(question, cell) | move(cell) }* -> answer()

## Files
- `loop.py` — `Brain(run_dir, question, vlm)`: the four verbs as in-process MCP
  tools, the `PostToolUse` state hook, the `PreToolUse` safety gate, the system
  prompt and the run budget. CLI:
  `p inspection/brain/loop.py inspection/data/runs/2408-cup1 "is there a logo on the cup?"`
- `render.py` — everything the planner reads that a model did not write:
  `state_delta`, `action_menu`, `gloss`, `opening`.

## Contracts & decisions
- **The orchestrator is blind** (Anton 2026-08-25). No images, ever. The
  description is therefore the world model: what the vision tier does not say
  is permanently invisible here. Description quality, not planner model choice,
  is the dominant capability lever.
- **Append-only context, not stateless re-render** (Anton 2026-08-25). The
  transcript accumulates; we keep the SDK loop intact. This is safe *because
  the store is our filesystem* — any captured view can be re-inspected at any
  time for zero motion, so stale context is always recoverable. What
  re-reading does NOT fix is a wrong hypothesis being reinforced turn after
  turn, which is why `thinking: adaptive` is on by default.
- **Content accumulates, state is recomputed.** Descriptions ride the
  transcript. Position/coverage/menu are derived from the store every turn and
  pushed onto tool results by hook — a push cannot be forgotten, a pull can.
- **Deltas, never snapshots.** A coverage table is false one view later; "4/12
  -> 5/12" stays true forever. Nothing that rots goes into an append-only log.
- **Four verbs, and that is a budget.** Pruning the action set is the largest
  measured lever on agent loops (AgentOccam: WebArena 16.5 -> 25.8% from
  removal alone). A fifth verb must earn its place on the bench.
- **`move` is deterministic robot code plus the inspect call itself** (Anton
  2026-08-25): go to the cell, capture the frame, inspect it against the run's
  question. It returns **exactly what `inspect` returns** — it IS `inspect`
  with a motion in front of it. The planner is blind, so a move that handed
  back only a pose would waste a turn. Position and menu are state and arrive
  by hook; a verb never renders its own context.
- **`inspect` is the inspection model alone** — same looking, no motion, and a
  question of the planner's choosing on any view already held. Chasing a detail
  through held views is therefore strictly cheaper than moving. Both verbs go
  through `Brain._inspect_cell`, so every description in a run is a
  FindingWriter entry with a transcript on disk; there is no unaudited way to
  see.
- **`answer_schema` is the planner constraining the readout shape** — "yes|no",
  "the text, verbatim", "a count". It is enforced by prompt text and nothing
  else: `models.respond`'s schema argument is only a truthiness flag that flips
  Gemini into JSON mode (`models.py:111`). A caller may constrain what the
  answer contains, never the field ORDER — evidence, reasoning, answer is fixed,
  because answer-first erases 100% of the CoT gain.
- **Failures come back categorical, not verbatim** (LLM3 `2403.11552`: 60% vs
  40% success *and* fewer retries).
- **The full finding crosses back, not the one-line summary** — Anton
  2026-08-25 overrode "distilled in context" for v1; Opus carries the tokens
  and OpenEQA's one datapoint favours raw (36.5 captions < 43.6 < 49.6 raw).
  Transcripts still stay on disk.
- **Answer policy is judgment, not arithmetic.** One clear view settles a yes;
  a no is a coverage argument. No vote counting, no confidence numbers,
  reasoning before verdict.
- **No `undetermined()` escape hatch for the model** — held back deliberately
  (`action-space-absolute-look-hv`, Anton 2026-08-12, reaffirmed 2026-08-25):
  GPT-4 wrongly declared 54.9% of feasible WebArena tasks impossible, so an
  offered hatch gets over-used. The prompt does not invite "cannot determine".
  Budget exhaustion still produces that verdict, but as a HARNESS outcome the
  model cannot invoke — a loop must be able to stop.
- **Replay first.** `move` jumps to cells a data-engine sweep already captured,
  so the loop runs and benches with no arm. The live executor binds at exactly
  one seam: `Brain._gate`, where `inspection.safety.config_is_safe` /
  `path_is_safe` run before any real motion.

## Does NOT belong here
- Pixels, crops, detection, OCR — all of that is `eyes/`, behind `inspect`.
- Camera-centric language. The subagent speaks in-frame; this tier speaks
  azimuth/elevation, because it is the tier that knows the pose.
- Post-run validation and reflection. Stubs by design (Anton 2026-08-25):
  `answer()` writes `eyes/answer.json` with the evidence a later validator
  needs to disagree with it.
