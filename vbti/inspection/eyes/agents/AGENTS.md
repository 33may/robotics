# eyes/agents — one loop, several specs

## Purpose
Every eyes-tier agent — a VLM with tools over ONE captured frame — runs the
same loop. A variant is DATA: rules text, a tool subset, an emit tuple, a
turn budget. The loop lives in `vlm_agent.py` once and no variant rebuilds it
(Anton 2026-08-27). Three exist:

- `inspect_agent.py` — reads a view against the run's question; emits the
  `view {saw, recommendation}` appendix the move menu renders.
- `survey_agent.py` — declares the object before `plan()`: identity,
  geometric primitive, per-surface orientation, visible features. Declares
  only; the planner infers (the tier split, 2026-08-27).
- `evidence_agent.py` — at answer time, hunts ONE named thing in one frame:
  locate → verify (text read verbatim or it is not found) → frame. "not
  found" is a successful hunt; a fresh miss bounces the orchestrator's
  answer (`brain/loop.py:_collect_evidence`).

## Files
- `vlm_agent.py` — `VlmAgent(rules, tool_names, emit, extras, max_turns)`
  and `run(tools, verbs, writer, model, task=..., cell=/view_rec=...) ->
  Finding`. The loop's invariants are documented in its module docstring and
  are properties of the loop — no spec can disable them.
- `inspect_agent.py` / `survey_agent.py` / `evidence_agent.py` — a rules
  string, a spec, normalisers, and a thin entry function with a stable
  signature. Nothing else.

Tests: `p inspection/tests/test_vlm_agent.py` (the loop and its structural
guarantees), `test_eyes_inspect.py`, `test_survey.py`, `test_evidence.py`
(one per variant).

## How to add a new agent variant — the flow

1. **Write the rules prompt first.** It IS the contract: what the agent
   does, in what order, what it may not do, and the exact reply shape
   `{"evidence": [...], "reasoning": "...", "answer": "...", <appendix>}`.
   State any vocabulary WITH its meaning (the degree-mapped phrases are the
   pattern: the number defines the word, the model says the word). Contracts
   are prompt text on both sides, leniently parsed, never strictly validated
   (Anton 2026-08-26).
2. **Decide the appendix.** Only add one if a consumer exists — every field
   needs exactly one consumer (`view.saw` → menu row, `view.recommendation`
   → orchestrator arithmetic, `evidence_image` → path resolution). Write a
   `normalise_<x>(raw)` that never raises and returns a harmless value for
   garbage: a malformed appendix must not discard a good answer.
3. **Declare the spec** next to the rules:
   `MY = VlmAgent(rules=MY_RULES, tool_names=..., emit=EMIT_CORE + ("x",),
   extras={"x": normalise_x}, max_turns=...)`. The constructor enforces at
   import time: evidence→reasoning→answer order, no `confidence`, appendices
   trail the answer, tools ⊆ the tier's five. Prune tools the variant does
   not need — removing distractor actions is the biggest measured lever on
   agent loops (AgentOccam, WebArena 16.5→25.8%).
4. **Write the thin entry function** — today's signatures are the pattern:
   `my_agent(tools, verbs, writer, model, <what varies per call>, on_turn=None)`
   calling `MY.run(...)`. Anything variant-specific that happens AFTER the
   run (the inspect exhausted-note, the evidence citation→path resolution)
   lives here, not in the loop: resolution that needs the transcript is
   wrapper work.
5. **Decide what context rides along.** `hypothesis` is a per-variant call:
   inspect gets the run hypothesis; the evidence hunter deliberately does
   NOT (confirmation pressure vs the honest miss). Same for `answer_schema`
   — only variants without a bespoke emit accept one.
6. **Wire the caller in `brain/loop.py`** with an `on_turn` closure that
   traces `sub_step` events (add a discriminator field if two variants can
   interleave at one cell — the hunt passes `hunt=find`).
7. **Add `tests/test_<variant>.py`**: the rules pin (the phrases/prohibitions
   a consumer relies on), the emit normalisation, the wrapper's resolution
   logic, and the graceful-degradation paths (missing input → None, garbage
   appendix → dropped). Run the whole suite — the loop tests must not change.

## Contracts & decisions
- **The loop is fixed; variants are data.** Rejected: inheritance with
  overridable methods (each variant regrows its own loop — the disease being
  cured) and free-form reply parsers (they could reorder fields the emit
  never declared).
- **The tier's verb table is closed.** `TOOLS` in `vlm_agent.py` is the whole
  surface; a spec chooses a subset, nothing can add a verb without editing
  the loop file. `view_at` is absent by construction: fetching another frame
  is moving, and this tier cannot move. Asserted by tests.
- **Every run persists through `FindingWriter`** — transcript verbatim on
  disk, `Finding.transcript_rel` points at it. Not configurable: there is no
  unaudited way to see.
- **Images are numbered as handed over** ("image 0 — full frame", "this is
  image N"). Always on: inert text for most variants, the citation mechanism
  for evidence — a model may echo a number it just read, never author a path.
- **Emit order is enforced at import time.** evidence→reasoning→answer, no
  confidence field, appendices trail. A broken spec fails the import, not a
  2am run.

## Does NOT belong here
- The orchestrator's use of a finding (menu rendering, bounce logic,
  answer.json assembly) — `brain/`.
- Verb implementations — `../verbs_local.py`; policy vs backend split holds.
- A variant that looks at more than one frame. That is a second decider;
  the orchestrator owns cross-view reasoning, always.
