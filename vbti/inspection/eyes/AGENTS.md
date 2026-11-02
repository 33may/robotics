# eyes — image machinery

## Purpose
The tools a VLM agent uses to read captured views and know what it has seen.
Machinery, not a question-answerer: **the module owns no question**; the
planner (`brain/`) does. Design: `docs/blueprints/image-machinery.md`.

Two tiers. The planner owns geometry, the run structure and the final answer;
the VLM subagents (`agents/`) read one captured view with their own tools and
return text. That split keeps raw pixels out of the planner's context.

## Files
- `agents/`: the VLM subagents: one loop (`vlm_agent.py`), one spec per
  variant (inspect / survey / evidence). Recipe for a new variant: `agents/AGENTS.md`.
- `tools.py`: the verb surface: `ViewTools(run, writer=None)` over a
  `record/run.py:Run`: `view_at` (newest view of a cell), `views_near`
  (4-connected; an uncaptured neighbour returns None and is noted), `get_view`
  → `ViewImage` (pixels **and** deterministic text), `crop`, `coverage`
  (ASCII + count), `note`, `survey_bearing`. `upright()` corrects only a 180°
  capture. Debug CLI: `python tools.py <run_dir>`.
- `store.py`: mutable AI state: `RunStore` (flushes `store.json` on every
  write) and the writers `FactWriter`, `PlanWriter`, `FindingWriter`. Views
  are read from `Run`, not from here.
- `verbs_local.py`: local verbs: `LocalVerbs(backend)` with `detect` (text →
  boxes), `segment` (box → frame-sized mask), `read_text`. Policy lives in
  `LocalVerbs`, tested on `StubBackend` (no GPU); `Sam3Backend`,
  `PaddleOcrBackend`, `SamOcrBackend` hold only model I/O. On the 5090: SAM 3
  63–74 ms/frame warm, PP-OCR ~16 ms.
- `models.py`: the swappable VLM interface `respond(parts, schema)`:
  `GeminiVlm` (default `gemini-robotics-er-2-preview`), `StubVlm`,
  `first_json_object`. No retries/batching/caching here.
- `cognition.py`: `real_cognition()` / `stub_cognition()`: the caller picks
  explicitly; nothing falls back to a stub. A live run that cannot see fails
  at boot, and a stub run is labelled in the trace.
- `preview.py`: draws detect boxes / masks / OCR onto captures
  (`<run>/eyes/preview/<cell>.png`) for eyeball checks.
- `env_check.py`: model-stack preflight (sm_120, real bf16 matmul,
  transformers ≥5, HF token, gated fetch of `facebook/sam3`).
- `bench/labels.py`: per-capture human ground truth (yes / no / partial)
  in a separate file beside the run.

Debug a run without code: `python -m inspection.record card <id>`, or
`Run.load(run_dir)` from a REPL. Tests: `tests/test_eyes_*.py`, `tests/test_vlm_agent.py`.

## Contracts & decisions
- **Three trust levels, enforced by construction.** `FactWriter` is the only
  writer of geometric facts (views, visited, coverage) and is held by
  deterministic code. `PlanWriter` (plan/hypothesis/notes) goes to the planner,
  `FindingWriter` (findings, verbatim transcripts, notes) to the subagent.
  Neither model-facing writer has `add_view`; a test locks their public API.
  A hallucinated "I have seen the far side" must not reach the coverage signal.
- **Never imports `inspection.motion` or `inspection.run`**: the image tier
  never moves the robot. Asserted by tests (`test_eyes_store.py`,
  `test_eyes_inspect.py`, `test_vlm_agent.py`). Grid vocabulary comes from
  `view/grid.py` (numpy only), never `view/viewsphere.py`/`rays.py`.
- **A step IS the capture↔cell pair.** `steps/NNN/step.json` carries its
  `view.address` (`null` for the survey, always step 0); `Run.at(cell)` is the
  only lookup. Legacy layouts are adapted in `record/legacy.py`.
- **Verbatim on disk, distilled in context.** Full subagent transcripts go to
  `<run>/ai/<seq>/transcripts/tNNN.json`; only summaries return to the planner.
- **Never de-rotate a capture before detection.** SAM 3 resizes to a square
  1008, so landscape→portrait changes the object's scale: straightening ±90°
  captures made 5 of 7 detections worse. Only the 180° flip is corrected.
- **SAM 3 always returns 200 queries**: the score threshold is the detector.
- **Answering is not a vote.** One positive view settles a yes; coverage
  supports *absence* only. Naive tallying degraded presence detection
  (0.652 → 0.592); do not add a scoring formula.

## Does NOT belong here
- Robot motion, IK, reachability, capture (`motion/`, `view/`, `perception/`).
- Question-specific logic (lives in the agent prompt / planner).
- World-frame reasoning inside a subagent: it reasons camera-centric; the
  planner, which knows the pose, translates.
- The next-move decision (`brain/`).
