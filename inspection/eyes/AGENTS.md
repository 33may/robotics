# eyes — image machinery (MAY-186)

## Purpose
The run structure and tool API an AI agent uses to navigate our geometric
space, inspect captured views, and know what it has seen. Machinery, not a
question-answerer: **the module owns no question** — the agent does. Design:
`../2026-08-20-image-machinery-design.md`; plan:
`docs/superpowers/plans/2026-08-24-image-machinery.md`.

Two agent tiers. The **orchestrator** owns geometry, the run structure and
the final answer; the **inspection subagent** reads one captured view with
its own tools and returns text. That split is what keeps raw pixels from
accumulating in the orchestrator's context over a 5–30 step run.

## Files
- `store.py` — the run structure. `RunStore` (read API + flush to
  `<run>/eyes/store.json` on every write) and the three writers.
  `ViewRecord(cell, pose_id, cap_dir, t, T_base_cam)`.
- `tools.py` — the verb surface: `ViewTools(store, writer=None)` with
  `view_at` (newest view of a cell), `views_near` (4-connected; an uncaptured
  neighbour returns None and is logged as a note), `get_view` → `ViewImage`
  (pixels **and** text), `crop` (clipped), `coverage` (ASCII + count), `note`.
  Debug CLI: `p inspection/eyes/tools.py inspection/data/runs/2408-seeded`.
- `verbs_local.py` — the subagent's local verbs: `LocalVerbs(backend)` with
  `detect` (text → boxes), `segment` (box → frame-sized mask), `read_text`.
  Policy (thresholds, clipping, ordering) lives in `LocalVerbs` and is tested
  on `StubBackend` with no GPU; `Sam3Backend` and `PaddleOcrBackend` hold only
  model I/O. Measured on the 5090: SAM 3 63–74 ms/frame warm, PP-OCRv6 ~16 ms.
  `p inspection/eyes/verbs_local.py <run_dir> <h> <v>`.
- `env_check.py` — preflight for the model stack (sm_120, a real bf16 matmul,
  transformers ≥5, HF token, and a genuine gated-file fetch for `facebook/sam3`).
- `replay.py` — `load_run(run_dir, v_elevs, h_bins) -> RunStore`: a captured
  data-engine run becomes the store's substrate. Idempotent.
  Debug CLI: `p inspection/eyes/replay.py inspection/data/runs/2108-d`.

Tests (no pytest): `p inspection/tests/test_eyes_store.py`,
`p inspection/tests/test_eyes_replay.py`.

## Contracts & decisions
- **Three trust levels, enforced by construction.** `FactWriter` is the only
  writer of geometric ground truth (views, visited, coverage) — deterministic
  code holds it. `PlanWriter` (plan/hypothesis/notes) goes to the
  orchestrator, `FindingWriter` (findings, verbatim transcripts, notes) to the
  inspection subagent. Neither model-facing writer *has* an `add_view`; a test
  locks their public API to an exact set. A hallucinated "I have seen the far
  side" would corrupt the one signal independent of the VLM.
- **Never imports `inspection.motion` or `inspection.run`** — the image tier
  does not move the robot, in any form. Asserted by a test, not by convention.
  Grid dimensions are passed in as parameters rather than imported from
  `view/viewsphere.py` (which pulls in the whole pinocchio/IK stack).
- **Capture↔cell join** (`run/machine.py:516,570`): dir `000` = survey
  (`cell is None`); dir `NNN` has `meta.json:pose_id == N`, whose cell is the
  N-th **non-survey** turn of `run.json:turns[]`. Turn `step` and `pose_id`
  are deliberately different namespaces — do not conflate. Failed captures
  consume an ordinal but leave no dir; the loader walks dirs, so disk is truth.
- **Verbatim on disk, distilled in context.** Full subagent transcripts land
  in `<run>/eyes/transcripts/`; only summaries return to the orchestrator.
  You can always drop detail later, never recover it.
- **Never de-rotate a capture before detection** (measured 2026-08-24). SAM 3
  resizes inputs to a square 1008, so turning a landscape frame into a portrait
  one changes the object's rendered scale: losslessly straightening the ±90°
  captures made 5 of 7 detections *worse*, two by ~0.70. The model tolerates
  rotation, not aspect change. `upright()` corrects only the 180° flip.
- **SAM 3 always returns 200 queries** — the score threshold is the detector.
- **Answering is not a vote.** One positive view settles a yes; coverage is a
  supporting argument for *absence* only. Naive tallying across views degrades
  presence detection (PInVerify: 0.652 → 0.592) — do not add a scoring formula.
- No pruning/summarisation in v1. Images used as captured, 848×480.

## Does NOT belong here
- Robot motion, IK, reachability, capture (`motion/`, `view/`, `perception/`).
- Question-specific logic (the demo's "is there a logo" lives in the agent).
- World-frame spatial reasoning inside the inspection subagent — it reasons
  camera-centric; the orchestrator, which knows the pose, translates.
- The next-move decision itself — that is the decider (MAY-187, `run/decider.py`).
