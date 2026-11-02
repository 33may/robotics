# record — the run recording contract

## Purpose
Every run (live inspection or data sweep) is written to one schema and read
back through one object. Files are the truth: no database, no second write
path. Design: `docs/blueprints/data-engine/` (`schema-draft.md`,
`schema-review.md`, `flows-design.md`); generated reference:
`docs/blueprints/data-engine/generated/schema.md`.

## Files
- `schema.py`: pydantic models that ARE the spec: `RunRecord`, `StepRecord`,
  `ViewState`, `DecompositionRecord`, `SessionRecord`, `ConfigSnapshot`,
  `AIRunRecord`, `MenuDef`, `MenuInput`, `TranscriptRecord`, `AnswerRecord`,
  `OperatorEvent`, `Manifest`, `DerivationMeta`; `validate_for_write`. The
  module docstring holds the on-disk layout.
- `writer.py`: `RunWriter`, the single writer for run-level and step files:
  `create`, `begin_step` (dense ids, survey = 0), `write_capture`,
  `write_fused`, `write_decomposition`, `mark_step`, `event`, `close`.
  `write_step_decomposition` is the shared path for live fit and backfill.
- `ai_writer.py`: `AIRunWriter`, the planner thread's own writer, only under
  `ai/<seq>/` (airun.json, transcripts/, menu/, answer.json); `event` appends
  to the shared `events.jsonl`.
- `run.py`: `Run`, the one read object: `Run.load`, `steps`, `survey`,
  `captured`, `at(address)`, `fused`, `ai`, `validate`, `derived`; `Step`
  (`rgb`, `depth`, `mask`, `cloud`, `T_base_cam`, `decomposition`);
  `AIRun`. Lazy for binaries, read-only.
- `legacy.py`: adapt-on-read for pre-schema runs (`NNN/meta.json` +
  `run.json:turns`): in-memory records with `provenance="legacy"` and
  fabricated fields listed in `synthesized`. Never writes.
- `validate.py`: `validate_run` / `validate_archive`: per-file schema,
  cross-file refs, manifest bytes (`deep=True` re-hashes), detect-on-read
  crash rule (`running` + no writes for `STALE_S`).
- `catalog.py`: `runs`, `card`: cards and filters over `run.json` files.
- `derive.py`: `run_derivation`: idempotent derivations under
  `derived/<run>/<method>[/<variant>]`, skip-if-fresh by source hash, explicit
  per-step failure entries.
- `migrate.py`: `Migration`, `migrate_run`, `migrate_archive`: the only legal
  writer of a closed run. Add-only, idempotent, recorded in `run.json`
  `migrations`. Dry-run first.
- `story.py`: the AI narrative of a run, read from disk in order.
- `show.py`: reduced two-pane Rerun viewer from run files alone.
- `docgen.py`: regenerates `generated/schema.md` from the models.
- `__main__.py`: CLI: `python -m inspection.record ls|card|validate|show|story`.
  `ls` reads `data/runs/`, `ls --data` `data/datasets/`, `--all` both.
- `rr/`: Rerun review workspaces: `log.py` (the only module that calls
  Rerun; owns entity paths, colours, blueprint), `workspace.py` (builds
  `data/derived/<run>/rrd/<run>.rrd`), `view.py` (list/open).

## Contracts & invariants
- **One writer per subtree.** `RunWriter` owns the run root and `steps/`;
  `AIRunWriter` owns `ai/<seq>/`. Capture code writes only bytes into the step
  dir `begin_step` hands out. Everything else is read-only on runs, except `migrate.py`.
- Every JSON write: `validate_for_write` (refuses unknown fields) → temp file in
  the same dir → `os.replace()`. Reads are tolerant (unknown fields ignored).
- Versioning is per model (`VERSION`); a reader refuses a `schema_version`
  newer than it knows.
- **A step is the join key.** `steps/NNN/step.json` carries its view address
  (`null` for the survey, step 0). Ids are dense over steps that happened; a
  failed capture keeps its id with an outcome and detail, never a gap.
- The manifest hashes immutable binaries only; JSON is checked by re-validation.
- `config.json` records the resolved `T_flange_cam_*.npy` file + sha256, not "latest".
- `data/runs/` holds inspection runs, `data/datasets/` collection sweeps; both
  share the schema.

## Dependencies
`run.py` imports `logic/cell/geometry.py:rotate180` (upright images);
`rr/workspace.py` lazily imports `logic/cell/geometry.py`. Nothing else outside `logic/record/`.

## Does NOT belong here
- Deciding what to record or when (`logic/run/machine.py`, `logic/brain/`).
- Fitting or geometry computation (`logic/fit/`, `logic/cell/`); this module stores results.
- Any server database.
