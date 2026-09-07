# Recording Management — user stories (draft for discussion)

Status: DRAFT v0, 2026-09-03. Step 3 of the data-engine plan: the management
system over ALL recordings (live-loop + data-engine). Stories first — storage/
indexing/retrieval design only after these are agreed.

Actor is Anton unless stated. Priority tags follow the consumer ranking:
**[now]** = algorithm-dev loop · **[soon]** = debugging/tracing · **[later]** =
eval/training.

---

## A. Capture campaigns (the data engine itself)

- **A1 [now]** I start a capture run giving `name` + `object` (+ optional
  question); the system stamps `source=data-engine`, config snapshot, and I see
  survey + 3D preview and approve each step. A finished run validates and closes
  (manifest, status=completed).
- **A2 [now]** I can abort at any point and the run is still a valid, marked
  (`aborted`) partial record — per-step artifacts written online survive.
- **A3 [now]** I capture several objects in one session with minimal friction:
  finish → new name/object → go. No config re-entry; camera/session reused.
- **A4 [soon]** A crashed run is detectable as `crashed` (not silently
  indistinguishable from running/aborted).

## B. Browse & find

- **B1 [now]** I list all runs with a summary card each: name, object, date,
  source, status, #steps, size, question, verdict-if-any.
- **B2 [now]** I filter by object / source / status / date / view-method
  ("all completed cup runs from the data engine").
- **B3 [now]** The catalog is AI-queryable: an agent (you) can answer "which
  runs have >20 views and a completed fused cloud?" without me writing queries.
- **B4 [soon]** From a card I jump straight to: run dir, Rerun view, trace view.

## C. Inspect one run

- **C1 [now]** I open any run in Rerun: per-step clouds (colored), fused cloud,
  camera frustums at `T_base_cam`, step images — no robot needed.
- **C2 [now]** I verify a run's integrity on demand: manifest re-hash + schema
  validation + sibling-file uniformity, one command, clear verdict.
- **C3 [soon]** I read the AI story of a run in order: plan → per-step
  reasoning → approvals → answer, rendered readable from trace/transcripts.

## D. Algorithm development loop (consumer #1)

- **D1 [now]** I select a run set ("all cups", "these 5 ids") and run a
  deconstruction method across it; outputs land in `derived/<run>/<method>/`
  keyed (run, method[, variant]) with provenance (code version, source hashes).
- **D2 [now]** Re-running skips up-to-date runs and recomputes stale/missing
  ones (source-hash comparison) — no `_2` dirs, no manual bookkeeping.
- **D3 [now]** I compare results across runs and methods in one table:
  scores/metrics per (run, method, variant) queryable together.
- **D4 [now]** I overlay fits/primitives against the source cloud in Rerun for
  any (run, method) pair.
- **D5 [now]** A per-step processing failure (like cup5's silent step-029 drop)
  is recorded as an explicit failure entry, never a silent gap.

## E. Replay & experiments

- **E1 [soon]** I rerun the AI over a captured run (`mode=replay`) with a
  different menu / model / prompt and it lands as a NEW AIRun on the same run —
  original untouched.
- **E2 [soon]** I compare AIRuns side by side: same evidence, different
  menu/model → different moves/answers, diffable.
- **E3 [soon]** Exact replay: re-inject recorded rendered text verbatim;
  determinism check hash-compares a re-render against the record.

## F. Curation & health

- **F1 [now]** Archive-wide health check: every run validated (schema, manifest,
  uniformity), report of violations — runnable any time, cheap.
- **F2 [now]** Junk/debug runs are excluded from the working set without
  deleting bytes (status/kind filter, or an exclusion list in code — never
  moving/renaming).
- **F3 [soon]** Schema migrations run archive-wide through the gate (add-only,
  backfill-or-declared-missing, recorded) with a dry-run report first.
- **F4 [soon]** Legacy runs (the 27 pre-schema ones) become readable through
  the same catalog via a legacy adapter — read-only, no rewriting.

## G. Evaluation (later consumers)

- **G1 [later]** I attach ground-truth labels to runs/steps (human trust class,
  separate from AI outputs) and compute verdict-vs-label metrics across a set.
- **G2 [later]** I define an eval set (list of run ids + questions) and run the
  system against it, getting a scorecard.
- **G3 [later]** I export a selected run set to a training format (LeRobot/…)
  under `exports/` — a projection, rebuildable, never hand-patched.

---

## Functional requirements distilled (v0)

1. **Catalog** — rebuildable index over run datasheets; files are the truth
   (DuckDB/jq first, SQLite only when slow). Answers B1-B3.
2. **Run lifecycle ops** — create/close/abort/crash-mark + validation at close.
   Answers A1-A4.
3. **Integrity ops** — validate one run / whole archive. Answers C2, F1.
4. **Derived-tier runner** — (run, method, variant) addressing, provenance,
   staleness detection, failure records. Answers D1-D5.
5. **Viewers** — Rerun projection of any run (+ derived overlays); trace
   reader. Answers C1, C3, D4.
6. **Replay harness** — AIRun replays with menu/model substitution. E1-E3.
7. **Migration runner** — the gate, archive-wide. F3.
8. **Legacy adapter** — old 27 runs readable, never rewritten. F4.

Non-functional: files-first, single machine, no server processes; every op
available to an AI agent (CLI + python API); read paths never mutate runs.
