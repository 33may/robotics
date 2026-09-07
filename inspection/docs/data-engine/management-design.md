# Recording Management — storage / indexing / retrieval design (draft)

Status: DRAFT v0, 2026-09-03. Against the agreed user stories
(`management-user-stories.md`); contract = `inspection/record/schema.py` (v1).
Canon applied: files are the truth, indexes disposable, no server processes.

---

## 1. Catalog (stories B1-B3) — no database until it hurts

`run.json` IS the catalog item (STAC pattern: small descriptive record beside
the bytes). The catalog layer is a **query API over the files**, not a store:

- Engine: DuckDB over globs —
  `SELECT ... FROM read_json_auto('runs/*/run.json')`, joined with
  `manifest.json` for sizes. Always fresh, zero index maintenance, good to
  thousands of runs.
- Surface: `record/catalog.py` — `runs(filter...)`, `card(id)`, typed results;
  a thin CLI on top. AI-queryable = the same API + raw SQL passthrough (B3).
- Escalation path (only when slow): rebuildable single-file SQLite built FROM
  the same run.jsons. Never a server DB.
- Fan-out guard (critic): everything B2 filters on lives in run.json
  (object, source, rig, status, tags, question, steps roster, view_methods);
  step-level queries are explicit opt-ins that glob `steps/*/step.json`.

## 2. Run lifecycle (A1-A4) — one writer library, two callers

`record/writer.py` — used by BOTH the live loop and the data-engine runner
(source stamped by caller):

- `RunWriter.create(dir, datasheet, config, session?)` → writes run.json,
  config.json, session.json; status=running.
- `write_step(...)` two-phase per schema: phase="captured" at capture,
  phase="fused" after geometry — each write temp-in-same-dir → `os.replace()`.
- `append_event(...)` → events.jsonl (single appender).
- `close(status)` → final run.json, manifest (binaries: sha256+bytes+mtime),
  full `validate_for_write` pass. A run is valid-at-close or marked.
- Crash handling: see OPEN-1.

## 3. Integrity (C2, F1) — `record/validate.py`

`validate_run(dir, deep=False)` — the Bucket-C engine:
- per-file schema validation (read models)
- cross-refs: `step.view.method` resolves in `view_methods`; steps roster ↔
  step dirs; transcript/answer/menu-hash references resolve;
  `completed ⇒ answer.json`
- manifest: coverage of binaries; cheap pre-pass (bytes+mtime), `deep=True`
  re-hashes
- sibling uniformity per outcome class (captured steps carry the full file set)
Archive health (F1) = the same check looped, one report.

## 4. Derived tier (D1-D5) — helper library, not framework

Layout stays `derived/<run>/<method>[/<variant>]/` + per-derivation
`meta.json`: `{version: "<semver>+p:<code-hash>", params, source_hashes,
steps: {<step_id>: ok|failed(reason)}}` — the scores.jsonl pattern made
uniform; explicit per-step failure entries (D5).
`record/derive.py`: addressing, provenance stamping, staleness check
(recorded source hashes vs manifest), skip-or-recompute (D2). Fit methods
stay plain functions that call it — no plugin framework.

## 5. Viewers (C1, C3, D4) — build items

- `record/show.py` (or investigation/rr growth): schema'd run → Rerun
  (per-step clouds, fused, frustums from session intrinsics + T_base_cam,
  images; derived overlays for D4).
- Trace reader (C3) — later, reads trace.jsonl + transcripts by reference.

## 6. Replay harness (E1-E3) — later, on the AI tier

New AIRun dir with mode=replay; menu substitution via MenuDef + MenuInput;
exact replay injects recorded rendered text. Touches brain/loop.py — build
phase of its own.

## 7. Legacy adapter (F4) — see OPEN-2

27 pre-schema runs readable through the same catalog/validate/show surface,
`provenance="legacy"` + `synthesized=[...]` per schema. Never writes into
`runs/`.

## 8. Migration runner (F3) — later

Registered scripts, add-only, backfill-or-declared-missing, recorded in run
metadata; dry-run report first. Design exists in the skill; build when the
first real migration arrives.

---

## Resolved 2026-09-03 (Anton)

- **Crash detection: detect-on-read** — catalog/validate flag `running` runs
  with stale mtimes as crashed; no daemon, no writer burden.
- **Legacy runs: adapt-on-read** — read-only in-memory conversion with
  synthesized fields declared; no new bytes ever.
- **Package home: grow `inspection/record/`** — schema + writer + catalog +
  validate + derive, one import surface.

Step 4 (API layer) then = the public surface of writer/catalog/validate/derive
+ CLI, end-to-end tested against a synthetic run + a real one.
