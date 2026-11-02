# Single Recording — entity map draft (design rationale)

Status: SUPERSEDED for structure, 2026-09-03 — the binding spec is
`inspection/record/schema.py` and the generated reference is
`generated/schema.md` (regenerate: `p inspection/record/docgen.py`).
This file remains only as the record of the design conversation and rationale.

Source: Anton's user story (2026-09-03) + `loop-data-audit.md`.

---

## Decisions already made (Anton)

- **Run identity**: user gives `name` + optional `object` tag (free string: box, cup,
  tube…; AI queries it later). `source` = `live` | `data-engine`, stamped
  programmatically by the process that runs it.
- **Motion**: do NOT store trajectories/planner internals. Only the waypoints the robot
  *is at* (capture poses). Movement is recomputable; we only care it was safe & doable —
  which the config snapshot makes verifiable.
- **Atomic step** = one robot position. THE join key: every artifact carries `step_id`.
  Several AI runs/reasoning rounds may attach to one step. Survey = step 0.
- **Config/calib snapshot per run**: yes.
- **Run datasheet**: question at start, answer at completion, outcome status.
- **AI subagent runs**: store them, ONCE (no 4× duplication). Crops keyed so they can't
  collide.
- **Menu** (the programmatically injected context/moves block): a separate, versioned
  object — enables menu experiments and cross-run queries.

## Resolved 2026-09-03 (Anton)

- **Per-step geometry: ONLINE** — written during the run at the fusing phase (data
  already in memory, crash-safe); replay remains the repair path.
- **Segmentation record: trimmed** — `{score, box, px}` only (today's fields).
  Model identity stays run-level in `config.json`, not per step.
- **Menu: snapshot + shared id** — content embedded in the run (self-contained) AND
  `menu_id` + hash for cross-run queries.
- **Views decoupled from the viewsphere** — `step.view = {method, address}`;
  run declares `view_methods` (viewsphere is one method among future ones); physical
  truth (`T_base_cam`) is method-invariant.
- **Per-step ViewState snapshot** — the dynamic candidate-view set recorded at every
  step (addresses/poses/status), because geometry-driven methods recompute it each
  step. It is the menu's declared input.
- **Menu = pure function of ViewState** — record input (ViewState) + output (trace
  `state` text) + definition; exact replay injects recorded text, menu experiments
  re-render the same ViewState with a different menu.

---

## Entities

```
Run 1───1 Config          (snapshot at start)
 │  1───1 Session         (camera identity + intrinsics — exists today)
 │  1───1 Answer          (0..1 — completion artifact)
 │  1───1 Manifest        (close ritual)
 │  1───N Step            (atomic; ordered; survey = 000)
 │         │ 1───1 Capture      (rgb, depth×2, ir×2, rotation flag)
 │         │ 1───1 Segmentation (mask + score/box/px)
 │         │ 1───1 Geometry     (step cloud, plane, chain stats — written online)
 │         │ 1───1 ViewState    (candidate views at this moment: address/pose/status)
 │         │ 1───N AIEvent      (events of AI runs that touched this step)
 │  1───N ViewMethod       (viewsphere | primitive-nbv | taught | … ; version + params)
 │  1───N AIRun           (orchestrator session; ≥0; references Menu)
 │         │ 1───N Transcript   (subagent runs — canonical single copy)
 │         │ 1───N Crop         (keyed transcript_id + turn — collision-proof)
 │  N───1 Menu            (versioned injected context; snapshot + shared id)
Derived (outside the run, existing tier: keyed run+method[,variant], provenance meta)
```

### Run — `run.json` (datasheet + status)
| field | notes |
|---|---|
| `schema_version` | loaders check, fail loudly on unknown |
| `id`, `name` | name given by user at start |
| `object` | free tag: cup, box, tube… (queryable) |
| `source` | `live` \| `data-engine` — stamped by the running code |
| `question` | inspection question, at start |
| `status` | `running` → `completed` \| `aborted` \| `crashed` (heartbeat or close-marker) |
| `created_at` / `closed_at` | wall clock |
| `steps` | ordered list of step_ids (the "ordered list of positions") |
| `view_methods` | `[{id, kind, version, params}]` — viewsphere is ONE method (`params={h_bins,v_elevs,r}`); future primitive-nbv etc. define their own params + address shape. A run can mix methods. |
| `q_survey` | taught survey config (rig-level fact) |

### Config — `config.json` (world-in-effect snapshot, at start)
git SHA · `cell.yaml` sha256 (+ embedded copy, it's small) · calib file name + sha256
(`T_flange_cam_*.npy`) · tuning constants in effect (speeds, gates, voxel, mask
thresholds, workspace box) · model identities (segmentation, VLM, orchestrator).
Makes "safe and doable" re-checkable and old runs renderable after cell.yaml edits.

### Step — `steps/<NNN>/step.json`
`step_id` · `view: {method: <method_id>, address}` — address shape is defined by the
method's sub-schema (viewsphere: `[h,v]`; survey: `taught`; free 6-DoF views may carry
no address — `T_base_cam` is then the only identity) · `t_arrived`, `t_captured` (unix
wall clock, declared) · `joints_rad[6]` · `T_base_flange`, `T_base_cam` (direction
declared in-band) · `rgb_rotation_deg` · segmentation record · geometry stats
(`offered/kept/dropped/extent_mm`, `plane[4]`) · self-description constants
(depth scale ref, mask polarity) or run-level equivalents.

**Robot-free 3D inspection**: `T_base_cam` is the camera frame's full 3D pose in base;
with the session intrinsics it renders the frustum/image plane in any 3D viewer without
the robot. `joints_rad`/`T_base_flange` are only needed for the robot-model overlay.

### Capture — `steps/<NNN>/`
`rgb.png`, `depth_aligned.npy` (color viewport, rotated with rgb), `depth_raw.npy`,
`ir_left.png`, `ir_right.png` (raw viewport, never rotated), `mask.png`.
Same bytes as today; the *description* moves into step.json.

### Geometry — `steps/<NNN>/cloud.ply` + run-level `fused/`
Per-step object contribution (colored PLY — colors travel in-file, row-filter bugs
impossible by construction). Written ONLINE at the fusing phase; fused cloud updated
per step → crash leaves a valid partial record.

### AIRun — `ai/<seq>/`
`mode: live | replay` (same schema serves data-engine reruns over captured steps).
`trace.jsonl` — append-only, never truncated; every event carries `step_id` where
applicable and references transcripts by id (no inlined `sub` copies).
`transcripts/<tNNN>.json` — canonical single copy, `kind` field
(survey/inspect/move/evidence).
`artifacts/<tNNN>_<turn>_<tool>.png` — **all** image-producing tool uses, not just
crops: crop, detect, segment, read/OCR overlays — keyed (transcript, turn, tool) so
collisions are impossible; structured tool results (boxes, read text) live in the
transcript turns as today.
Records: model ids, token usage (already computed, currently discarded), the opening
prompt verbatim (today's one unlogged model input).
`eyes/frames/` dies: the upright frame is recomputable from `rgb.png` +
`rgb_rotation_deg` — no second copy of the same pixels.

**`store.json` dissolves**: `views[]` → step.json (authoritative, no positional
rebuild); `plan`/`findings` → trace `write` events + transcripts (append, not
overwrite); `notes` → trace events. Nothing left for a second mutable ledger.

### Operator events — `events.jsonl` (run level, append-only)
Approvals/redirects/cancels/faults/stops with `t` + `step_id` refs — exists for BOTH
run kinds. Today approvals are recorded only for brain runs; the data-engine flow is
operator-approved by design, so the approval trail can't live inside AIRun.

### ViewState — `steps/<NNN>/view_state.json`
The candidate view set **as it existed at this step**: `[{address, pose, status:
available|visited|blocked|current}]` + the geometry context that produced it (object
centroid/extent, shell r). Needed because geometry-driven methods recompute view
coordinates every step as the object estimate updates — the set is dynamic, not a
run-level constant. (Today this is the ephemeral `_reach` cache + visited/blocked sets,
lost at exit.) Mid-step mutations (a refusal marking a cell blocked) land in
`events.jsonl`.

### Menu — referenced by AIRun (snapshot + shared id)
**Contract: a menu is a pure function of ViewState (+ AI context) — it never reads
live process state.** The menu is the transform from geometric/factual view knowledge
into the LLM text representation, so the record keeps BOTH sides of
`text = menu(view_state)`:
- **input** — `view_state.json` per step (above)
- **output** — the rendered text, per turn, in trace `state` events (already captured)
- **definition** — snapshot embedded in the run + `menu_id`/hash: verb set offered to
  the model, templates, rendering params, code version; declares which view-method
  kinds it can render.

Dual purpose, one mechanism:
- **Exact replay** → inject the recorded rendered text verbatim (byte-what-the-model-saw)
- **Menu experiment** → render the recorded ViewState with a different menu and inject
  that instead — same physical evidence, different representation (AIRun `mode: replay`)
- **Determinism health check** → re-render recorded ViewState with the original menu,
  hash-compare against the recorded text.

### Answer — `answer.json`
As today (reasoning, verdict, evidence, evidence_images, views_inspected) + explicit
back-references: `transcript_ids`, `step_ids`. Run status `completed` requires it;
absence ⇒ aborted/crashed — no more silent trailing-off.

### Manifest — `manifest.json` (at close)
sha256 per file. Migrations extend it (add-only); original entries never change.

---

## What this design kills (from the audit)

| defect | killed by |
|---|---|
| positional view↔turn↔AI joins | `step_id` on everything |
| crop overwrite corruption | `(transcript_id, turn)` keying |
| trace truncation vs store accumulation | append-only trace per AIRun dir |
| finding stored 4× | canonical transcripts + references |
| fused cloud lost on crash | per-step online write |
| run kind/status invisible | datasheet `source` + `status` |
| config free variables | `config.json` snapshot |
| two-pass non-atomic meta.json | step.json single writer + temp→rename |
| uninterpretable disk (units/frames/polarity) | self-description in-band + `schema_version` |
```
