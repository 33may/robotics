# Loop Data Audit — what the inspection system produces today

Date: 2026-09-03 · code at `76fabc3` · evidence: 5 parallel code+disk audits
(run loop, sensing/cell, motion, AI tier, on-disk archive of 27 runs / 920 MB).
Purpose: the ground truth for designing the Single Recording schema (MAY-193 step 2).
Every claim carries a `file:line` or a run path — nothing here is from memory.

---

## 1. The process — tiers and phases

Five tiers touch data; strict trust boundaries between them:

| Tier | Modules | Role | Moves robot? | Sees pixels? |
|---|---|---|---|---|
| **Orchestration** | `run/machine.py` (Supervisor), `run/app.py` | state machine, turn ledger | via motion | no |
| **Motion** | `motion/{plan,ik,rrt,smooth,direct,execute}.py` | target → IK → plan → execute | yes | no |
| **Sensing/geometry** | `perception/{camera,capture}.py`, `cell/geometry.py`, `run/segmenter.py` | capture, segment, deproject, fuse | no | yes |
| **Eyes** (AI vision) | `eyes/` — VLM subagents (Gemini ER-2) | frames, crops, findings | never (asserted by test) | yes |
| **Brain** (AI orchestrator) | `brain/` — Claude SDK loop | plan/inspect/move/answer, verdict | via operator gate | never ("blind planner") |

State machine (`machine.py`, single dispatcher thread):

```
idle → planning → previewing → executing → capturing → fusing → idle
                     ↑ operator/brain approval gate (view/confirm)
sinks: fault, done          busy-block: executing|capturing|fusing|fault
```

**Decision points** (who chooses the next action):
1. **Target selection** — human UI click OR `Brain.move(cell)` tool call. `run/decider.py` is **dead code** on the current path (its own docstring says so).
2. **Approval gate** (previewing→executing) — always a human click on `view/confirm`; the brain never self-approves (`brain/live.py:118-184` polls the gate). A second code-level gate (`Brain._gate` PreToolUse hook, `brain/loop.py:493-515`) denies out-of-range cells before the operator is even asked.
3. **Planner tier choice** — direct joint move (~75%) vs RRT-Connect + polish + detour gate 2.5 (`motion/plan.py:31-74`).
4. **Segmentation trust** — mask path vs depth-growth fallback, 4 fallback reasons (`run/segmenter.py:118-167`).
5. **Plane gate** — per-view RANSAC table plane vs tolerance (`machine.py:77-99`).
6. **Verdict** — `Brain.answer()` → evidence hunts → `eyes/answer.json`. Only exists for brain runs that complete.

---

## 2. What lands on disk today (current generation, 3108-era)

```
runs/<id>/
├── session.json            writer: RealRig.__init__ (rigs.py:217-220), ONCE, before Supervisor exists
│                             serial, resolution, depth_scale_m_per_unit, intrinsics{color,depth,ir_l,ir_r},
│                             extrinsics_ir1_to_ir2.   FakeRig runs: FILE ABSENT.
├── run.json                writer: Supervisor._save (machine.py:676-708), flushed EVERY turn
│                             {q_survey[6], r, turns[{step,target,t,result,stopped}]}
│                             target = "survey" | [h,v]  (untyped union). result = free text.
├── <NNN>/                  per view; writer save_bundle (capture.py:35-75) then save_mask (78-103, 2nd pass)
│   ├── rgb.png             color-aligned viewport, may be rotated 180° (rgb_rotation_deg)
│   ├── depth_aligned.npy   uint16 848x480, color viewport, rotated WITH rgb
│   ├── depth_raw.npy       uint16, IR/depth viewport, NEVER rotated (deliberate asymmetry, capture.py:45-59)
│   ├── ir_left.png ir_right.png   raw viewport, unrotated, unused by loop (future stereo)
│   ├── mask.png            0/255, polarity undeclared on disk; orientation = rgb's
│   ├── chain_prompt/mask/kept.png  diagnostic renderings (overlay.py), each optional
│   └── meta.json           pose_id, timestamp, rgb_rotation_deg, joints_rad[6],
│                             T_base_flange[4x4], T_base_cam[4x4], mask{score,box,px}|null
├── eyes/                   AI tier
│   ├── store.json          grid{h_bins,v_elevs,r}, views[{cell|null,pose_id,cap_dir,t,T_base_cam}],
│   │                         notes[], plan (scalar, overwritten), hypothesis (dead field), findings[]
│   ├── trace.jsonl         run|thinking|text|tool_call|tool_result|state|write|answer|approval|sub_step
│   │                         TRUNCATED on every Brain() construction (trace.py:38-39) — latest run only
│   ├── transcripts/fNNN.json  full VLM subagent transcript incl. verbatim prompt; one GLOBAL counter
│   │                         across survey/inspect/move/evidence-hunt — kind not in filename
│   ├── frames/<cap>.png    upright frame the VLM saw (idempotent overwrite)
│   ├── crops/<cap>_<NN>.png  ⚠ NN = per-invocation turn counter → FILENAME COLLISIONS (see §5.1)
│   └── answer.json         reasoning, verdict, evidence, evidence_images (2708+), views_inspected,
│                             coverage (pre-rendered ASCII art). Absent if answer() never reached.
├── fused_cloud.npy         float64 (N,3) base frame; writer app.py:182-186 ONCE AT PROCESS EXIT
├── fused_colors.npy        uint8 (N,3) row-aligned (from 2026-09-03; absent on all archive runs)
└── keepout_blob{0,1}.obj   display-only re-export of cell.yaml keep-outs (publisher.py:277-297), 3108+
```

Join rules (all by convention, no explicit IDs):
- view dir `NNN` ↔ turn: "dir NNN's cell = N-th non-survey turn of run.json" — duplicated
  independently in `machine.py:686-690` and `eyes/replay.py:17-41`, matched by *position*.
- `turns[].step` and view `pose_id` are **disjoint numbering namespaces** (survey = pose 0 but turn 1).
- depth arrays ↔ depth_scale: positional (separate files, no reference field).
- tool_call ↔ tool_result in trace.jsonl: sequence order (`tool_use_id` available but never written).

---

## 3. Ephemeral inventory — computed every run, lost at exit

### Motion (the biggest hole)
| lost item | exists at | note |
|---|---|---|
| planned joint path (the thing previewed AND executed) | `machine.py:380` `self._path` | never written anywhere |
| planner report: tier, detour, tries, ms, branch counts, refusal reason | `plan.py:43-74` `rep` dict | only a log line survives |
| chosen roll (0/180) per cell | `machine.py:465` | discarded; recoverable from T_base_cam |
| executed trajectory (RTDE `actual_q` streams at 125 Hz) | `execute.py:396-399` poll loop | never buffered — inter-view motion unrecoverable |
| exec report {waypoints_done, s, final_err_deg, stopped} | `machine.py:580` | one bool read, rest discarded |
| IK failure reasons, collision diagnoses (`first_collision`) | plan/world call sites | branch-on-ok only |
| reachability map `self._reach` per turn | `machine.py:457` | drives all UI cell colors, gone at exit |

Consequence: kinematic replay can only **teleport between capture poses**; planner
benchmarking from recorded runs is **impossible**; re-planning would produce a
*different* path (OMPL unseeded) — rendered inter-view motion would be fiction.

### Configuration & provenance (the silent free variables)
- No git SHA, no schema version, no run kind/status anywhere.
- All tuning constants unstamped: SPEED_RAD_S/ACC/SPEED_SLIDER, plane-gate tolerances,
  JUMP_GATE_M/VOXEL_M/GROW_EPS_M/DBSCAN_*, MIN_MASK_SCORE/PX, paddings, WORKSPACE crop box.
  Two runs with different constants are indistinguishable on disk.
- Which `calib/T_flange_cam_*.npy` was loaded: resolved by filename-sort "latest"
  (`geometry.py:71-86`), identity never recorded (only baked into T_base_cam).
- `cell.yaml` content in effect: loaded fresh each process, never snapshotted — an edit
  silently invalidates sim renders / collision reconstruction of every older run.
- Segmentation model identity, VLM identity (`gemini-robotics-er-2-preview`), SAM3/OCR
  backends, sampling params: nowhere on disk (only a "real"/"stub" label).

### Sensing/geometry
- Per-view object cloud (pre-fusion contribution) — only the fused result survives, and
  only at exit. `chain_kept.png` is a *picture* of it, not the data.
- Per-view RANSAC table plane [a,b,c,d] — used for the gate, discarded pass or fail.
- `n_scene`, chain stats (offered/kept/dropped/extent_mm), prompt box, fallback reason —
  bus-only; the chain PNGs are saved but the numbers explaining them are not.

### AI tier
- Token usage / cost: captured in `ResultMessage`, printed or discarded (`loop.py:629`,
  `live.py:252`); Gemini `usage_metadata` never read.
- Orchestrator thinking text: SDK exposes only estimated token counts — `thinking` events
  carry `text:""` (`loop.py:565-582`). Not fixable by schema; record the limitation.
- The opening prompt (`render.opening()`) — the ONE model input not logged verbatim
  (per-turn `state` blocks ARE logged).
- Per-call latencies (reconstructable from trace `t` deltas, never computed).
- Operator identity + approval records for pure-UI (non-brain) runs.

---

## 4. On-disk reality — 27 runs, 920 MB, 6 layout generations

Timeline of drift: base 6 files (2108) → `eyes/` tree (2408-cup1) → `mask.png` +
`meta.mask` + `rgb_rotation_deg` (2408-cup3) → `chain_*.png` (2508) → `eyes/frames`
1-based→0-based + `answer.json.evidence_images` (2708) → `keepout_blob*.obj` +
`fused_colors.npy` (3108/0309).

Stable across ALL generations: `run.json` keys, `session.json` keys, `store.json` keys.
Drift is concentrated in per-view meta and the eyes artifacts.

Integrity findings (real, current):
- `2408-seeded`: view `003` missing — sequences are NOT contiguous.
- `derived/2408-cup5`: step 029 silently missing across `steps/` and all 3 fit methods.
- `2608-aicam`/`2708-aicam`: some views legitimately have `mask:null` + no mask.png —
  "every view has a mask" is false by design.
- 5 aborted/debug runs (2108-b/c, anim-debug, 3108-camdemo/3) sit in the same namespace
  as real runs — nothing on disk marks run kind or completion status.
- 7 runs have no `answer.json` (never reached verdict); trace just stops, no abort marker.
- Empty `.claude/agent-memory/` dirs contaminate 3 run trees + `runs/` itself.
- No zero-byte files anywhere. Depth npys are the largest per-view artifacts (814 KB ×2).

The **only provenance token in the whole archive**: `derived/*/fits/*/scores.jsonl`
carries `"version": "cyl/1.0+p:bf23b3aa"` (method/semver + code hash) — the pattern to
generalize, not invent.

`derived/` layout (4 runs): `steps/{step_NNN.ply,summary.json}`,
`fits/{cyl,ems,planar}/{step_NNN.json,scores.jsonl}`, `fits/renders/*.png`, `rrd/<run>.rrd`.
`summary.json` records source run + params + per-step `pose_id` join key.

---

## 5. Verified defects the new schema must design away

1. **Crop filename collision — actual data corruption.** `crops/<cap>_<NN>.png` uses a
   per-invocation turn counter (`vlm_agent.py:325-326`); two subagent visits to the same
   view overwrite each other's crops. Verified at pixel level in 3108-camdemo5:
   `crops/003_07.png` now holds f005's crop while `transcripts/f004.json` cites it as its
   own evidence. Key crops by `(transcript_id, turn)`.
2. **Ledger desync by construction.** `trace.jsonl` truncates on every `Brain()`
   construction; `store.json` findings/notes accumulate forever. Two ledgers, opposite
   retention, no run-boundary marker.
3. **Positional joins everywhere** (view↔turn, arrays↔scale, call↔result). One renamed
   dir or reordered turn silently mis-labels every view. Explicit IDs needed.
4. **`meta.json` two-pass write, non-atomic** (bundle then mask) — a crash between leaves
   a file whose missing `mask` key is indistinguishable from "segmentation never ran".
5. **Fused cloud written only at process exit** while run.json flushes per turn — a crash
   loses the cloud that `turns[].result` already claims ("fused N pts").
6. **Finding stored 4× ** (transcript file, inlined `sub` in trace, `tool_result.text`,
   one-line summary) — redundancy without a declared canonical copy.
7. **No run datasheet**: object identity is a pun in the run name; run kind (real /
   debug / mock / aborted) undeclarable; FakeRig runs lack session.json.
8. **Self-description gaps**: depth units, transform directions, mask polarity,
   `joints_rad` vs `joints_deg` (hand-eye tools use degrees!), grid-cell semantics —
   all only in code, and the code will outlive none of it.

---

## 6. The data sources — the list we decide formats for

Each of these is one conversation item: format, where it lands, what's added vs today.

| # | Source | Today | Gap severity for consumer #1 (algorithm dev) |
|---|---|---|---|
| 1 | **Run identity / datasheet** | nothing (name puns) | HIGH — "benchmark across objects" needs queryable object/kind/status |
| 2 | **Capture bundle** (rgb, depth×2, ir×2, per-view meta) | solid, quirky (rotation asymmetry, 2-pass meta) | LOW — works; needs self-description + atomicity |
| 3 | **Calibration & config snapshot** (session.json, T_flange_cam identity, cell.yaml, tuning constants, git SHA) | intrinsics only | HIGH — geometry algorithms need to trust the transform chain per run |
| 4 | **Per-view geometry** (object cloud contribution, plane, chain stats) | ephemeral; `derived/steps` for 4 runs via replay | **HIGHEST** — this IS the algorithm-dev working set |
| 5 | **Fused cloud** (+colors) | at-exit npy pair | MED — make incremental/crash-safe |
| 6 | **Segmentation record** (mask + score + model identity + fallback reason) | mask.png + 3 meta fields | MED |
| 7 | **Motion record** (planned path, planner stats, exec report, executed trajectory) | nothing | MED now, HIGH for later next-view work & sim render |
| 8 | **AI records** (store/trace/transcripts/answer, dedup, IDs, model identities, usage) | rich but 4× redundant, collision bug, truncation | LOW now, HIGH for debugging/eval consumers |
| 9 | **Operator actions** (approvals, redirects, identity) | brain runs only | LOW |
| 10 | **Derived tier** (steps, fits, renders, rrd) | good pattern, ad-hoc schema, silent step drops | HIGH — this is the benchmark loop's output side |
| 11 | **Exports** (LeRobot/RLDS, someday) | n/a | deferred by decision |

Consumer priority (Anton, 2026-09-03): **1) primitives-deconstruction algorithm dev**
(wider object set, Rerun, refine until sound) → 2) deployed-system tracing/debugging →
3) evaluation set → 4) someday VLA/LLM training. Schema optimizes for #1 without
foreclosing #2-4; capture stays native (export-layer rule).
