# Context for demo sections 8, 9, 10

Collected 2026-09-22, read-only, from `inspection/` code and `inspection/data/runs/`.
Every number below was computed from an artifact or read off a cited line. Nothing
here is estimated. Anything not verifiable from an artifact is marked `(suggested)`
or listed under *Open questions*.

---

## Section 8 — Facts

### The run directory layout (exact)

The layout is fixed as code in `inspection/record/schema.py:1-36` (the module docstring
*is* the spec) and validated on every write. A real completed run,
`inspection/data/runs/2109-box-ray-demo19/`, holds exactly that shape:
`run.json` (`RunRecord`, `schema.py:324` — id, name, object, question, status,
`created_at`/`closed_at`, `view_methods`, `steps` roster, `q_survey`, `conventions`);
`session.json` (`SessionRecord`, `schema.py:369` — camera identity + intrinsics);
`config.json` (`ConfigSnapshot`, `schema.py:393` — `git_sha`
`5013157cbcb12d6fb074354b6f0caa0806684b6c`, `cell_yaml` {file, sha256
`bbafeb7d…a00b053`, full content inlined}, `calib` {`T_flange_cam_2026-08-19.npy`,
sha256}, `constants` {`padding_m` 0.02, `voxel_m` 0.003, `jump_gate_m` 0.08,
`range_m` [0.13, 0.6], workspace box, plane tolerances, `h_bins` 12, `v_elevs`
[10, 40, 70]}, `models` {`vlm: real`, `segmenter: sam3`, `fit: boxguard`});
`events.jsonl` (`OperatorEvent`, `schema.py:519` — one JSON line per
`requested`/`awaiting_approval`/`approved`/`captured`, append-only, written by both
writers); `steps/NNN/` with `rgb.png`, `depth_aligned.npy`, `depth_raw.npy`,
`ir_left.png`, `ir_right.png`, `mask.png`, `chain_prompt.png`, `chain_mask.png`,
`chain_kept.png`, `decomposition.json`, `fused/{cloud.npy,colors.npy}`, `step.json`
(`StepRecord`, `schema.py:172`) and `view_state.json` (`ViewState`, `schema.py:285`);
`ai/000/` with `airun.json`, `trace.jsonl`, `transcripts/t000..t016.json`,
`artifacts/*.png`, `answer.json`, `store.json`, `menu/{menu_def.json,inputs.jsonl}`;
run-level `fused/{cloud.npy,colors.npy}`, `keepout_blob*.obj`, and `manifest.json`
(`Manifest`, `schema.py:545`) carrying sha256 + bytes + mtime per binary file —
**119 files, 87.1 MB hashed** in this run (79 png, 38 npy, 2 obj;
`BINARY_EXTS` at `inspection/record/writer.py:44`, hashing at `writer.py:340-343`,
written by `RunWriter.close`, `writer.py:154`).

### Schema principles — one line each

- **Files are the truth / native folder capture.** One directory per run, plain files;
  capture code writes its own bytes into the step dir `begin_step` hands it, the writer
  owns only the contract JSON (`inspection/record/writer.py:1-18`). Every record carries
  `provenance: "native" | "legacy"` — a native run was written by this writer; a legacy
  one is read through an adapter and must list the fields the adapter fabricated
  (`schema.py:48-60`, `record/legacy.py:105`). Canon: "files are the truth, indexes
  disposable, no server processes" (`docs/blueprints/data-engine/management-design.md:5`).
- **Atomic step.** "One atomic step = one robot position. The universal join key"
  (`schema.py:172-173`); survey is step 0; `phase` discriminates the crash window
  (`captured` = geometry not yet written, `fused` = complete), so a crash is never
  ambiguous (`schema.py:176-180`). The steps roster flushes at the first write of each
  step, so a crash never leaves a step dir the roster doesn't know (`writer.py:16-17`).
- **Per-model versioning.** `SCHEMA_VERSION = 1` is only a baseline; each record kind
  bumps its own `VERSION` independently, readers are forward-tolerant (unknown fields
  ignored), writers go through `validate_for_write` which *refuses* unknown fields —
  the typo catcher (`schema.py:26-34`, `39`, `48-79`).
- **Config snapshot.** The world-in-effect at start, stored once per run: git sha,
  `cell.yaml` hashed *and inlined*, hand-eye calib file + hash, every tuning constant,
  and the model identities. Motion trajectories are deliberately **not** stored —
  "movement is recomputable; we only care it was safe & doable — which the config
  snapshot makes verifiable" (`docs/blueprints/data-engine/schema-draft.md:16-22`).
- **Transcripts once.** `ai/<seq>/transcripts/<tNNN>.json` is "one VLM subagent run,
  stored ONCE" — verbatim prompt, every turn with a `type` discriminator, final answer,
  usage; images keyed `(transcript, turn, tool)` so collisions are impossible
  (`schema.py:453-462`). The decision: "AI subagent runs: store them, ONCE (no 4×
  duplication)" (`schema-draft.md:23-25`).
- **StepRecord v2 — per-step fused cloud.** Since 2026-09-10 a fused step also persists
  the accumulated object cloud in its own dir (`fused/cloud.npy` + `colors.npy`,
  uint8 row-aligned), mirroring the run-level pair, because the accumulator voxelises
  and outlier-filters the merged cloud on every add — run-time truth, not a derivable
  quantity. v1 steps predate this and `validate` must not demand the files
  (`schema.py:181-190`).

### Replay / trace in the UI

`inspection/ui/app.py` exposes three fire commands (`ui/app.py:216`):

```
p inspection/ui/app.py trace --run_dir=inspection/data/runs/<run>   # ui/app.py:151
p inspection/ui/app.py serve --run_dir=inspection/data/runs/<run>   # ui/app.py:130
p inspection/ui/app.py mock                                          # ui/app.py:81
```

- `trace` (`ui/app.py:151-165`) — "Open a brain run's trace: every stage of the agentic
  loop, as a stream." It owns a bus, drives no robot, plans nothing: it reads the newest
  `<run>/ai/<seq>/trace.jsonl` and republishes on every mtime change. **The live story
  and the replay are the same code path** — "a run in progress is just a file still being
  appended to, so watching one and re-opening a finished one are the same code path"
  (`ui/app.py:157-161`). It also republishes `run/meta` from `run.json` so the UI shapes
  itself to the flow it is viewing (`ui/app.py:186-192`).
- `serve` (`ui/app.py:130-138`) — window only, no bus; this is what `run/app.py` spawns
  as a child process during a live run (`run/app.py:_open_ui` docstring).
- `mock` (`ui/app.py:81-99`) — a real `Supervisor` + `FakeRig` run over the real bus, no
  hardware, no camera; writes real capture files.
- The frontend must be built first (`ui/app.py:15-17`): `npm run build` in
  `~/projects/porthole` (framework) then in `inspection/ui`.

### e2e tests over the mock app

`inspection/tests/e2e/` — **5 tests collected**, Playwright/chromium against the mock
app launched as a real subprocess on free ports, fresh run dir per test
(`tests/e2e/conftest.py:1-12`):
`test_flow_a_ask_approve_writes_a_valid_live_run`,
`test_flow_b_no_ai_panel_sweep_writes_a_valid_data_engine_run`,
`test_flow_b_auto_collects_without_a_single_click`
(`tests/e2e/test_flows_e2e.py:38,91,177`), plus
`test_pin_brain_live_does_not_mistake_a_slide_leg_for_a_refusal` and
`test_flow_a_ray_transaction_with_a_face_and_a_scripted_fault`
(`tests/e2e/test_flow_a_rays_e2e.py:137,257`).

### The ray-views replay bench — and it is broken right now

`inspection/investigation/ray_views/ray_replay.py` replays **recorded runs offline** to
answer, before anything touches the machine (`ray_replay.py:1-20`): for every view a
recorded run actually executed, would the ray sampler have put the camera closer to the
object surface than the old formula did; do side faces come out fronto-parallel instead
of the 35–42° keystone; and do the paper timings survive real IK. Old endpoint = the
recorded `T_base_cam`; new endpoint = `resolve_cell`/`resolve_face`; the run's own last
`decomposition.json` is the belief it is judged under.

`evidence/bench_summary.txt` is the Stage B gate evidence over 4 runs
(`1709-cam-ai-demo1`, `1509-cam-ai_bench5`, `1509-cam-ai_bench4`,
`1509-lan-ai_bench5`), run as
`p inspection/investigation/ray_views/ray_replay.py --run all --world guard`. Headline
numbers it reports: 9/11, 7/9, 1/1 and 9/10 views closer to the surface, 0 inaccessible;
mean surface distance 0.194 → 0.158 m, 0.190 → 0.158 m, 0.150 → 0.133 m, 0.249 →
0.184 m; every executed row clears the D405 sensor floor (all along-ray ≥ 0.105 m);
measured timings one ray ~4.7–5.2 ms (est 6), 36-cell refresh 168–186 ms (est 230),
face side family 640–847 ms (est ~441), face top family ~1.0–1.1 s (est 954 ms) — the
only row over 2× estimate is the face shift rung (48–66 ms vs 20 ms est).

**It does not run today.** Say this plainly:

```
ray_replay.py:85:  from inspection.view.grid import H_BINS, R_MAX, R_MIN, V_ELEVATIONS
ImportError: cannot import name 'R_MIN' from 'inspection.view.grid'
```

`R_MIN` was deleted this morning by `48fd128` ("view: the fixed-position viewsphere
dies — Stage D deletes the old path", 2026-09-22 12:46), which removed `FILL_TARGET`,
`EXTENT_RULE`, `CLOSENESS`, `R_MIN`, `object_extent` and `radius_for_extent` from
`inspection/view/grid.py`. The `evidence/bench_summary.txt` numbers above are still
valid as a committed record; the bench itself needs a one-line fix before it runs again.

### Counts (all computed today)

| thing | value | how |
|---|---|---|
| run directories under `inspection/data/runs/` | **65** | `ls -1 \| wc -l` |
| …from 2109 (21 Sep) | **19** | `grep -c '^2109-'` |
| …from 2209 (22 Sep, today) | **7** | `grep -c '^2209-'` |
| total archive size | **2.6 G** | `du -sh inspection/data/runs` |
| schema-native runs (`provenance: native`) | 38 | read every `run.json` |
| legacy runs (no status/provenance) | 26 + 2 dirs with no `run.json` | same |
| statuses across the archive | completed 23 · aborted 11 · running 4 · legacy/none 26 | same |
| runs with an `ai/000/answer.json` | 21 | file existence |
| tests in the suite | **488 collected in 0.55 s** | `pytest --collect-only -q inspection/tests` |

---

## Section 9 — Run table

All ten runs are `source: live`, `rig: real`, `models = {vlm: real, segmenter: sam3,
fit: boxguard}` (each run's own `config.json`). Every one has `object: "box"` in
`run.json` — including the NETGEAR switch and the CropTeq PC — because `OBJECTS` is the
archive query tag, not the primitive (`run/app.py:25`).

Questions are verbatim from `run.json` (typos included — they are real operator input).
`wall` = `created_at` → `closed_at`. `active` = `created_at` → last `trace.jsonl` event
(this is the honest "how long did the agent work" number; `wall` includes the operator
leaving the window open after the answer). Tool calls counted from
`ai/000/trace.jsonl` `kind == "tool_call"`. Bounces = `kind == "tool_result"` with
`failed: true` and `tool == "answer"`. Every run also made exactly 1 `plan` call.

| run | subject | question (verbatim) | status | steps | wall | active | move | inspect | answer | bounces |
|---|---|---|---|---|---|---|---|---|---|---|
| `2209-box-ray-demo1` | AI KIT box | "inspect this box, what is the VBTI serial number?" | completed | 5 | 5m 18s | 5m 04s | 4 | 1 | 2 | 1 |
| `2209-box-ray-demo2` | AI KIT box | "what is teh VBTI serial numebr on this box?" | **running** | 2 | — | 2m 33s | 1 | 0 | 0 | 0 |
| `2209-box-ray-demo3` | AI KIT box | "What is the VBTI serial number?" | **aborted** | 1 | 0m 54s | 0m 48s | 0 | 0 | 0 | 0 |
| `2209-box-ray-demo4` | AI KIT box | "what is the VBTI serial number?" | completed | 6 | 6m 22s | 6m 03s | 6 | 2 | 2 | 1 |
| `2209-smallcam-ray-demo5` | D3 camera box | "inspect and give the complete spec of this camera" | completed | 9 | 16m 16s | 11m 59s | 9 | 7 | 2 | 1 |
| `2209-bigcam-ray-demo5` | Intel L515 box | "inspct this vam, what are the serial numbers?" | completed | 11 | 21m 54s | 14m 14s | 15 | 4 | 2 | 1 |
| `2209-compute-ray-demo5` | CropTeq "Chort" PC | "inspect this machine in details and give the compelte spec" | completed | 9 | 13m 17s | 12m 02s | 14 | 10 | 1 | 0 |
| `2109-box-ray-demo18` | AI KIT box | "inspect this box" | completed | 9 | 9m 36s | 9m 00s | 8 | 3 | 2 | 1 |
| `2109-box-ray-demo19` | AI KIT box | "inspect this box" | completed | 9 | 12m 00s | 9m 21s | 9 | 6 | 1 | 0 |
| `2109-lan-ray-demo16` | NETGEAR switch | "inspect this object" | completed | 8 | 11m 40s | 10m 32s | 7 | 8 | 1 | 0 |

**Run time, the honest headline:** median *active* time over the 8 completed runs is
**9.9 min** (596 s); median *wall* including the post-answer idle is **11.8 min**
(710 s). Range 5.1–14.2 min active. The demo spine's "~9 min" is right for the
box runs (`2109-box-ray-demo18` 9m 00s, `2109-box-ray-demo19` 9m 21s) and optimistic
for the "give me the complete spec" ones (12–14 min).

### Per-step phase breakdown (from `events.jsonl` `t` deltas)

Phases, as the events define them: `requested → approved` = **operator wait** (the
approval gate); `approved → captured` = **motion + capture**; `captured → next
requested` = **model thinking** (orchestrator turn + any vision-subagent hunts).
`awaiting_approval` lands 0.3–0.8 s after `requested` in every run.

| run | med operator wait | med motion+capture | med thinking |
|---|---|---|---|
| `2209-box-ray-demo1` | 3.8 s | 16.6 s | 21.1 s |
| `2209-box-ray-demo2` | 2.5 s | 15.5 s | 28.1 s |
| `2209-box-ray-demo3` | 1.1 s | 6.8 s | — (never got a 2nd turn) |
| `2209-box-ray-demo4` | 2.4 s | 21.5 s | 22.4 s |
| `2209-smallcam-ray-demo5` | 5.2 s | 14.8 s | 28.7 s |
| `2209-bigcam-ray-demo5` | 2.6 s | 20.4 s | 29.9 s |
| `2209-compute-ray-demo5` | 2.9 s | 17.4 s | 38.6 s |
| `2109-box-ray-demo18` | 3.7 s | 21.4 s | 20.2 s |
| `2109-box-ray-demo19` | 3.3 s | 25.2 s | 19.8 s |
| `2109-lan-ray-demo16` | 5.9 s | 24.0 s | 29.9 s |

**Medians pooled over all ten runs:** operator wait **3.2 s** (n=70), motion + capture
**19.4 s** (n=69), model thinking **28.5 s** (n=74). So a step costs ~50 s, of which the
operator's approval press is ~6 % and the model is ~56 %. The arm is the *cheap* part.

---

## Section 9 — Incomplete runs

Only what the artifacts say.

**`2209-box-ray-demo2` — `status: "running"`, `closed_at: null`.** Two steps (survey +
cell [1, 2]). The last `events.jsonl` line is `captured step_id 1` at t+153.2 s; the last
`trace.jsonl` line (seq 24) is `sub_step cell [1,2] n=1 of 16 event=thinking` at the same
instant. Both files simply stop mid-turn: the process died or was killed while the
vision subagent was on its first sub-step. There is no fault, no refusal, no abort record
— and no `closed_at`, which is exactly what `RunWriter.close` would have written. The
question is the one with two typos ("teh", "numebr"); `2209-box-ray-demo3` re-asks the
same question cleanly 3 min later, so this looks like an operator restart, but the
artifacts do not say that.

**`2209-box-ray-demo3` — `status: "aborted"`, 1 step, 54 s.** Survey only. Trace ends at
seq 11: the vision subagent had run `detect("cardboard box")` (0.98 at [261,369,717,575]
plus five weaker boxes), then `crop([454,169,755,529])` which returned "survey view · dir
000 · **TILTED -16 deg, not corrected** · crop 454,169-755,529 (301x360 px)", then
`event: thinking` on sub-step 3 of 8 — and stops. `closed_at` is present, so this one was
closed by the app, and `run/app.py:202-204` writes `"aborted"` when a run with a question
finishes with no `ai/*/answer.json`. No tool_call ever reached the orchestrator tier
(`tools: {}`), so the run was stopped during the very first survey hunt.

**`2209-bigcam-ray-demo5` — the artifacts say `status: "completed"`, not running.**
11 steps, `closed_at` present, an `answer.json` written (Intel RealSense L515 box: B/N
F0244802, product code 82638L515G1PRQ, MM# 999NGP, Made in Thailand). Worth showing
anyway because it is the most expensive run in the set: **5 refused `move` calls**, all
the same message — `['face','x+'] could not be planned — no collision-free path was
found from the arm's current stance. The arm has not moved. Plans are stance-dependent…`
— plus 1 answer bounce, and 460 s of wall time after the last trace event (the window
was left open). If Anton meant a different run, the table above is the one to trust.

**Also worth one line (`2209-compute-ray-demo5`):** 6 failed `move` calls, the last one
`the move to [0, 1] faulted; the run is halted.` (`brain/live.py:247` — the move ended in
the `fault` phase). The orchestrator then called `answer` 25 s later and produced the
full CropTeq spec from the 9 views it already had. A halted arm did not cost the answer.

**`pictures/badtcp.png`** — a UI 3D-scene screenshot: the UR5e wrist and gripper at
right with a **large translucent amber box that extends well past the physical gripper**
(the tool/camera collision hull), the fitted object box with its brown point cloud
sitting on the table at lower left, and the viewsphere marker spheres (green =
reachable, dark = not) floating in front of the dark red keep-out region. *(inference,
not stated by the file)*: this is the tool hull before `7e4531b` ("cell: cam_stack is
one-sided — the blue bracket came off the gripper", 2026-09-22), which shrank the
`cam_stack` slab from Z −100…+70 mm to Z −85…−25 mm after Anton unscrewed the angled
bracket. An oversized tool hull is exactly what produces the "could not be planned"
refusals above.

---

## Section 9 — Commands & checklist

### Launching a live run

```
p inspection/run/app.py run --name=<name> --object=box
```

- `run/app.py:1-11` (docstring): `run` lands in `data/runs/DDMM-<name>`; `--outdir`
  overrides. `_default_outdir` (`run/app.py:37-52`) suffixes `-2`, `-3`… if today already
  has that name; an explicit `--outdir` that exists **fails**, because evidence is never
  overwritten.
- Allowed `--object` values: `OBJECTS = ("box", "cup", "cylinder", "duck")`
  (`run/app.py:25`). A typo is a hard `SystemExit` (`run/app.py:28-33`) — "the archive is
  queried by this tag, so a typo would silently fragment the dataset".
- Other flags on `run` (`run/app.py:217-221`): `--ip` (default `192.168.2.50`,
  `run/app.py:19`), `--fit` (default `boxguard`; `none` disables), `--port` 8767,
  `--bus_port` 8765, `--no_window`, `--gui`.
- The question is **not** a flag — it is typed into the UI's Ask box; `run.json` is
  created with `question=None` (`run/app.py:300`) and filled when the operator asks.
- Ctrl-C once = safe shutdown (stop arm, save, close); twice = hard exit
  (`run/app.py:install_sigint`, `run/app.py:54-66`).

### Opening the UI on its own

```
p inspection/ui/app.py serve --run_dir=inspection/data/runs/<run>    # window, no bus
p inspection/ui/app.py mock                                          # no robot, no camera
p inspection/ui/app.py trace --run_dir=inspection/data/runs/<run>    # replay a recorded run
```

### Fallback replay — the run to open if the robot misbehaves

```
p inspection/ui/app.py trace --run_dir=inspection/data/runs/2209-compute-ray-demo5
```

Why this one: completed, **zero answer bounces**, 10 `inspect` calls over 8 views, the
richest structured answer in the set (CropTeq "Chort" fanless PC, full port inventory),
and it survives a mid-run motion fault — which makes the recovery story visible rather
than hypothetical. Second choice, if the AI KIT box is the object on the table:
`--run_dir=inspection/data/runs/2109-box-ray-demo19` (9 steps, 0 bounces, 9m 21s active,
reads "VATI AI KIT / A00110-INTL / AAB0007-0001 / MADE IN CHINA" off the face views).

### Pre-demo checklist

| # | item | grounded in |
|---|---|---|
| 1 | Robot powered, brakes released, **Remote Control** on the pendant | `preflight()` returns `go` only if `robot_mode == RUNNING` **and** `safety_mode == NORMAL` **and** `dash.isInRemoteControl()` (`motion/execute.py:112-134`). `run()` does `if not pf["go"]: raise SystemExit(f"preflight NO-GO: {pf}")` (`run/app.py:273-275`) — so a wrong pendant mode kills the app before the arm moves. `bringup(ip)` (`execute.py:137`) is the power-on + brake-release equivalent if needed. |
| 2 | RTDE stream fresh — no stale `q` | `run/settle.py:18-30`: run `2108-ui` had "a silently frozen RTDE q stamp five views with the survey pose; their planes came out at z 90/−490/242 mm, tilted up to 48 deg". The per-view plane gate (`PLANE_Z_TOL_M = 0.03`, `PLANE_TILT_TOL_DEG = 5.0`) now rejects any view whose fitted table plane contradicts the probed cell. `motion/rtde_soak.py` is the stream-death matrix if it recurs. |
| 3 | SAM 3 warm before the first ask | `7af62a1` — `run()` spawns a daemon `sam3-warm` thread right after `Sam3Backend()` is built; it logs `"sam3 warm in %.1f s — the first mask won't stall"`. **The duration is logged to stdout only, not into the run directory** — no recorded run carries the number. *(suggested)*: start the app, watch for that line, and don't ask until it prints. |
| 4 | Frontend built | `ui/app.py:_require_build` prints "frontend not built" and returns 1; build with `npm run build -w @porthole/framework` in `~/projects/porthole`, then `npm install && npm run build` in `inspection/ui` (`ui/app.py:15-17`). |
| 5 | UI window up, Stop button reachable | Stop hierarchy, strongest first: pendant e-stop (hardware) > Ctrl-C (`SIGINT` → `request_shutdown`) > UI Stop — "the button itself is never disabled, because a stop you have to wait for the UI to re-enable is not a stop" (`ui/AGENTS.md:10-14`). The UI can never move the arm (design FR11, `ui/AGENTS.md:8-10`). |
| 6 | Object placed, gripper bracket as-built | `7e4531b` (today) changed the `cam_stack` hull after the angled bracket came off the RG2. If the bracket state on the arm differs from `cell.yaml`, planning will refuse views. *(suggested)*: eyeball the wrist before the first run. |
| 7 | `--object` spelled from the vocabulary | `run/app.py:25` — `box` for all three demo objects. |
| 8 | Screen recording | *(suggested — no `wf-recorder` reference exists anywhere in the repo; nothing in the codebase sets this up.)* |
| 9 | Have the fallback trace command pasted and ready | see above. |

---

## Section 10 — Limits

Each one grounded.

1. **Box-only primitive.** `inspection/fit/apply.py:36-38`: `METHODS = {"boxguard":
   boxguard_fit}` — one entry, "the ONLY place an algorithm's name is bound to code; live
   loop and backfill both read it". Every run in the table records `models.fit =
   "boxguard"`. Face views are a *boxguard-face* view method (`run.json`
   `view_methods[1].kind = "boxguard-face"`), so no box fit ⇒ no orthogonal face views.
   The demo's own line: "Box only today" (`docs/demo/demo.md`, slide 6).
2. **Run time ~9–14 min.** Median active 9.9 min over 8 completed runs; a "complete
   spec" question costs 12–14 min (table above). The model is the cost centre: 28.5 s
   median thinking vs 19.4 s motion+capture vs 3.2 s operator approval per step.
3. **Oblique reads are leads, not evidence.** `03c185c` (2026-09-21): "the brain burned
   its first turns OCR-grinding oblique grid cells — and misread them ('Made in Italy'
   off [3,1] where the face view read 'MADE IN CHINA')". Verifiable in the artifacts:
   `2109-box-ray-demo7/ai/000/trace.jsonl` contains "Made in Italy" 4 times, all from
   cell `[3, 1]`, and it reached the final answer; `2109-box-ray-demo18` and
   `-demo19/ai/000/answer.json` both read `origin: "MADE IN CHINA"` off face views. The
   fix made the face-views paragraph **normative**: details are read from the face that
   owns them; an oblique read is a lead (where to look), never evidence (what is
   written).
4. **Refused faces need a reposition — of the arm, by the arm.** `9418bed`
   (2026-09-21): in demo10 "face y+ refused twice, and the orchestrator retried from
   wherever it happened to stand, then gave up on the face entirely and answered from a
   suboptimal view". Verifiable: `2109-lan-ray-demo10/ai/000/trace.jsonl` has
   `['face','y+'] could not be planned — no collision-free path was found` twice. The
   refusal message now carries the fact the model cannot know: **plans are
   stance-dependent**. Today's runs still hit it — 5 refusals in
   `2209-bigcam-ray-demo5`, 6 in `2209-compute-ray-demo5`, 1 each in
   `2209-box-ray-demo4`, `2209-smallcam-ray-demo5`, `2109-box-ray-demo19`.
5. **The system reports conflicts instead of hiding them — but the conflict is real.**
   `2109-lan-ray-demo16/ai/000/answer.json`: front panel silkscreen reads `GS308EPP`,
   the rear compliance sticker's small print reads `GS305EPP V1`. The verdict declares
   the conflict, then decides on physical evidence — eight jacks labelled 1–8 and a
   54 V/2.4 A (~130 W) supply match the 8-port PoE+ model — and calls the small-print
   '5' a misread. Honest, and still a sub-millimetre OCR limit.
6. **Reachability: a third of the sphere is unreachable.** Computed from each run's
   `ai/000/menu/inputs.jsonl` (`agent_context.nav.reachable`, 36 cells total): steady
   state is **25–33 of 36** reachable, and the *first* turn is worse (21–26 of 36) before
   the fit settles. Worst in the set: `2209-box-ray-demo4` decays 22 → 25 over 10 turns;
   `2109-lan-ray-demo16` is best at 32–33. The menu prints this to the model as
   "visited N of M reachable" (`brain/render.py:150`); an unreachable cell is "not an
   option at all" (`render.py:187-188`). Cells near the base column and behind the
   keep-out corners are what is missing.
7. **The operator approval gate — this is not autonomous.** Every move is
   `requested → awaiting_approval → approved` in `events.jsonl` before the arm is allowed
   to execute; the UI never moves the arm itself (`ui/AGENTS.md:8-10`). Autonomy is
   written down as a *seam*, not a build: "Autonomy (auto-confirm) — the seam is marked,
   not built" (`docs/blueprints/data-engine/flows-design.md:224`). Cost measured: 3.2 s median per
   step, ~6 % of step time.
8. **The record stopgap: one radius float carries a mean.** A cell is a ray with its own
   resolved depth now, but the schema still keeps a single `r` and the UI draws a single
   marker shell — so both carry the **mean resolved depth over accessible cells**
   (`run/machine.py:1324-1339` `_mean_depth`, stamped at `machine.py:1391-1393` and into
   every `ViewState` at `machine.py:1638-1642`). It is `None` when nothing resolved —
   "never a constant dressed as a mean". Visible in the artifacts: `2109-box-ray-demo19`
   `run.json` `view_methods[0].params.r = 0.2528571…` — a mean, not a design radius.
   Stage C is where rays get their own fields.
9. **Depth quality on glossy/textureless surfaces is the standing dependency.** "the
   D405's own passive stereo fails on precisely the glossy, textureless surfaces the demo
   uses" (`docs/report/project-definition.md:151-155`, also `:199-201`).
   *(No document in `inspection/docs/` says "transparent objects are out of scope" —
   see Open questions.)*
10. **Productisation is explicitly out of scope.** "Cycle time, throughput and robustness
    beyond demo conditions are out of scope"
    (`docs/report/project-definition.md:95-96`).
11. **Planner benchmarking from recorded runs is impossible today.** Trajectories,
    planner reports, IK failure reasons and the per-turn reachability map are all
    ephemeral: "kinematic replay can only **teleport between capture poses**; planner
    benchmarking from recorded runs is **impossible**; re-planning would produce a
    *different* path (OMPL unseeded)" (`docs/blueprints/data-engine/loop-data-audit.md:96-99`).
    Cognition replay is fine; motion replay is not.
12. **A garbled citation used to bounce a finished hunt.** Fixed today by `884d895`: in
    `2209-box-ray-demo4` the hunt at face y+ succeeded, the `done` tick fired at trace
    seq 150, and the next line raised `TypeError('sequence item 0: expected str instance,
    list found')` — the answer bounced and no transcript was written (transcripts stop at
    t008). The same bounce text is in `2209-smallcam-ray-demo5` and
    `2109-box-ray-demo18`. Three of the five bounces in the table are this bug, now
    fixed; the other two (`2209-box-ray-demo1`, `2209-bigcam-ray-demo5`) are genuine —
    the evidence hunt looked and did not find, and the answer was correctly sent back.

---

## Section 10 — Written-down next steps

Only items that exist in a file. Nothing invented.

- **VLA as phase two** — `docs/report/project-definition.md:829-843`,
  "Phase two — the IK system as a data engine": "The VLA is not abandoned, it is
  sequenced. Once the IK loop works, every episode it runs produces exactly the
  supervision a view-selection policy needs: scene, question, candidate set, chosen view,
  and whether that view resolved the answer." The cited evidence: three 2025–26 results
  where a distilled/fine-tuned 7B beats frontier models at viewpoint selection (83.72 vs
  GPT-5's 72.09; 47.8 % vs 18.5 %; EyeVLA 96 % task completion from **500 training
  samples**). "So phase one ships a working inspector; phase two distills it into the
  self-contained model the product vision ultimately wants."
- **Primitives beyond the box** — `docs/plans/2026-09-17-ray-views-stage-b-plan.md:13-17`:
  "No obstacle unification in Stage B: padding-on-top-of-decomposition (multi-primitive)
  is a future design of its own." This is the padded-decomposition obstacle item: the
  reason more primitives are a design, not a config change.
- **Closeness control** — `docs/blueprints/ray-views.md:129-131`, "Later
  (explicitly out of scope): a closeness *cap* as one more stop condition —
  camera-closeness control plugs in with zero new structure." (Note `CLOSENESS` itself
  was deleted today by `48fd128`, per that doc's "Dies" list.)
- **Benchmark arbitrating the engineering defaults** —
  `docs/blueprints/face-evidence.md:106-108`: "No benchmark work in this build —
  the gate tolerance and search defaults are engineering defaults until the benchmark
  arbitrates them." Also `:43`, the 5° face tolerance is "tunable; benchmarked later, not
  sacred".
- **AI replay harness (E1–E3)** — `docs/blueprints/data-engine/flows-design.md:222-223`: "AI replay
  harness (E1-E3) — next branch; it consumes `Run`, designed for it." Design sketch at
  `docs/blueprints/data-engine/management-design.md:66-72`: a new AIRun dir with `mode=replay`, menu
  substitution via `MenuDef` + `MenuInput`, exact replay injecting recorded rendered text
  — "Touches brain/loop.py — build phase of its own." This is what makes "benchmarks run
  on replays" true rather than aspirational.
- **Evaluation, the three `[later]` stories** — `docs/blueprints/data-engine/management-user-stories.md:81-88`:
  G1 attach ground-truth labels and compute verdict-vs-label metrics across a set;
  G2 define an eval set (run ids + questions) and get a scorecard; G3 export a selected
  run set to a training format (LeRobot/…) under `exports/` — "a projection, rebuildable,
  never hand-patched". G3 is the concrete bridge from the archive to phase two.
- **Autonomy (auto-confirm)** — `flows-design.md:181-182` and `:224`: "the driver sends
  the confirm itself — one marked line, the same seam as the brain's future autonomy";
  listed under *Out of scope (deliberate)*.
- **Trace reader, migration runner, catalog/exports** —
  `management-design.md:67` (trace reader, later), `:81` (migration runner F3, later),
  `flows-design.md:225-226` (catalog/API server, exports — later consumers).
- **Asking the user when the reference is ambiguous** —
  `docs/report/project-definition.md:276`: "Out of scope, and the
  natural extension: letting the model ask the user when the reference…".

---

## Evidence to show

- Layout, on screen: `inspection/data/runs/2109-box-ray-demo19/` — `tree -L 2`, then
  `run.json`, then `steps/000/`, then `ai/000/`.
- Schema as spec: `inspection/record/schema.py:1-36` (the docstring), `:172-190`
  (StepRecord + v2), `:453-462` (transcripts once).
- Writer contract: `inspection/record/writer.py:1-18`.
- Manifest: `inspection/data/runs/2109-box-ray-demo19/manifest.json` — 119 files, 87.1 MB.
- Config snapshot: `inspection/data/runs/2109-box-ray-demo19/config.json` — git sha +
  inlined `cell.yaml` + constants + models.
- Trace/replay: `p inspection/ui/app.py trace --run_dir=inspection/data/runs/2209-compute-ray-demo5`.
- Bench evidence: `inspection/investigation/ray_views/evidence/bench_summary.txt` and the
  renders beside it (`1709-demo1_comparison.png`, `1509-cam-bench5_comparison_guard.png`).
- The misread: `2109-box-ray-demo7/ai/000/trace.jsonl` (cell [3,1], "Made in Italy") vs
  `2109-box-ray-demo18/ai/000/answer.json` (`origin: MADE IN CHINA`).
- The declared conflict: `2109-lan-ray-demo16/ai/000/answer.json`.
- The refusal message: any failed `move` tool_result in
  `2209-bigcam-ray-demo5/ai/000/trace.jsonl`.
- The fault-then-answer: `2209-compute-ray-demo5/ai/000/trace.jsonl` seq 354-395.
- Answer cards already picked for slide 1 (`docs/demo/demo.md`):
  `pictures/card_aikit.png` (2209-box-ray-demo4), `pictures/small_cam.png`
  (2209-smallcam-ray-demo5), `pictures/bigcam_card.png` (2209-bigcam-ray-demo5);
  plus `pictures/compute_card.png` (2209-compute-ray-demo5) and
  `pictures/badtcp.png`.

---

## Claims to avoid

- **Do not describe a fixed-radius viewsphere.** It was deleted this morning by `48fd128`
  — `derive_r_floor`, the fill formula, `R_MIN`, `max(want, floor)`, the frozen shell
  radius, the face floor and the z-lift rung are all gone. A cell is a **ray** with its
  own resolved depth. `ViewSphere` survives only as the cell-frame helper the UI draws
  markers with, and `run.json`'s single `r` is the **mean of the depths actually
  resolved**, a declared Stage B stopgap (`run/machine.py:1324-1339`).
- **Do not mention the SO-ARM101.** Explicitly not used in this project (Anton,
  2026-08-14): "Its 5 DOF makes look-at exactly determined with no branch choice, so it
  would validate plumbing rather than kinematics"
  (`docs/report/project-definition.md:105`). This is a UR5e with a
  wrist D405.
- **Do not say the model controls the robot (VLA).** The model picks from a discrete
  menu computed by logic; the harness resolves the pick into motion
  (`docs/demo/demo.md` line 1: "Not a VLA (yet)"). VLA is phase two, sequenced, not
  running.
- **Do not imply autonomous operation.** Every move passes the operator approval gate;
  it is in every run's `events.jsonl` and it is what the 3.2 s median wait measures.
  Auto-confirm is a marked seam, not built (`flows-design.md:224`).
- **Do not claim the ray replay bench runs today.** It fails at import on `R_MIN`
  (`ray_replay.py:85`). The committed `bench_summary.txt` numbers stand; the tool needs a
  fix.
- **Do not claim planner/motion replay.** Only cognition replays; trajectories were never
  recorded, by design (`loop-data-audit.md:96-99`).

---

## Open questions for Anton

1. **`2209-bigcam-ray-demo5` is `completed`, not `running`.** Its `run.json` has
   `closed_at` and an `answer.json` exists. Was a different run meant, or was it left
   open in the UI for ~7.7 min after the answer (which the timestamps do support)?
2. **MAY-188 is not referenced anywhere in the repo.** `grep -rn "MAY-188"` finds nothing
   (MAY-183, 186, 187, 193 all appear). The benchmark commitments I could ground are
   `face-evidence-design.md:106-108`, the replay harness E1–E3
   (`flows-design.md:222`, `management-design.md:66-72`) and stories G1–G3
   (`management-user-stories.md:81-88`). Is MAY-188 the ticket for one of those?
3. **"Transparent objects out of scope" is not written down.** The closest written
   statements are about *glossy and textureless* surfaces being the hard case
   (`project-definition.md:151-155`, `:199-201`) and the blanket "robustness beyond demo
   conditions is out of scope" (`:95-96`). Want me to say "glossy/textureless depth is
   the standing dependency" instead, or is there a transparency statement somewhere I
   did not find?
4. **SAM 3 warm-up duration is not recorded.** `7af62a1` logs
   `"sam3 warm in %.1f s"` to stdout only; no run directory captures it. If you want a
   number for the slide, it has to come from a terminal log of a live run.
5. **`wf-recorder` appears nowhere in the repo.** Screen recording is entirely a
   workstation-side step — nothing in `inspection/` sets it up or expects it.
