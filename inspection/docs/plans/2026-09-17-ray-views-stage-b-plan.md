# Ray Views — Stage B Implementation Plan (machine: the three-leg transaction)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Design: `inspection/docs/2026-09-17-ray-views-design.md` (§Move transaction, §Failure semantics, §Dies/stays).
Stage A (merged on this branch): `view/rays.py` sampler + `resolve_cell`/`resolve_face`, containment floors, replay bench.
Anatomy reference: machine.py move path mapped 2026-09-17 (phases `_set_phase`, `_plan_worker`, `_exec_worker`, `_retreat_worker`, `_recenter`).

## Rulings this plan is built on (Anton 2026-09-17)

- **Stage A gate passed** — Stage B approved on the bench evidence.
- **Obstacle world:** rays sample against the SAME world the motion validator
  uses — live `self.world` carrying `geometry.object_box` + coal padding. No
  obstacle unification in Stage B: padding-on-top-of-decomposition (multi-
  primitive) is a future design of its own. Consequence: the boxguard-world
  bench numbers are optimistic by ~2–4 cm; Task 5 re-benches in guard-world
  mode as this stage's gate evidence.
- **Rolls stay locked** {0°, 180°}; flips happen only in the hop leg. The
  top-face diagonal containment bound stands.
- **Containment stands:** θ=0 face members keep the whole face in frame,
  cell views the whole belief box (per-corner bound, true box corners).

## Global Constraints

- Fake rig only. Nothing in this stage touches the UR5e; no robot-contact
  commands. Live bench + demo are Stage D.
- `brain/` untouched; record **schema untouched** (Stage C owns migration).
  Interim honesty stopgap: `ViewState.r` (schema float) stamps the mean
  resolved cell depth; the proper ray fields land in Stage C.
- Old resolution path (`grid.radius_for_extent`, `viewsphere.plan_to_cell`,
  `viewsphere.derive_r_floor`, `faceview.resolve_face_view`) stays importable
  and tested until the Stage D merge — machine.py just stops calling it.
- Every task: failing test first, registered in the file's `main()`. Full
  affected suites green before each commit.
- Machine invariants preserved: dispatcher-only state mutation (documented
  exceptions unchanged), approval gate untouched, gen-counter staleness
  rules unchanged, `_BUSY_PHASES` extended with `"adjusting"`.

## Task 1: cell accessibility map — rays replace `sphere.reachability`

- Code: `inspection/view/rays.py` (add `cell_map`), `inspection/run/machine.py`
- Test: `inspection/tests/test_rays.py`, `inspection/tests/test_machine.py` (append)

**Interfaces:**
- Produces: `cell_map(world, ik, t_flange_cam, center, aabb, fy, height_px, corners=None, q_near=None) -> dict[(h,v), RayResolution|None]` — all `H_BINS × len(V_ELEVATIONS)` cells, `None` = inaccessible. Measured cost ~130–180 ms (bench) — runs where `sphere.reachability` runs today (`_plan_worker`, post-fuse refresh).
- machine: `self._reach` value type becomes `RayResolution|None` keyed by the SAME cell ids (request gate, menu status, `_set_phase` total, sweep candidates all keep their keys). `self.r`, `self.r_floor`, `derive_r_floor`, `radius_for_extent` calls die in `_recenter`; `self.sphere` survives only as the cell-frame helper (center + `cam_pose` for markers) until Stage C.

- [ ] **Step 1: failing tests** — `test_cell_map_covers_grid_and_matches_resolve_cell` (map entry == `resolve_cell` for a spot-checked cell; inaccessible cell is `None` not missing); machine test: request gate refuses a `None`-map cell with the honest detail.
- [ ] **Step 2: run to verify failure**
- [ ] **Step 3: implement** `cell_map` (thin loop over `resolve_cell`, shared corners) + machine `_reach` rewire + `_recenter` diet (r/r_floor/derive die; `ViewState.r` stopgap = mean resolved depth of accessible cells).
- [ ] **Step 4: run to verify pass** (`test_rays`, `test_machine`, `test_flow_a`, `test_live_faces`)
- [ ] **Step 5: commit** `machine: ray cell map replaces shell reachability — r/r_floor die`

## Task 2: the transaction planner

- Code: `inspection/motion/transaction.py` (new)
- Test: `inspection/tests/test_transaction.py` (new)

**Interfaces:**
- Produces: `Legs = namedtuple("Legs", ["out", "hop", "slide_in", "world"])` (each leg a waypoint list; `out` may be empty — first move after survey / unknown current ray). `plan_transaction(world, ik, t_flange_cam, current, target, q_now, seed=0) -> Legs | None`:
  - `current`: `RayState = namedtuple("RayState", ["ray", "depth", "roll", "samples"])` or `None`.
  - `target`: a `RayResolution` (from `resolve_cell`/`resolve_face`; its `samples` list is the slide-in corridor).
  - **Slide-out** (design §transaction): entry on the target ray PINNED to its farthest valid sample; walk the current ray's valid out-depths current-depth-first, first out-depth from which the hop plans wins — linear, never out×entry quadratic. Widen entry only if the pinned search exhausts.
  - **Hop**: `plan_viewpoint` two-tier ladder between the slide-out pose and the pinned entry pose. The ONLY leg where roll may flip.
  - **Slide-in**: entry sample → closest sample, fixed IK branch, roll-locked, no ±π re-expression; straight joint interpolation validated segment-wise against `world`.
  - `None` → the address is transition-inaccessible (honest; caller marks blocked).
- Consumes: `world` = the LIVE planner world (ruling) at plan time; execution re-validates each leg against the world it was planned under via `rig.move(path, world=...)` (retreat precedent).

- [ ] **Step 1: failing tests** — pinned-entry search plans ≤ n_out_depths hops (count via stub planner); no-valid-out-depth → `None`; roll flip appears only between `out[-1]` and `slide_in[0]` (hop absorbs it); `current=None` → `out == []` and hop starts at `q_now`; slide-in waypoints reverse to exactly the forward list; every slide-in configuration is collision-free in the plan world.
- [ ] **Step 2: run to verify failure**
- [ ] **Step 3: implement** `transaction.py`
- [ ] **Step 4: run to verify pass**
- [ ] **Step 5: commit** `motion: plan_transaction — slide-out, hop, slide-in with pinned entry`

## Task 3: machine executes transactions

- Code: `inspection/run/machine.py`
- Test: `inspection/tests/test_machine.py`, `inspection/tests/test_execute_wait.py` (append)

**Interfaces:**
- `_plan_worker` branches: cell → `resolve_cell(self.world, ...)` + `plan_transaction`; face → `resolve_face(self.world, ...)` + `plan_transaction` (face findings now carry `{theta, shift, depth, pos}` from `RayResolution`); survey unchanged. `plan_to_cell` / `resolve_face_view` / all three `face_radius` call sites die.
- New phase `"adjusting"`: slide legs run under it; the hop keeps `"executing"`. Whitelist in `_on_worker_phase` + `_BUSY_PHASES` + UI passthrough (no UI layout work — Stage C owns surfaces).
- `self._ray_state` (dispatcher-owned): set on arrival from the executed target resolution; cleared on retreat, fault, and failed settle. It is leg-1's `current`.
- `_arrival_path` becomes the executed transaction's waypoint record (legs concatenated with boundaries); the retreat's retrace tier reverses the EXECUTED waypoints — mid-slide fault retraces from the fault point back through hop and out (design: retreat machinery as executor of a normal state).
- Preview publishes the concatenated path exactly as today plus a `legs` boundary index list (additive payload field).
- `pose_active`/`_pose_ack` armed once, held across legs; capture only after slide-in.

- [ ] **Step 1: failing tests** — full 3-leg lifecycle on the fake rig (request → adjusting/executing/adjusting phase trace → capture at slide-in end); first move after survey = 2 legs (no slide-out); transition-inaccessible → refusal names the class, arm never moves; mid-slide-in fault (fake rig scripted stop) → retreat retraces the reversed executed waypoints exactly; `_ray_state` cleared after retreat; per-leg `rig.move` receives the plan world.
- [ ] **Step 2: run to verify failure**
- [ ] **Step 3: implement** (worker executes legs sequentially; keep the arrived-check against `slide_in[-1]`)
- [ ] **Step 4: run to verify pass** (`test_machine`, `test_flow_a`, `test_live_faces`, `test_execute_wait`, `test_collect`)
- [ ] **Step 5: commit** `machine: three-leg move transaction — adjusting phase, ray state, retrace of executed legs`

## Task 4: sweep ranks transactions

- Code: `inspection/run/collect.py`
- Test: `inspection/tests/test_collect.py` (append)

- [ ] **Step 1: failing test** — sweep candidate cost = transaction joint-L1 on the private mirrored world; a cell whose transaction is `None` is dropped from ranking (not crashed on).
- [ ] **Step 2: run to verify failure**
- [ ] **Step 3: implement** `_default_planner` via `resolve_cell` + `plan_transaction` on the sweep's own world (`_mirror_object_box` unchanged).
- [ ] **Step 4: run to verify pass**
- [ ] **Step 5: commit** `collect: sweep ranks ray transactions`

## Task 5: guard-world replay bench — the Stage B gate

- Code: `inspection/investigation/ray_views/ray_replay.py` (add `--world guard`)
- Evidence: `inspection/investigation/ray_views/evidence/`

The ruling makes live sampling validate against `object_box` + padding, not
the boxguard box the Stage A bench used. Measure what that honestly costs.

- [ ] **Step 1:** `--world guard` mode: world carries `object_box(fused cloud)` exactly as `_recenter` builds it (same helper), boxguard prim still provides addresses/containment corners.
- [ ] **Step 2:** run both modes over the 4 runs; per-address delta table (boxguard-world vs guard-world endpoint) + refreshed renders into `evidence/`.
- [ ] **Step 3: GATE — present to Anton:** expected give-back ~2–4 cm; verdicts must hold (still closer than the old shell on the clear majority). If guard-world sampling flips the verdicts, STOP — the obstacle design question reopens before Stage C.
- [ ] **Step 4: commit** `ray_views: guard-world bench — what live-world sampling costs`

## Task 6: cross-tier e2e (boundary test)

- Test: `inspection/tests/e2e/` (extend the existing flow harness)

- [ ] **Step 1: failing test** — one Flow-A question run on the fake rig end to end: survey → sweep step → AI face request → 3-leg transaction → capture → step record carries `{method, address}` unchanged and the face finding's `{theta, shift}` reaches the brain's view state; a scripted mid-move fault mid-run leaves the run alive and answerable (brain atomicity gate honored).
- [ ] **Step 2–4:** verify failure → implement glue → verify pass.
- [ ] **Step 5: commit** `e2e: flow A over ray transactions on the fake rig`

## Self-review notes

- Spec coverage: transaction ✓ (T2/T3), adjusting phase ✓ (T3), resolution
  replacement ✓ (T1/T3), sweep ✓ (T4), honest world-consistency evidence ✓
  (T5), cross-tier e2e ✓ (T6, per plans-need-cross-tier-e2e-tasks). Menu
  marks / prompt reality / schema / UI markers = Stage C by design.
- Writer's calls made here (flag if wrong): `adjusting` = slide legs,
  `executing` = hop; `ViewState.r` stopgap = mean resolved cell depth until
  Stage C; retrace reverses executed waypoints only (never re-plans).
- Risk watched: `_reach` value-type change ripples into `_view_state`,
  `_set_phase` totals and the sweep — T1 sweeps all three in one commit.
