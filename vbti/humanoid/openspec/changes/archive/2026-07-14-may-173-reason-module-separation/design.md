# Design — reason module separation (oli-localization / oli-mapping / oli-navigation)

## Context

The nav stack shipped working (costmap → A* → pure-pursuit behind the `Localizer` seam, verified
live on the warehouse map) but grew as one package: `Nav` owns map state, planning policy, and
follow; the pose seam lives inside nav. MAY-173 now needs a **real localization algorithm,
implemented and iterated autonomously against ground truth** — which demands explicit module
contracts a candidate can implement and a bench can host. Designed stepwise with Anton
2026-07-13; every decision below is his call (agent memory
`architecture-reason-module-separation`).

## Goals / Non-Goals

**Goals**

- One responsibility per package: `reason/localization/`, `reason/mapping/`, `reason/nav/`.
- Modules communicate via **emitted data contracts, never module imports** — bus/OOP-ready.
- The localization module contract is **host-agnostic**: the offline bench (change b) and the live
  node (MAY-173 endgame) host the identical object.
- Behavior-preserving: the existing demo (GT-pose glide nav) works unchanged; suite stays green.

**Non-Goals**

- The harness itself (locbench — change b), any real SLAM candidate, the live OOP node.
- Plan→motion improvements (follower kept as-is; separate later work).
- Dynamic mapping / partial map updates (contract leaves room; design deferred).
- A `world_representation` module (deferred until broader semantics; mapping is its embryo).

## Decisions

### D1 — Package per responsibility

`reason/localization/` (pose), `reason/mapping/` (world truth), `reason/nav/` (plan + follow +
orchestrate). Dependency direction: nav → localization, nav → mapping; localization and mapping
independent of each other and of nav.

### D2 — `LocalizationIn` is the frame-paced input bundle

One `LocalizationIn` per camera tick (~15–30 Hz): `stamp_ns`, `frames: dict[str, CameraFrame]`,
nearest `Observation` ≤ stamp (joints → FK extrinsics, IMU), `Optional[Intent]` (commanded twist
as motion prior). Localization is vision-paced — the ~100 Hz Observation attaches to the frame,
not the reverse. This bundle is also the bag's record unit (change b).

### D3 — `LocalizationOut` carries status, not confidence

`stamp_ns`, `Optional[RobotPose]`, `status ∈ {TRACKING, DRIFTING, LOST}` (pose None iff LOST),
`last_fix_stamp_ns`. Three statuses because that is the machine loc-mode SLAM actually is: map
fix / VO-propagated / lost — and the E2 experiment says the brain must treat them differently.
`last_fix_stamp_ns` is the objective trust signal (`now − last_fix` maps onto the E2 budget).
`quality` was dropped — algorithms won't honestly provide a calibrated confidence.

### D4 — Map-frame output is the module's job

`LocalizationSetup` hands the module its map artifacts + an initial-pose hint + the calibration
blob; the module owns internal alignment and always emits **map-frame** SE(2). Consequence: the
bench scores RAW estimate vs GT — no anchoring/refit step anywhere downstream, so the E1-binding
constant bias has nowhere to hide.

### D5 — Module lifecycle: `start(setup) / step(loc_in) / stop()` — no `reset` in v1

Known-start assumption (research: no kidnapped-robot recovery in v1). A `reset(pose_hint)` is
added when a candidate actually supports recovery, not before.

### D6 — `RobotPose` moves to `localization/contracts.py`

It is localization's output type. Moving it keeps dependencies pointing nav→localization, never
backwards. `GoalCoordinate` stays in nav (nav's own input).

### D7 — `Map` = grid + monotonic version + stamp; pull seam

`MappingModule.latest() -> Optional[Map]`; `Map(grid: OccupancyGrid, version: int, stamp_ns:
int)`. `version` bumps only on content change — downstream caches key on it. v1 realization
`StaticMapping(map_dir)` wraps `load_occupancy`, never bumps → today's behavior and cost
preserved exactly. `latest()` full-snapshot only; partial/tile updates deliberately deferred.
`OccupancyGrid` + `occupancy_io` move to `mapping/` (grid ops `inflate`/`clearance_cost` stay
methods; the planner calls them — map = world truth, robot layer = planner's, per the costmap
layering decision).

### D8 — Planner consumes the emitted `Map` value, never the mapping module

`Planner.plan(pose, goal, map)`. Chosen over holding a `MappingModule` ref (Anton: "B is
better"): (1) consistency — modules in this brain consume emitted contracts, not each other
(localization doesn't import a camera module either); (2) bus/OOP-ready — when mapping emits
snapshots over a channel, nothing in Planner changes. The pull duty (`mapping.latest()`) belongs
to Nav — that is the orchestrator's job.

### D9 — Planner owns its derivations and caches

Private to `Planner`: derived robot layer (inflated grid + clearance field) cached keyed on
`map.version`; path cache + goal-change detection (new goal → full A*; same goal → local
horizon re-plan + tail splice; local-fail → full fallback — today's `nav.py` logic behind the
module wall). A `map.version` bump rebuilds the robot layer **and drops the cached path** (a
changed world invalidates the spliced tail — a correctness win the bundled code couldn't
express). Pure `plan_path()` stays as-is underneath.

### D10 — Nav shrinks to a thin orchestrator

Goal state (`set_goal`/`clear_goal`), module refs, per-tick chain
`localize → mapping.latest() → plan → follow`, zero-velocity hold semantics, mode stamping.
Unchanged: `PurePursuit`, `ArmedNav`, `GoalCoordinate`, path as bare `List[Tuple[float, float]]`.

### D11 — The GT/debug shortcut stays a first-class seam realization

`Localizer` protocol + `GroundTruthLocalizer` + `DebugPoseLocalizer` move to `localization/`
unchanged — the current demo must keep working (maintainability). The live SLAM client later
becomes the third ~10-line realization of the same protocol: read newest `LocalizationOut`
datagram, check status, unwrap `RobotPose`. Nav never changes.

### D12 — Algorithm 3D worlds are localization-internal

An algorithm's map (RTAB-Map `.db`, feature db, …) is private module state injected as an opaque
`map_dir` — formats are mutually incompatible; a shared 3D contract would be a fake abstraction.
The brain's shared world = mapping's `Map`. When broader semantics need a 3D answer, mapping
grows into `world_representation` — the seam is already placed; widen when a second consumer
exists.

### D13 — Two-host principle: this refactor is what makes the bench meaningful

The same `LocalizationModule` object is hosted by (1) the **bench** (`logic/locbench/`, change b):
replay a frozen bag as `LocalizationIn` sequence → `step()` → score raw map-frame output vs GT
against the E1 gates; and (2) the **live node** (MAY-173 endgame): fed from sockets, emitting
`LocalizationOut` datagrams (debug-pose pattern) to the thin in-brain client. A candidate that
passes offline differs from the live robot only by transport. The bench imports nothing from
`reason/` except these contracts (brain-pure, loadable in any env).

### D14 — A candidate is two artifacts

The `LocalizationModule` implementation (hosted by bench AND live) plus a bench-only
`build_map(bag_dir) → map_dir` entry. Map building does not pollute the live module protocol —
live never builds maps (until dynamic mapping, much later).

## Risks / Trade-offs

- **Move churn** — imports break across `devapp/brain_link.py`, dev_app sources, tests. Fixed
  properly (no re-export shims); the committed suite is the net. Refactor lands as one reviewed
  change while the stack is still small.
- **Frame-paced `LocalizationIn`** assumes localization never needs >30 Hz — true for glide and
  for every surveyed candidate; revisit only if a high-rate VIO joins.
- **Full-snapshot `Map`** could get heavy with a large live map — explicitly deferred; the
  `version` field is the hook a patch scheme would key on.
- **No `reset` in v1** — a lost module stays lost until restart; acceptable under the known-start
  demo scope, revisit with recovery-capable candidates.

## Migration Plan

1. Create `localization/` + `mapping/` packages with contracts + moved code (tests move with
   their modules).
2. Introduce `Planner`; shrink `Nav`; rewire `brain_link.py` and dev_app imports.
3. Full suite green in `brain` env; live glide-nav smoke (GT localizer) unchanged.
4. Change (b) — locbench — then builds only on the new contracts.

## Open Questions

- None blocking. Deferred by decision: partial map updates (D7), `reset` lifecycle (D5),
  `world_representation` split (D12).
