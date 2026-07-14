## Why

The nav stack works (localize→plan→follow, verified live on the warehouse map) but its
responsibilities are bundled: `Nav` owns the costmap, the robot layer, plan pacing, goal state,
and the follow chain, while the pose seam (`Localizer`) hides inside the nav package. MAY-173's
next step — a real localization algorithm evaluated and iterated by an autonomous agent — needs
each concern behind its own explicit in/out contract: a localization module a candidate algorithm
can implement, a mapping module that can later grow into live world reconstruction, and a planner
that consumes whatever map is emitted rather than owning one.

This change is the **module separation refactor**: define the contracts, split the packages, and
rewire the existing (working) logic behind them. Pure restructuring — no new behavior; the test
suite stays green throughout. The evaluation harness (locbench, change b) builds on these
contracts.

## What Changes

- **NEW package `reason/localization/`** — the pose-source module.
  - Contracts: `LocalizationIn` (frame-paced bundle: stamp_ns, camera frames dict, nearest
    `Observation`, optional `Intent` motion prior), `LocalizationOut` (stamp_ns,
    `Optional[RobotPose]` in **map frame**, `status ∈ {TRACKING, DRIFTING, LOST}`,
    `last_fix_stamp_ns`), `LocalizationSetup` (opaque algorithm-private `map_dir`, initial-pose
    hint, calibration blob).
  - Module protocol: `LocalizationModule.start(setup) / step(loc_in) -> LocalizationOut / stop()`
    — the surface a candidate algorithm implements; hosted by the bench offline and by an
    out-of-process node live.
  - `RobotPose` **moves here** from `nav/types.py` (it is localization's output type; dependencies
    point nav→localization, never backwards).
  - The existing thin seam moves unchanged: `Localizer` protocol + `GroundTruthLocalizer` +
    `DebugPoseLocalizer` (the GT/debug shortcut stays a first-class realization — current demo
    keeps working).
- **NEW package `reason/mapping/`** — the world-truth module.
  - Contract: `Map` (`OccupancyGrid` + monotonic `version` + `stamp_ns`); protocol
    `MappingModule.latest() -> Optional[Map]`.
  - v1 realization `StaticMapping(map_dir)`: wraps `load_occupancy`, version never bumps.
  - `costmap.py` (`OccupancyGrid`) and `occupancy_io.py` **move here** (grid ops `inflate` /
    `clearance_cost` stay methods; the planner calls them).
  - Mapping is the designated embryo of a future `world_representation` module (3D/semantics);
    algorithm-private 3D maps stay inside localization until a second consumer exists.
- **CHANGED `reason/nav/`** — planning + following + orchestration only.
  - NEW `Planner` class over the pure `plan_path`: signature `plan(pose, goal, map)` — consumes
    the **emitted `Map` value, never imports the mapping module** (modules consume contracts, not
    each other). Owns the derived robot layer (inflate + clearance) cached on `map.version`, the
    path cache, goal-change detection, full/local re-plan with tail splice; a map-version bump
    rebuilds the robot layer AND drops the cached path.
  - `Nav` shrinks to a thin orchestrator: goal state, module refs, per-tick
    localize → `mapping.latest()` → plan → follow chain, zero-velocity hold semantics, mode
    stamping.
  - Unchanged: `PurePursuit` (follower; plan→motion transfer is later, separate work), `ArmedNav`,
    `GoalCoordinate`, path as bare `List[Tuple[float, float]]`.
- **Import fallout fixed properly** (`devapp/brain_link.py`, dev_app sources, tests) — no
  re-export shims; the committed test suite is the safety net.

## Capabilities

### New Capabilities

- `oli-localization`: the pose-source module — in/out contracts (`LocalizationIn` →
  `LocalizationOut`), the `LocalizationModule` protocol candidates implement, map-frame output
  semantics, and the thin in-brain `Localizer` seam with its GT/debug realizations. Externally
  observable: any conforming module (GT passthrough, offline bench candidate, live node client)
  drops into the same seam with zero planner change.
- `oli-mapping`: the world-truth module — the versioned `Map` contract, `MappingModule` protocol,
  static baked-map realization, and the version semantics downstream caches key on.
- `oli-navigation`: the navigation module post-split — thin `Nav` orchestration, the `Planner`
  contract `plan(pose, goal, map)` with its caching/invalidation semantics, follower and hold
  behavior.

### Modified Capabilities

- None in `openspec/specs/` (the nav stack predates spec-anchoring; these three specs anchor it
  now).

## Impact

- **Moved/split code**: `reason/nav/{localizer,types,costmap,occupancy_io}.py` →
  `reason/localization/` + `reason/mapping/`; `Nav` internals → `Planner`. Pure moves + new
  contract types; behavior-preserving.
- **Callers touched**: `devapp/brain_link.py`, dev_app map panel imports, `tests/oli/reason/nav/*`
  (split into per-module test dirs mirroring the packages).
- **Invariance**: all three packages stay brain-pure (numpy + stdlib; PIL/yaml lazy in
  `occupancy_io`) — the `brain` pytest marker guards them, unchanged.
- **Env**: `brain` only. No new dependencies.
- **Linear**: [MAY-173](https://linear.app/may33/issue/MAY-173) — prerequisite of the locbench
  harness change (b), which consumes `LocalizationModule` + `LocalizationIn/Out`.
