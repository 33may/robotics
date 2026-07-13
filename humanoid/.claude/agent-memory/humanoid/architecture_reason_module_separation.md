---
name: architecture-reason-module-separation
description: Locked design (2026-07-13) — reason/ split into localization/mapping/nav modules with data-contract seams; modules consume emitted values, never import each other
metadata:
  type: project
---

Ground-up module separation of the nav stack, designed stepwise with Anton 2026-07-13 (every
point his call). Package layout: **`reason/localization/`, `reason/mapping/`, `reason/nav/`** —
own responsibility in own folder.

- **Localization**: `LocalizationIn` (stamp_ns, frames dict, nearest Observation, optional Intent —
  frame-paced) → `LocalizationOut` (stamp_ns, Optional[RobotPose], status TRACKING/DRIFTING/LOST,
  last_fix_stamp_ns; `quality` dropped — algorithms won't honestly provide it). Module protocol
  `LocalizationModule.start(LocalizationSetup)/step/stop`; `LocalizationSetup` = map_dir (opaque,
  algorithm-private) + initial_pose hint + calibration. **Output is map-frame** (module owns
  internal alignment given the hint) → bench scores RAW, no scorer-side anchor. `RobotPose` MOVES
  from nav/types.py to localization/contracts.py (deps point nav→localization). GT/debug
  passthrough (`DebugPoseLocalizer`) stays as a seam realization — demo maintainability.
- **Mapping**: `Map` (OccupancyGrid + monotonic version + stamp_ns) + `MappingModule.latest() ->
  Optional[Map]`. v1 = `StaticMapping(map_dir)` wrapping load_occupancy, version never bumps.
  OccupancyGrid + occupancy_io move here. Partial/tile updates deferred ("will design it later").
  Mapping is the EMBRYO of world_representation — grows into 3D/semantics when a second consumer
  exists; algorithm 3D maps stay localization-internal until then (Anton's call).
- **Planner** (nav/): stateful class over pure `plan_path`; **decision B: consumes the emitted
  `Map` value — `plan(pose, goal, map)` — never imports MappingModule** (consistency: modules
  consume contracts, not each other; bus/OOP-ready). Owns robot layer (inflate+clearance) cached
  on map.version, path cache + full/local re-plan + goal-change detection (moved from Nav); map
  version bump invalidates cached path.
- **Follower**: PurePursuit kept as-is (plan→motion transfer is later separate work). Path stays
  bare List[Tuple] (no typed contract — Anton). ArmedNav unchanged.
- **Nav**: thin orchestrator — goal state, module refs, hold semantics, mode stamping. Tick:
  `pose=localizer.estimate; map=mapping.latest(); path=planner.plan(pose,goal,map);
  twist=follower.command(pose,path)`.
- **OpenSpec shape: TWO changes** — (a) this refactor (pure moves + contracts, suite green),
  (b) locbench harness on top. Live wiring: heavy module OOP, emits LocalizationOut datagrams
  (debug_pose pattern); in-brain thin client = third `Localizer` realization; Nav closed.

Links: [[architecture-locbench-harness]], [[research-nav-localization]],
[[feedback-plan-together-stepwise]], [[nav-brain-localizer-seam]].
