## 1. OpenSpec scaffolding

- [x] 1.1 Author `proposal.md` / `design.md` / `tasks.md` (stepwise with Anton)
- [ ] 1.2 Author `specs/{oli-localization,oli-mapping,oli-navigation}/spec.md`
- [x] 1.3 Branch `33may/may-173-reason-module-separation` off `33may/nav-localization-research`

## 2. `reason/localization/` — contracts + module protocol (D1–D6, D11)

- [x] 2.1 TDD: `contracts.py` — `RobotPose` (moved from `nav/types.py`, unchanged),
  `LocalizationStatus`, `LocalizationIn` (frame-paced bundle, D2), `LocalizationOut`
  (pose None iff LOST invariant, D3), `LocalizationSetup` (D4); construction validation tests
- [x] 2.2 `module.py` — `LocalizationModule` protocol (`start/step/stop`, D5) + a protocol-conformance
  test helper (any impl can be checked: map-frame output, monotonic stamps, LOST⇔pose-None)
- [x] 2.3 Move `Localizer` protocol + `GroundTruthLocalizer` + `DebugPoseLocalizer` here
  unchanged (D11); their tests move along
- [x] 2.4 Invariance check: package imports numpy+stdlib only; `brain`-marked suite green

## 3. `reason/mapping/` — Map contract + static realization (D7)

- [x] 3.1 Move `costmap.py` (`OccupancyGrid`) + `occupancy_io.py` here unchanged; tests move along
- [x] 3.2 TDD: `Map` dataclass (grid + monotonic `version` + `stamp_ns`) + `MappingModule` protocol
- [x] 3.3 TDD: `StaticMapping(map_dir)` — serves the same `Map` forever (version constant);
  docstring records the world_representation growth path (D12)

## 4. `reason/nav/` — Planner (D8, D9)

- [x] 4.1 TDD: `Planner.plan(pose, goal, map)` happy path — full plan on new goal, local
  horizon re-plan + tail splice while goal holds, full fallback on local failure
  (behavior parity with today's `nav.py:93-112` — port the existing tests first, they must
  pass against the new class)
- [x] 4.2 TDD: robot-layer derivation cached on `map.version` — same version ⇒ derived exactly
  once; version bump ⇒ rebuild AND cached path dropped (D9)
- [x] 4.3 TDD: no-goal / blocked-goal ⇒ `None`; `.path` property reflects last plan
- [x] 4.4 `plan_path()` stays pure and untouched; `Planner` only wraps it

## 5. `reason/nav/` — Nav shrink (D10)

- [x] 5.1 Rewrite `Nav` as the thin orchestrator: goal state, module refs, per-tick
  `localize → mapping.latest() → planner.plan → follower.command`, hold semantics, mode stamping
- [x] 5.2 Port existing `Nav` tests — behavior parity: hold on no-goal/no-pose/no-path, intent
  emission, goal set/clear; `PurePursuit` + `ArmedNav` untouched
- [x] 5.3 `nav/types.py` keeps `GoalCoordinate` only; package `__init__` exports updated

## 6. Rewire callers — no shims

- [x] 6.1 `devapp/brain_link.py`: construct `StaticMapping` + `Planner` + thin `Nav`; imports off
  the new packages
- [x] 6.2 Sweep remaining importers (dev_app sources/panels, converter CLI entry, any scripts)
  — grep-audit `reason.nav` imports repo-wide
- [x] 6.3 Test tree mirrors packages: `tests/oli/reason/{localization,mapping,nav}/`

## 7. Verification — behavior preserved

- [x] 7.1 Full `brain` suite green (was 284+; count only grows)
- [x] 7.2 Live smoke: `launcher --sim isaac --mode glide --dev-app` with map + debug-pose —
  click-goal → plan → armed drive works exactly as before the split (Anton at the stick)
- [ ] 7.3 Planning perf unchanged: full plan ~ms-scale, local re-plan sub-ms (spot-check logs)

## 8. Docs & memory

- [x] 8.1 Per-package `AGENTS.md` (+ one-line `CLAUDE.md` shim): module contract, invariants,
  what may import what — localization/, mapping/, nav/
- [ ] 8.2 Update agent memory `architecture-reason-module-separation` with build outcome/gotchas
- [ ] 8.3 Daily note block (draft → approve → append)
