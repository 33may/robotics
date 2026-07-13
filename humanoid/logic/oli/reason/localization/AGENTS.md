# localization/ — the pose-source module

Answers **"where am I"**: turns camera frames + proprioception into a map-frame SE(2)
`RobotPose`. Read `../AGENTS.md` first (module map, hard rules, the Protocol pattern).

## The two sockets (do not conflate them)

| seam | shape | who implements it | who calls it |
|---|---|---|---|
| `Localizer` (localizer.py) | `estimate(observation[, camera_frame]) -> Optional[RobotPose]` | thin in-brain pose READERS: `GroundTruthLocalizer`, `DebugPoseLocalizer`, (later) the live SLAM client | `Nav`, every control tick |
| `LocalizationModule` (module.py) | `start(LocalizationSetup)` → `step(LocalizationIn) -> LocalizationOut` → `stop()` | the HEAVY algorithm (RTAB-Map, cuVSLAM, bench candidates) | its hosts: the locbench harness (bag replay) and the live out-of-process node |

They meet at a datagram, never in a class hierarchy: the live node hosts a `LocalizationModule`
and emits `LocalizationOut`; a ~10-line `Localizer` client in the brain reads the newest one,
checks `status`, unwraps the pose. Nav never changes.

## Contract invariants (self-enforced in contracts.py — do not weaken)

- `LocalizationIn` is **frame-paced** (one per camera tick; nearest `Observation` ≤ stamp rides
  along) and **sensor-only** — NEVER add the module's own previous pose (echo-chamber + makes
  bags unreplayable; statefulness lives INSIDE the module, that's why the lifecycle exists).
- `LocalizationOut.pose` is **always map frame** (the module got the map + initial-pose hint in
  `LocalizationSetup` and owns internal alignment). Downstream never anchors or refits — a
  constant bias must stay visible (it is the measured nav-killing failure mode).
- `pose is None ⇔ status is LOST`; `status` is coerced through `LocalizationStatus` (it crosses
  a wire as an int); `DRIFTING` = propagated by VO since the last map fix (usable, aging).
- No `reset()` in v1 — known-start only (design.md D5).

## Rules

- Imports: `oli/contracts.py` + stdlib/numpy ONLY. Never `..mapping`, never `..nav`, never
  world SDKs. (Enforced: `tests/oli/reason/test_architecture.py`.)
- Algorithm-private 3D maps (RTAB-Map `.db`, feature dbs) are INTERNAL to a module, injected as
  an opaque `LocalizationSetup.map_dir` — they are not system citizens (design.md D12).
- Map BUILDING is not in the module protocol: a candidate ships a bench-only
  `build_map(bag_dir) → map_dir` entry beside its module (design.md D14).

## Adding a realization (checklist)

1. Implement the shape (no inheritance needed — see the Protocol pattern in `../AGENTS.md`).
2. Add a static canary + `isinstance` test in `tests/oli/reason/test_architecture.py`.
3. `LocalizationModule` impls: run `testing.verify_module_contract` in your tests — it is the
   conformance gate (raises `ContractViolation`; works under `python -O`). Note `stop()` must
   tolerate being called after a failed `start()`.
4. Accuracy is judged by the locbench scorer against ground truth, not by any type check.

`testing.py` ships with the package ON PURPOSE (importable by the bench and candidate suites in
other envs) — do not move it into `tests/`.
