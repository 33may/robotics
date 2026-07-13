# reason/ — the Brain (world-invariant decision layer)

`Reason` decides **what** to do and emits `intent` (`PolicyIn`). Since change
`may-173-reason-module-separation` it is split into **one package per responsibility**, joined
by data contracts — read this file before touching any of them.

## The module map

| package | responsibility | emits (contract) | consumes |
|---|---|---|---|
| `localization/` | where am I | `LocalizationOut` / `RobotPose` (map frame) | `LocalizationIn` (frames + obs + intent) |
| `mapping/` | what does the world look like | `Map` (grid + version + stamp) | (v1: baked artifacts; later: frames+poses) |
| `nav/` | get to a goal | `Intent` (body twist) via `PolicyIn` | `GoalCoordinate`, `RobotPose`, `Map` |

## Hard rules (enforced by `tests/oli/reason/test_architecture.py` — red there = your change is wrong, not the test)

1. **Invariance boundary:** no package here imports `isaacsim` or `limxsdk`, directly or
   transitively. The `brain` pytest marker + the architecture guard both check it.
2. **Import direction:** `nav → localization`, `nav → mapping`, and **localization ⊥ mapping**
   (no edge either way). Never add a reverse import.
3. **Modules consume emitted contracts, never each other's module objects.** The canonical
   example: `Planner.plan(pose, goal, world_map)` takes the `Map` VALUE; it must never hold a
   `MappingModule` ref (design.md D8). Nav is the only place module refs live — it is the
   orchestrator.
4. **Emit intent, never joint commands** — turning intent into actions is `PolicyRunner`'s job.
5. **Consume invariant inputs only** — `Observation` / `CameraFrame`; never world-specific structs.

## The Protocol pattern (how "interfaces" work here — read if you come from Java)

Seams are `typing.Protocol` classes (structural interfaces): `Localizer`, `MappingModule`,
`LocalizationModule`. **Nothing inherits from them** — a class satisfies a protocol by having
the right method shapes, and the relationship is checked at the *wiring site* (the annotated
constructor parameter), not at declaration. A protocol in this codebase means "we intend to
swap this"; a concrete class means "there is one of these" (that is why `Planner` has no
protocol — one realization exists; extract a protocol only when a second is real).

Conformance is NOT left to convention — three enforcement layers, all mandatory:

- **Pyright at wiring sites + static canaries** in `tests/oli/reason/test_architecture.py`
  (`_static_…: Localizer = …` — the declaration-site check, Java-style);
- **self-validating contracts** — the dataclasses raise on malformed data
  (`LocalizationOut.__post_init__`: status coerced through the enum, pose⇔LOST invariant);
- **executable conformance** — `localization/testing.py::verify_module_contract` runs any
  `LocalizationModule` through its lifecycle and raises `ContractViolation` on drift.

**When you add a new realization of any protocol, you MUST add** (a) a static canary line +
`isinstance` test in the architecture guard, and (b) for `LocalizationModule` impls, a
`verify_module_contract` call in its test suite. No exceptions — this replaces Java's compiler.

## Where things are decided

Design + rationale: `openspec/changes/may-173-reason-module-separation/{proposal,design}.md`
(D1–D14). Per-package details: each package's own `AGENTS.md`. The localization evaluation
harness (locbench) builds on these contracts — see the `may-173-locbench-harness` change.

See `docs/architecture/architecture.md` §6–§7.
