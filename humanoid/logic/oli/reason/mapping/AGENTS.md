# mapping/ — the world-truth module

Answers **"what does the world look like"** as a versioned snapshot. Read `../AGENTS.md` first
(module map, hard rules, the Protocol pattern).

## Contract

`MappingModule.latest() -> Optional[Map]` — pull seam; `None` = no map yet (Nav holds).
`Map = (grid: OccupancyGrid, version: int, stamp_ns: int)`, frozen.

- **`version` is the load-bearing field**: it bumps ⇔ grid CONTENT changed. Downstream caches
  key on it (the planner's derived robot layer + path cache key on `(id(grid), version)`).
  `StaticMapping` never bumps → derivations happen exactly once. A future live reconstructor
  bumps on real updates and downstream rebuilds exactly then. Do not emit a mutated grid under
  an unchanged version — that is the one convention the type system cannot catch.
- **World truth ONLY.** Footprint inflation / clearance cost are the planner's ROBOT layer —
  never bake them into the map (measured decision: one baked map serves any footprint).
- Full-snapshot semantics; partial/tile updates are a deferred design ("will design it later").

## Realizations

- `StaticMapping(map_dir)` — v1: the baked artifacts (`occupancy.npy` + `occupancy.json`,
  produced by `occupancy_io.py` from the Isaac omap bake). `from_grid(grid)` for tests/harnesses.
- (future) live reconstruction — same seam, fed by frames+poses, possibly the same OOP process
  as the SLAM localizer serving both seams.

## Growth path — read before adding 3D/semantics anywhere

This module is the designated **embryo of `world_representation`** (decision with Anton,
2026-07-13): when dynamic reconstruction / semantic maps / the reasoning demo need a 3D answer,
`latest()` grows richer views and the 2D grid becomes the planner's projection. Widen THIS seam
when a second consumer with a 3D need actually exists — do not create a parallel world module.
Algorithm-private 3D maps stay inside `localization/` (opaque `map_dir`).

## Rules

- Imports: stdlib/numpy (+ PIL/yaml/scipy LAZY inside functions — keep the load path light).
  Never `..localization`, never `..nav`, never world SDKs.
  (Enforced: `tests/oli/reason/test_architecture.py`.)
- New realizations: static canary + `isinstance` test in the architecture guard (see
  `../AGENTS.md` checklist).
