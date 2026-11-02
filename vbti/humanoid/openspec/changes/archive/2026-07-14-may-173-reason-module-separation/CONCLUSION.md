# Conclusion — Reason split into localization / mapping / nav: shipped

**Status:** closed 2026-07-14. Core shipped and on `main`; four bookkeeping tasks deferred (below).

## What it delivered

Unbundled `Nav`'s tangled responsibilities into **one package per responsibility**, joined by
data contracts instead of shared objects:

- `localization/` — "where am I" → emits `RobotPose` / `LocalizationOut` (map frame).
- `mapping/` — "what does the world look like" → emits `Map` (grid + version + stamp).
- `nav/` — "get to a goal" → emits `Intent`, the only place module refs live (the orchestrator).

Import direction is enforced (nav→localization, nav→mapping, localization ⊥ mapping) and the
Protocol seams (`Localizer`, `MappingModule`, `LocalizationModule`) are checked three ways:
Pyright static canaries, self-validating dataclasses, and `verify_module_contract`. All pinned by
`tests/oli/reason/test_architecture.py`.

## Why it matters downstream

This is the foundation the entire locbench harness (`may-173-locbench-harness`) and the AI
dev-loop (`may-173-locdev-flow`) build on: a localization candidate is exactly a
`LocalizationModule` realization, hosted behind these contracts, scored without the rest of the
brain ever knowing which algorithm ran. Verified live (localize→plan→follow on the warehouse map)
and by 516 green brain tests.

## Deferred (non-functional bookkeeping)

- §1.2 author `specs/{oli-localization,oli-mapping,oli-navigation}/spec.md` deltas (capture-not-
  encode; skipped, as with the sibling changes).
- §7.3 planning-perf spot-check (full plan ms-scale, local re-plan sub-ms).
- §8.2 refresh the `architecture-reason-module-separation` memory leaf.
- §8.3 daily-note block.

The code is complete and load-bearing; these are docs/telemetry tails, captured here so nothing is
lost by archiving.
