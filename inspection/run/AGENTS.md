# run — orchestration

## Purpose
Top-level entry points: the exploration loop and demo scripts. This is
the only layer allowed to talk to everything — perception, cell, view,
motion — and wire them together.

## Files
- `explore.py` — interactive azimuth-orbit exploration loop with the
  three-panel dashboard (live RGB | fused side view | top-down ring).
  SO-101 era; the UR5e loop will supersede it.

## Contracts & decisions
- Nothing imports from `run/`. If a sibling package needs something
  defined here, that something is in the wrong place — push it down.
- Orchestration only: sequencing, UI, logging, entry points. Any
  geometry, planning, or camera logic that accretes here gets moved to
  its pipeline stage.
- The VLM slot: it replaces exactly the viewpoint picker inside the
  loop, nothing else.

## Does NOT belong here
- Reusable logic of any kind — this layer is glue, not a library.
