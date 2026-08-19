# view — viewsphere & coverage

## Purpose
Enumerate candidate viewpoints on a sphere around the object primitive
and track which have been seen. Later, the VLM picks cells from this
enumeration — it replaces the picker, nothing else.

## Files
- `viewsphere.py` — v2, UR5e stack. `ViewSphere(center, r)`: {h, v} cells
  (12 azimuth × 3 elevation bins, ONE shell of configurable radius, h0
  faces the base — azimuth computed from the actual center), cell →
  camera pose → flange pose via nominal `T_FLANGE_CAM` (replace after
  hand-eye), `reachability()` map, `plan_to_cell()` = roll search + the
  motion ladder. `p inspection/view/viewsphere.py [--r R] [--demo]`.

## Contracts & decisions
- Camera ROLL about the boresight is a free parameter: upright preferred,
  rolled variants tried before declaring a cell blocked. Image rotation
  is acceptable; unreachability is not.
- The object is a HARD obstacle in the world (2026-08-18 decision) —
  no separate standoff; endpoint validity at env padding is the standoff.
  Radius must respect the tool: closed tip rides ~19 cm AHEAD of the
  camera (r < ~0.30 physically pierces the object).
- Coverage = a bitmap over the enumerated cells, not surface patches.
- Blocked cells stay visible as facts, invisible as actions.
- Ring elevations 10/40/70 deg above the table (Anton 2026-08-18). Higher
  rings crowd near the pole (at 70 deg the ring radius is cos70 ≈ 0.34 r), so
  a top-heavy set gives far less view diversity than the cell count suggests.
- Measured (r=0.35, demo cup, 10/40/70): 31/36 cells reachable. The 10 deg
  ring loses the four cells nearest the base — a near-horizontal view there is
  blocked by the arm's own column.

## Does NOT belong here
- IK / reachability solving internals (motion/), object modeling
  (cell/), capture (perception/), the loop that walks the cells (run/).
