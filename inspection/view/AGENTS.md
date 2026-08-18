# view — viewsphere & coverage

## Purpose
Enumerate candidate viewpoints on a sphere around the object primitive
and track which have been seen. Later, the VLM picks cells from this
enumeration — it replaces the picker, nothing else.

## Files
- `viewsphere.py` — {h, v} cell addressing (12 azimuth × 3 elevation
  bins, single shell), cell → camera pose, IK-filtered reachability map.

## Contracts & decisions
- Coverage = a bitmap over the enumerated cells, not surface patches.
  A cell is visited or not; no partial-visibility bookkeeping.
- The viewsphere radius guarantees camera standoff by construction — no
  separate proximity check is needed for the camera itself.
- Blocked cells stay visible as facts, invisible as actions.

## Does NOT belong here
- IK / reachability solving internals (motion/), object modeling
  (cell/), capture (perception/), the loop that walks the cells (run/).
