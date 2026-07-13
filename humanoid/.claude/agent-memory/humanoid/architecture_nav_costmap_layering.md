---
name: architecture-nav-costmap-layering
description: Nav costmap is layered — baked boolean map (world truth) vs runtime footprint/clearance (robot policy); clearance NOT baked into convert_ros_map
metadata:
  type: project
---

The nav costmap is split into two layers, and they live at different lifecycles by design:

- **Map layer = world truth, baked once.** `convert_ros_map` (`reason/nav/occupancy_io.py`) →
  boolean `occupancy.npy` + `occupancy.json`. Its only bake-time knobs are the ones deciding
  *occupied vs free* (`occupied_thresh`/`free_thresh`/unknown). A property of the WORLD.
- **Footprint/clearance = robot policy, runtime.** `OccupancyGrid.inflate(radius)` (hard,
  impassable) + `OccupancyGrid.clearance_cost(inflation_radius_m, weight)` (soft EDT gradient,
  fed to `plan_path(..., cost=)`). A property of the ROBOT.

**Why:** Anton asked (2026-07-10) whether the clearance gradient should be a `convert_ros_map`
param. Decided **no**: baking it freezes the robot radius/weight into the artifact, so a
different footprint or a tuning tweak forces a re-bake. Keeping it runtime means one baked map
serves any footprint, the knobs stay live-tunable (just re-plan, no re-bake), and it mirrors
Nav2's static-map-layer vs inflation-layer split — and Anton's own world-invariance instinct
(map is world, clearance is robot).

**How to apply:** never move footprint/clearance/inflation logic into the map bake. New robot-
side costmap concerns (extra penalty layers, oriented-polygon footprint) go next to `inflate`/
`clearance_cost` in `costmap.py`, owned at runtime by the caller — the MapPanel today (preview),
the Nav reason once execution is wired. See [[nav-brain-localizer-seam]]; clearance_cost is
admissible for A* because penalties are ≥0.
