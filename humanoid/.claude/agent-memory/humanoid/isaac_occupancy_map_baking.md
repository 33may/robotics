---
name: isaac-occupancy-map-baking
description: Isaac occupancy-map generator (isaacsim.asset.gen.omap) — static geometry bake, GUI gotchas, warehouse nav map v1 values
metadata:
  type: reference
---

**Isaac's occupancy-map generator (`isaacsim.asset.gen.omap`, Tools ▸ Robotics ▸ Occupancy Map) is a static geometry bake, not a sim** — CPU PhysX scene-query raycasts (a virtual lidar) against frozen collision geometry, flood-filled from an unoccupied origin; no dynamics stepped. RTX-Lidar mode samples original triangle meshes instead (use when geometry has no colliders → PhysX mode sees nothing → all-free map). Full research writeup: `Documents/vbti/vbti/humanoid/assets/isaac_occupancy_map_baking.md`.

**Gotchas (cost real time on 2026-07-10):**
1. **Origin seed must sit in free air, off the floor plane (Origin Z ≈ 0.1) — if it's on the floor collider the flood-fill can't start and the whole map comes back blank/all-white.** Lower-bound Z can stay 0.
2. Bounds define the *only* mapped region; out-of-bounds = blocked in the brain grid → use **BOUND SELECTION** on the floor prim to cover the full drivable footprint.
3. Open-frame shelves are porous — use a **tall Z-column band** (floor+ε → top-of-rack, e.g. 0→2.5 m) so any pallet/beam anywhere in the column stamps the cell occupied; else the planner routes into empty bays. Robust fix is footprint keep-out (author blocker prims or OR-in rack AABBs), then morphological close + inflate.
4. No "Save YAML" button in this build — copy the YAML text block by hand into a `.yaml` next to the PNG; keep `image:` pointing at the real filename.

**Warehouse nav map v1 baked:** `assets/envs/warehouse_nvidia/nav_maps/v1/` — `resolution: 0.05`, `origin: [-26.925, -23.425]`, 33×55 m, 659×1100 px. Converted PNG→`occupancy.npy` (bool, `flipud` since PNG row 0 = top but ROS origin = bottom-left) + `occupancy.json` for the A* planner (`nav/occupancy_io.py`). NB the PNG filename says `10cm` but it's a 5 cm/0.05 map — Anton knows, kept the name. Links: [[architecture-nav-costmap-layering]], [[isaac-asset-download-empty-usd]].
