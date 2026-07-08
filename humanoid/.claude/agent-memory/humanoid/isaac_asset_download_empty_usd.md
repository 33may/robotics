---
name: isaac-asset-download-empty-usd
description: Isaac Sim env .usd grabbed via Asset Browser single-file download or Save-As is an empty crate; real geometry needs Collect As / cloud stream / offline pack
metadata:
  type: reference
---

Downloading a single NVIDIA environment `.usd` (e.g. `full_warehouse.usd`) via the Isaac
Sim Asset Browser "Download", or doing `File > Save As` on it, yields a near-empty USD crate
with **no geometry**. The scene's meshes/materials live in a referenced payload + texture
tree that a single-file grab does not pull along.

Verified 2026-07-06 (MAY-171): `assets/envs/warehouse_nvidia/{warehouse_edit,warehouse_multiple_shelves}.usd`
were 492-byte crates whose `GetRootLayer().ExportToString()` was just `#usda 1.0` (0 prims,
no references, no sublayers). Booting a world that references them renders nothing.

To get real geometry, pick one:
- **Stream:** reference the S3 cloud root directly —
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/full_warehouse.usd`.
  USD resolves the whole dependency tree over HTTPS. Needs network; first load is slow.
- **Collect As:** open the scene in Isaac Sim → `File > Collect As...` → bundles USD + payloads
  + textures into a self-contained in-repo folder. Matches MAY-171's fixed ground-truth world.
- **Offline pack:** the 3-part `isaac-sim-assets-complete-5.0.0.zip.00{1,2,3}` → local
  `Isaac/Environments/...` tree, reference locally.

Isaac Sim 5.0's default `persistent.isaac.asset_root.default` already points at that S3 bucket.
Boot path context: [[vendor_patch_sim_freeze_until_cmd]] (glide world), warehouse loads at
`glide_world_main.py` next to `add_default_ground_plane`.

**Resolved 2026-07-06 (what worked):** mirrored the whole `Simple_Warehouse/` S3 prefix (public
ListObjectsV2 works: `?list-type=2&prefix=...`) **preserving the on-server dir layout** so every
relative ref resolves with zero USD editing — incl. full_warehouse's one `../../Props/Forklift/`
escape (mirror `Isaac/Props/Forklift/` too). Textures are in-prefix (`Materials/Textures/*.png`).
Reproduced by `assets/envs/warehouse_nvidia/fetch_warehouse.py` (~576 MB, gitignored). Validated
via `UsdUtils.ComputeAllDependencies` → 1851 layers + 161 textures resolve, 26k prims open.
**Gotcha:** standalone USD tooling reports `OmniPBR.mdl` UNRESOLVED — it is a renderer MDL
built-in (resolved at render time in Isaac), NOT a missing asset. Don't chase it.
