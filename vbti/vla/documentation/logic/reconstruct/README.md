# `logic/reconstruct`

## Purpose

`logic/reconstruct` turns videos/photos and composed scenes into simulation assets and Isaac/LeIsaac tasks.

## Files

| File | Purpose |
|---|---|
| `master.py` | Phase-based Fire CLI orchestrator. |
| `video_utils.py` | Video metadata, rotation fix, sharp frame extraction. |
| `colmap_utils.py` | Nerfstudio/COLMAP processing and undistortion. |
| `gs_milo_utils.py` | MILo GS training and SDF mesh extraction. |
| `clean_mesh.py` | Interactive Polyscope mesh cleaning. |
| `format_utils.py` | Splat/GLB/mesh to pointcloud/USD conversion. |
| `robot_utils.py` | Robot USD inspection/fix/drives/config extraction. |
| `isaac_cfg_utils.py` | Scene config extraction and LeIsaac/IsaacLab generation. |
| `cosmos_transfer.py` | Cosmos Transfer preparation pipeline. |

## Docs

- `pipeline.md` - full reconstruction sequence.
- `video_colmap_milo.md` - frame extraction, COLMAP, MILo.
- `usd_isaac.md` - USD conversion and Isaac/LeIsaac generation.
- `scripts.md` - `scripts/3d` and `scripts/sim` utilities.
- `cosmos.md` - Cosmos prep status and commands.

## Status

This is a prepared simulation extension path. It is not the main validated claim unless simulation-generated data is evaluated through the real protocol loop.
