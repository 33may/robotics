---
name: reconstruct-pipeline-milo
description: The project's own COLMAP→3DGS→MILo→mesh reconstruction stack lives in robotics/vbti/logic/reconstruct; ~/Pictures/dan_gs is an UNRELATED capture
metadata:
  type: project
---

The VBTI project has its own video→COLMAP→3DGS→MILo→mesh→Isaac reconstruction stack. This is the GS→mesh bridge for the humanoid SLAM PoC.

**Location:** `robotics/vbti/logic/reconstruct/` — note this is the sibling `vbti/` project dir next to `humanoid/`, NOT the Obsidian vault and NOT the humanoid repo. Modules: `video_utils`, `colmap_utils`, `gs_milo_utils` (wraps MILo `train.py` + `mesh_extract_sdf.py`: undistorted COLMAP → mesh PLY; subcommands reconstruct/train_gs/extract_mesh/create_config), `clean_mesh`, `cosmos_transfer`, `isaac_cfg_utils`, `master.py`.
**MILo install:** `robotics/vbti/libs/MILo/milo/`, env `gsplat-pt25`. Build env quirks (CUDA 12.9, gcc-14, TORCH_CUDA_ARCH_LIST=8.9 for the Ada 4070 Ti SUPER, don't set CPLUS_INCLUDE_PATH) are in global auto-memory's "MILo Installation" block — don't duplicate.

**CORRECTION (2026-07-08):** `~/Pictures/dan_gs` (25 GB drone capture) is a SEPARATE, unrelated personal experiment — NOT this project. Do not cite it for humanoid/VBTI work. (Supersedes an earlier wrong note that framed dan_gs as the project pipeline.)

**How to apply:** For the SLAM PoC (Albert feasibility deal, demo ~17-18 Jul 2026), the MILo mesh output is the **collision + planning** layer; the GS is the **appearance + camera-localization** layer. exp09 independently recommended MILo as the GS→mesh pick, so the recommended tool is already in-repo and working. See [[nurec-gs-isaac-render]] for the Isaac-render risk. Research workspace: `vbti/humanoid/research/slam sound/`.
