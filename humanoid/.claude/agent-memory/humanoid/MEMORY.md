# humanoid agent memory index

## Feedback
- [Show, don't tell in pitches](feedback_show_dont_tell.md) — don't claim capability directly; demonstrate with evidence/artifacts/diagrams
- [Architecture diagram style](feedback_architecture_diagram_style.md) — C2/C4 arm-project look (hand-composed HTML+SVG, code-grounded); never marketing cards or graphviz auto-layout
- [Daily-note timestamps](feedback_daily_note_timestamps.md) — every daily-note block stamped with current HH:MM for time tracking (since 08-07-2026)
- [Build on branches](feedback_build_on_branches.md) — dev work on `33may/` branches, not `main` (Anton-waivable per-task)

## Architecture
- [Brain backbone: DIY vs ROS2](architecture_brain_backbone_ros_vs_diy.md) — open decision on the brain's internal rails; stance = long-horizon build-it-right; leaning DIY-backbone + ROS-as-island

## Tools / references
- [VBTI visuals figure-factory](reference_vbti_visuals.md) — HTML+CSS+Playwright decks at vbti/oficial/docs/visuals; restore from ~/Downloads tarball if engine vanishes (vault not git)

## Vendor / patches
- [Quat-comma patch in HU_D04_01.xml](vendor_patch_quat_commas.md) — humanoid-description ships invalid comma-separated quats; patched in place 2026-06-19
- [limxsdk undeclared runtime deps](limxsdk_undeclared_deps.md) — wheel METADATA omits click (and possibly more); add to requirements/limx.txt as they surface
- [LimX deploy orphan processes](limx_deploy_orphan_processes.md) — main.py SIGSEGV leaves ability cli subprocess holding HTTP port; launcher reaps via /proc scan
- [Vendor patch: stand autostart](vendor_patch_stand_autostart.md) — controllers.yaml `stand.autostart: true` so Oli doesn't free-fall during the bring-up window
- [Vendor patch: simulator freeze until cmd](vendor_patch_sim_freeze_until_cmd.md) — simulator.py skips mj_step + ctrl write until first RobotCmd; rest-pose hold instead of bring-up collapse

## Isaac / assets
- [Isaac asset download → empty USD](isaac_asset_download_empty_usd.md) — single-file .usd download/Save-As = empty crate (0 prims); use Collect As / cloud stream / offline pack (MAY-171)

## SLAM / navigation PoC (Albert feasibility deal, demo ~17-18 Jul 2026)
- [Reconstruct pipeline (MILo)](reconstruct_pipeline_milo.md) — project COLMAP→3DGS→MILo→mesh stack at vbti/logic/reconstruct (MILo in vbti/libs/MILo); ~/Pictures/dan_gs is NOT this project
- [NuRec/GS render in Isaac](nurec_gs_isaac_render.md) — TiledCamera wouldn't render GS (real, ~Feb'26 on the 5090 box); maybe IsaacLab#4951 Blackwell bug; re-test on the box w/ Isaac 6.0
- Research base: `vbti/humanoid/research/slam sound/` (paper.md + synthesis_v2 + exp01–11)
