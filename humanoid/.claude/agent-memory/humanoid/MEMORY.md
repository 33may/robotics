# humanoid agent memory index

## Feedback
- [Show, don't tell in pitches](feedback_show_dont_tell.md) — don't claim capability directly; demonstrate with evidence/artifacts/diagrams
- [Architecture diagram style](feedback_architecture_diagram_style.md) — C2/C4 arm-project look (hand-composed HTML+SVG, code-grounded); never marketing cards or graphviz auto-layout
- [Daily-note timestamps](feedback_daily_note_timestamps.md) — every daily-note block stamped with current HH:MM for time tracking (since 08-07-2026)
- [Build on branches](feedback_build_on_branches.md) — dev work on `33may/` branches, not `main` (Anton-waivable per-task)
- [Manager plans, not builds](feedback_manager_plans_not_builds.md) — in /manager sessions stay at planning altitude; reshaping a task is fine, diving into code/verification is not
- [dev_app panels are pure I/O](feedback_devapp_pure_io.md) — no planning/logic in panels; brain modules own logic behind in/out contracts (GoalCoordinate in → path out)
- [Close the validation loop](feedback_close_validation_loop.md) — whatever we build, make "working" a computable pass/fail vs ground truth; build the scorer early
- [Explain plainly](feedback_explain_plainly.md) — lead with a plain mental model anchored to what Anton knows (COLMAP); strip jargon from the first pass; "again"/"I understand nothing" = restate simpler
- [Sandbox env policy](feedback_sandbox_env_policy.md) — no Docker; disposable per-candidate conda envs (cleanly removable + immediately works in our env)
- [Plan together, stepwise](feedback_plan_together_stepwise.md) — never bulk-author OpenSpec/planning artifacts solo; one section/decision at a time, agree in chat, then write

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
- [Isaac occupancy-map baking](isaac_occupancy_map_baking.md) — omap is a static geometry bake; origin seed must be off-floor or map is blank; warehouse nav map v1 = 0.05 / origin [-26.925,-23.425]

## SLAM / navigation PoC (Albert feasibility deal, demo ~17-18 Jul 2026)
- [Nav brain localizer seam](nav_brain_localizer_seam.md) — reason/nav/ is invariant + complete (A*→pursuit); GroundTruthLocalizer is the swap-point for real SLAM; world half is the open front
- [Localization research outcome](research_nav_localization.md) — two-layer (in-brain leg-FK odometry + OOP RTAB-Map reloc now / cuVSLAM later); full report docs/research/localization.md
- [Nav pose-tolerance experiment](experiment_nav_pose_tolerance.md) — measured budget: constant bias <0.1m is binding (not noise); odometry makes reloc-rate cheap (0.2Hz OK w/ dead-reckon)
- [Nav costmap layering](architecture_nav_costmap_layering.md) — map=baked boolean (world truth) vs footprint/clearance=runtime robot layer; clearance NOT baked into convert_ros_map
- [locbench harness design](architecture_locbench_design.md) — SLAM eval harness locked 13-07: socket-client recorder, frozen 2-pass bags, 2-fn adapter contract, fixed-anchor SE(2) gates, agent CLI
- [Reason module separation](architecture_reason_module_separation.md) — 13-07 locked: localization/mapping/nav split, LocalizationIn/Out + Map contracts, planner consumes emitted Map (B), 2 OpenSpec changes
- [Reconstruct pipeline (MILo)](reconstruct_pipeline_milo.md) — project COLMAP→3DGS→MILo→mesh stack at vbti/logic/reconstruct (MILo in vbti/libs/MILo); ~/Pictures/dan_gs is NOT this project
- [NuRec/GS render in Isaac](nurec_gs_isaac_render.md) — TiledCamera wouldn't render GS (real, ~Feb'26 on the 5090 box); maybe IsaacLab#4951 Blackwell bug; re-test on the box w/ Isaac 6.0
- Research base: `vbti/humanoid/research/slam sound/` (paper.md + synthesis_v2 + exp01–11)
