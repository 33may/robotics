# humanoid agent memory index

## Vendor / patches
- [Quat-comma patch in HU_D04_01.xml](vendor_patch_quat_commas.md) — humanoid-description ships invalid comma-separated quats; patched in place 2026-06-19
- [limxsdk undeclared runtime deps](limxsdk_undeclared_deps.md) — wheel METADATA omits click (and possibly more); add to requirements/limx.txt as they surface
- [LimX deploy orphan processes](limx_deploy_orphan_processes.md) — main.py SIGSEGV leaves ability cli subprocess holding HTTP port; launcher reaps via /proc scan
- [Vendor patch: stand autostart](vendor_patch_stand_autostart.md) — controllers.yaml `stand.autostart: true` so Oli doesn't free-fall during the bring-up window
- [Vendor patch: simulator freeze until cmd](vendor_patch_sim_freeze_until_cmd.md) — simulator.py skips mj_step + ctrl write until first RobotCmd; rest-pose hold instead of bring-up collapse
