# USD And Isaac Integration

## Mesh Cleaning

```bash
python -m vbti.logic.reconstruct.clean_mesh --mesh /path/to/mesh_learnable_sdf.ply
```

Operations:

- connected component filter;
- statistical outlier removal;
- OBB crop;
- preview;
- save `mesh_cleaned.ply`.

## Format Conversion

```bash
python -m vbti.logic.reconstruct.format_utils splat_to_pointcloud --input_path splat.ply --output_path pointcloud.ply
python -m vbti.logic.reconstruct.format_utils glb_to_usd --input_path object.glb --output_path object.usda
python -m vbti.logic.reconstruct.format_utils mesh_to_usd --mesh_path mesh_learnable_sdf.ply --output_path scene.usda
```

`mesh_to_usd` does:

- load vertex-colored mesh;
- sRGB to linear conversion;
- COLMAP to USD transform;
- PCA alignment;
- write USD mesh/material/physics/collision.

Defaults:

```text
static_friction = 0.7
dynamic_friction = 0.5
restitution = 0.1
apply_colmap_transform = true
apply_srgb_conversion = true
```

## Robot USD Utilities

```bash
python -m vbti.logic.reconstruct.robot_utils inspect --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils fix_base --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils fix_base --robot_path robot.usd --unfix True
python -m vbti.logic.reconstruct.robot_utils set_drives --robot_path robot.usd --stiffness 17.8 --damping 0.60 --max_force 10.0 --max_velocity 10.0
python -m vbti.logic.reconstruct.robot_utils extract --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils make_ready --robot_path robot.usd --save_path robot_ready.usd
```

## Isaac / LeIsaac Generation

```bash
python -m vbti.logic.reconstruct.isaac_cfg_utils no_robot_scene --scene_path scene.usda
python -m vbti.logic.reconstruct.isaac_cfg_utils extract --scene_path scene.usda --robot_prim_path /World/so101_simready_follower_leisaac
python -m vbti.logic.reconstruct.isaac_cfg_utils gen_scene --task_name vbti_so_v1 --scene_usd_path scene_no_robot.usda
python -m vbti.logic.reconstruct.isaac_cfg_utils gen_task_folders --task_name vbti_so_v1
python -m vbti.logic.reconstruct.isaac_cfg_utils gen_leisaac --scene_usda_path scene.usda --task_name vbti_so_v1
python -m vbti.logic.reconstruct.isaac_cfg_utils gen_isaaclab --scene_config data/scene_config.json --task_name vbti_so_v1 --output_dir data/vbti_so_v1_isaaclab
python -m vbti.logic.reconstruct.isaac_cfg_utils pipeline --scene_usda_path scene.usda --task_name vbti_so_v1
```

Generated artifacts:

```text
<scene>_no_robot.usda
scene_config.json
leisaac/source/leisaac/leisaac/assets/scenes/<task_name>.py
leisaac/source/leisaac/leisaac/tasks/<task_name>/
<task_name>_env_cfg.py
```

## Known Code Issues

- `LEISAAC_ROOT` is hardcoded.
- `create_task_boilerplate()` appears to reference undefined `pkg_dir`.
- `master.py` and `isaac_cfg_utils.pipeline()` have different default robot prim paths.
- Domain randomization generation is incomplete/placeholder in parts.
- Standalone IsaacLab export hardcodes robot joint init to zero.
