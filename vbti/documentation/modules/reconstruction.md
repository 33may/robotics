# Reconstruction And Simulation Module

## Scope

Reconstruction/simulation code lives in:

- `logic/reconstruct/`
- `scripts/3d/`
- `scripts/sim/`

This module turns real captures into simulation-ready assets and prepared Isaac/LeIsaac tasks.

## High-Level Flow

```text
video/photos
-> frame extraction
-> COLMAP/Nerfstudio SfM
-> MILo Gaussian Splat training
-> MILo learnable SDF mesh extraction
-> optional mesh cleaning
-> USD conversion with physics
-> Isaac Sim scene composition
-> LeIsaac/IsaacLab task generation
-> HDF5 sim data collection
-> LeRobot conversion
```

This flow is a prepared extension path. The validated project loop is real-world training/evaluation; simulation is not yet proven to improve the final real policy.

## Master CLI

Main file: `logic/reconstruct/master.py`.

Video processing:

```bash
python -m vbti.logic.reconstruct.master video_processing \
  --video_path data/scene.mp4 \
  --output_dir data/frames \
  --mode count \
  --value 200
```

COLMAP + MILo:

```bash
python -m vbti.logic.reconstruct.master gs_reconstruction \
  --frames_dir data/frames \
  --output_dir data/recon \
  --matching_method exhaustive
```

Mesh to USD:

```bash
python -m vbti.logic.reconstruct.master ply_to_usda \
  --mesh_path data/recon/milo/mesh_learnable_sdf.ply \
  --output_path data/scene.usda
```

Scene composition/export:

```bash
python -m vbti.logic.reconstruct.master scene_composition \
  --scene_usda_path data/scene_composed.usda \
  --task_name vbti_so_v1
```

Standalone IsaacLab export:

```bash
python -m vbti.logic.reconstruct.master export_isaaclab \
  --scene_config_path data/scene_config.json \
  --task_name vbti_so_v1
```

## Video Utilities

File: `logic/reconstruct/video_utils.py`.

Commands:

```bash
python -m vbti.logic.reconstruct.video_utils stats --video_path scene.mp4
python -m vbti.logic.reconstruct.video_utils count --video_path scene.mp4 --percentage 0.7
python -m vbti.logic.reconstruct.video_utils fix_rotation --video_path scene.MOV
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode count --value 200
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode every --value 2
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode percentage --value 0.7
```

Defaults:

- `mode=count`
- `value=200`
- output under `output_dir/<video_stem>/*.png`
- rotation fix only when requested by the current code path.

Dependencies: `cv2`, `ffmpeg`, `sharp_frame_extractor`, `loguru`.

## COLMAP / Nerfstudio

File: `logic/reconstruct/colmap_utils.py`.

Commands:

```bash
python -m vbti.logic.reconstruct.colmap_utils run_colmap --frames_dir data/frames --output_dir data/colmap --matching_method exhaustive
python -m vbti.logic.reconstruct.colmap_utils validate_models --colmap_dir data/colmap
python -m vbti.logic.reconstruct.colmap_utils undistort --colmap_dir data/colmap --output_dir data/undistorted
python -m vbti.logic.reconstruct.colmap_utils reconstruct --frames_dir data/frames --output_dir data/recon --matching_method exhaustive
```

Output:

```text
output_dir/colmap/
output_dir/undistorted/images/
output_dir/undistorted/sparse/0/
```

The wrapper chooses the best sparse model by `images.bin` size and moves it to `sparse/0` before undistortion.

## MILo GS Training And Mesh Extraction

File: `logic/reconstruct/gs_milo_utils.py`.

Commands:

```bash
python -m vbti.logic.reconstruct.gs_milo_utils create_config --output_dir data/recon/gs
python -m vbti.logic.reconstruct.gs_milo_utils train_gs --source_dir data/recon/undistorted --model_dir data/recon/milo
python -m vbti.logic.reconstruct.gs_milo_utils extract_mesh --source_dir data/recon/undistorted --model_dir data/recon/milo
python -m vbti.logic.reconstruct.gs_milo_utils reconstruct --source_dir data/recon/undistorted --model_dir data/recon/milo
```

Defaults:

```yaml
mesh_config: default
imp_metric: indoor
rasterizer: radegs
iterations: 18000
data_device: cpu
resolution: 4
refine_iter: 1000
remove_oof: true
```

Outputs:

```text
model_dir/point_cloud/iteration_*/point_cloud.ply
model_dir/mesh_learnable_sdf.ply
```

Environment assumptions:

- `CUDA_HOME=/usr/local/cuda-12.9`
- `CC=/usr/bin/gcc-14`
- `CXX=/usr/bin/g++-14`
- `TORCH_CUDA_ARCH_LIST=8.9`
- Fedora/CUDA patches described in older docs/memory.

## Mesh Cleaning

File: `logic/reconstruct/clean_mesh.py`.

```bash
python -m vbti.logic.reconstruct.clean_mesh --mesh /path/to/mesh_learnable_sdf.ply
```

Interactive Polyscope operations:

- connected component removal;
- statistical outlier removal;
- oriented bounding box crop;
- preview and save.

Output: `mesh_cleaned.ply` beside the input.

## Format Conversion

File: `logic/reconstruct/format_utils.py`.

Commands:

```bash
python -m vbti.logic.reconstruct.format_utils splat_to_pointcloud --input_path splat.ply --output_path pointcloud.ply
python -m vbti.logic.reconstruct.format_utils glb_to_usd --input_path object.glb --output_path object.usda
python -m vbti.logic.reconstruct.format_utils mesh_to_usd --mesh_path mesh_learnable_sdf.ply --output_path scene.usda
```

Important defaults:

- `static_friction=0.7`
- `dynamic_friction=0.5`
- `restitution=0.1`
- `apply_colmap_transform=True`
- `apply_srgb_conversion=True`

Mesh-to-USD does:

- load vertex-colored mesh;
- convert sRGB to linear;
- apply COLMAP-to-USD transform;
- PCA align and center on ground;
- write USD mesh, vertex color material, physics material, and collision.

Pitfall: without sRGB-to-linear conversion, reconstructed meshes can render washed out/white in Isaac.

## Isaac / LeIsaac Generation

File: `logic/reconstruct/isaac_cfg_utils.py`.

Commands:

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

Known pitfalls:

- `LEISAAC_ROOT` is hardcoded.
- `create_task_boilerplate()` appears to reference undefined `pkg_dir` in current code.
- `master.py` and `isaac_cfg_utils.pipeline()` have different default robot prim paths.
- Domain randomization generation is placeholder/TODO in parts.
- Standalone IsaacLab export hardcodes robot joint init to zero.

## Robot USD Utilities

File: `logic/reconstruct/robot_utils.py`.

```bash
python -m vbti.logic.reconstruct.robot_utils inspect --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils fix_base --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils fix_base --robot_path robot.usd --unfix True
python -m vbti.logic.reconstruct.robot_utils set_drives --robot_path robot.usd --stiffness 17.8 --damping 0.60 --max_force 10.0 --max_velocity 10.0
python -m vbti.logic.reconstruct.robot_utils extract --robot_path robot.usd
python -m vbti.logic.reconstruct.robot_utils make_ready --robot_path robot.usd --save_path robot_ready.usd
```

Defaults match LeIsaac-style actuator values: `stiffness=17.8`, `damping=0.60`, `max_force=10.0`, `max_velocity=10.0`.

## Cosmos Transfer Prep

File: `logic/reconstruct/cosmos_transfer.py`.

```bash
python -m vbti.logic.reconstruct.cosmos_transfer extract --episode 0 --dataset_file ./datasets/vbti_table_v2_cosmos/raw.hdf5
python -m vbti.logic.reconstruct.cosmos_transfer process --episode 0 --fps 30
python -m vbti.logic.reconstruct.cosmos_transfer config --episode 0 --camera side_cam --prompt "photorealistic scene, natural lighting" --variant default
python -m vbti.logic.reconstruct.cosmos_transfer transfer --config_file ./datasets/vbti_table_v2_cosmos/cosmos/configs/episode_000_side_cam_default.json
python -m vbti.logic.reconstruct.cosmos_transfer prepare --dataset_file ./datasets/vbti_table_v2_cosmos/raw.hdf5 --output_dir ./datasets/vbti_table_v2_cosmos/cosmos/cosmos_ready
```

Status:

- `transfer()` uses Cosmos Transfer 1 hosted API, not Transfer 2.5.
- `reassemble()` is not implemented.
- Actual Transfer 2.5 work uses the external repo/deployment, not this script.

## `scripts/3d`

Poisson reconstruction from object splat:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/scripts/3d/reconstruct_mesh.py \
  --input vbti/data/so_v1/assets/duck/object_0.ply \
  --output vbti/data/so_v1/assets/duck/object_0_reconstructed.glb \
  --depth 9 \
  --estimate-normals
```

Repair manifold mesh:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/scripts/3d/fix_manifold.py \
  --input vbti/data/so_v1/assets/duck/object_0.glb \
  --output vbti/data/so_v1/assets/duck/object_0_fixed.glb
```

Isaac Script Editor utilities:

- `create_deformable_cube.py`
- `convert_duck_deformable.py`
- `convert_duck_surface_deformable.py`
- `dump_soft_cube_schema.py`

Use surface deformable for the hollow duck; volumetric deformable can create ghost collision because it fills the interior.

`tune_deformable.py` currently shows parameters by default; edit/import to tune.

`manifold_check.py` is hardcoded and not a general CLI.

## `scripts/sim/teleop_playground.py`

Run with IsaacLab:

```bash
isaaclab -p vbti/scripts/sim/teleop_playground.py
isaaclab -p vbti/scripts/sim/teleop_playground.py --port /dev/ttyACM1
isaaclab -p vbti/scripts/sim/teleop_playground.py --recalibrate
```

Purpose: minimal IsaacLab playground for physical leader-arm teleop into a simulation scene. It has hardcoded asset paths and writes calibration cache under `scripts/sim/.cache/`.

## Pitfalls

- NuRec/GS USDZ rendered in viewport but failed/hung in robot-camera synthetic data paths; mesh-compatible assets are preferred.
- Reconstruction quality metrics do not automatically imply usefulness for robot policy training.
- Sim HDF5 actions/states must be converted to the same LeRobot interface as real data before training.
- Do not claim simulation improved real-robot performance unless evaluated through the same protocol loop.
