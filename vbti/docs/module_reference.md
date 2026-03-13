# Module Reference — Function Index

Per-file function index for all vbti modules. Use this as a checklist for the code review session.

---

## vbti/utils/master.py

Pipeline orchestrator. CLI via `fire.Fire()`.

| Function | Signature | Delegates To |
|----------|-----------|-------------|
| `video_processing` | `(video_path, output_dir, mode="count", value=200)` | `video_utils.extract_frames()` |
| `gs_reconstruction` | `(frames_dir, output_dir, config_path=None, matching_method="exhaustive")` | `colmap_utils` → `gs_milo_utils` |
| `ply_to_usda` | `(mesh_path, output_path, static_friction=0.7, dynamic_friction=0.5, restitution=0.1, apply_colmap_transform=True, apply_srgb_conversion=True)` | `format_utils.mesh_to_usd()` |
| `scene_composition` | `(scene_usda_path, task_name, robot_prim_path=..., robot_usd_path=None, cosmos_sensors=False)` | `isaac_cfg_utils.pipeline()` |
| `export_isaaclab_task` | `(scene_config_path, task_name, output_dir=None)` | `isaac_cfg_utils.generate_isaaclab_env()` |

---

## vbti/utils/video_utils.py

Video processing. OpenCV + ffmpeg + sharp-frame-extractor.

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `get_video_info` | `(video_path)` | `dict{path, fps, total_frames, width, height, duration_sec, filesize_mb}` | cv2.VideoCapture |
| `print_video_stats` | `(video_path)` | None (prints) | |
| `calculate_frame_count` | `(video_path, percentage: 0-1)` | `int` | |
| `fix_rotation` | `(video_path, output_path=None)` | `str` (path) | ffmpeg re-encode, bakes EXIF rotation |
| `extract_frames` | `(video_path, output_dir, mode="count", value=200)` | `int` (frame count) | Auto-calls fix_rotation first |

---

## vbti/utils/colmap_utils.py

COLMAP reconstruction. Wraps nerfstudio + colmap CLI.

| Function | Signature | Notes |
|----------|-----------|-------|
| `run_colmap` | `(frames_dir, output_dir, matching_method="exhaustive")` | `ns-process-data images` subprocess |
| `validate_models` | `(colmap_dir)` | Scores models by images.bin size, swaps best to 0 |
| `undistort` | `(colmap_dir, output_dir)` | `colmap image_undistorter`, OPENCV→PINHOLE |
| `process_colmap` | `(frames_dir, output_dir, matching_method="exhaustive")` | Master: run→validate→undistort |

---

## vbti/utils/gs_milo_utils.py

MILo GS training + mesh extraction.

| Function | Signature | Notes |
|----------|-----------|-------|
| `save_config` | `(config, output_path)` | YAML |
| `load_config` | `(config_path)` | → dict |
| `create_config` | `(output_dir)` | Interactive CLI |
| `train_gs` | `(source_dir, model_dir, mesh_config="default", imp_metric="indoor", rasterizer="radegs", iterations=18000, data_device="cpu")` | Subprocess in MILO_DIR with MILO_ENV |
| `extract_mesh` | `(source_dir, model_dir, mesh_config="default", imp_metric="indoor", rasterizer="radegs", refine_iter=1000, remove_oof=True)` | Learnable SDF extraction |
| `reconstruct_mesh` | `(source_dir, model_dir, config_path=None, ...)` | Master: train_gs → extract_mesh |

**Constants:** `MILO_DIR`, `MILO_ENV` (gcc-14, cuda-12.9, TORCH_CUDA_ARCH_LIST=8.9), `CONFIG_FIELDS`

---

## vbti/utils/format_utils.py

3D format conversions. open3d + trimesh + pxr.

| Function | Signature | Notes |
|----------|-----------|-------|
| `_load_and_filter_splat` | `(input_path, opacity_threshold=0.5, scale_percentile=95.0, outlier_nb=20, outlier_std=2.0)` | → o3d.PointCloud |
| `splat_to_pointcloud` | `(input_path, output_path, ...)` | Filtered PLY export |
| `glb_to_usd` | `(input_path, output_path, collision="convexDecomposition", rigid_body=True)` | Materials + UVs preserved |
| `mesh_to_usd` | `(mesh_path, output_path, static_friction=0.7, dynamic_friction=0.5, restitution=0.1, apply_colmap_transform=True, apply_srgb_conversion=True)` | Full pipeline: load→sRGB→transform→PCA→USD |
| `_srgb_to_linear` | `(colors)` | IEC 61966-2-1 gamma curve |
| `_align_and_center` | `(vertices)` | PCA up-direction + center at origin |
| `_transform_vertices` | `(vertices, matrix)` | Homogeneous 4×4 |
| `_rotation_between_vectors` | `(a, b)` | 3×3 rotation matrix |

**Constants:** `SH_C0 = 0.28209...`, `COLMAP_TO_USD` (4×4 coordinate transform)

---

## vbti/utils/isaac_cfg_utils.py (~1300 lines)

LeIsaac/IsaacLab code generation from USDA scenes.

| Function | Signature | Notes |
|----------|-----------|-------|
| `create_no_robot_scene` | `(scene_path, robot_prim_path=..., save_path=None)` | Strips robot, lights, /Render |
| `extract_scene_config` | `(scene_path, robot_prim_path=None, robot_usd_path=None, save_path=None)` | → dict (robot, cameras, lights, subassets) |
| `generate_scene_asset` | `(task_name, scene_usd_path, leisaac_root=LEISAAC_ROOT)` | Writes assets/scenes/{task}.py |
| `create_task_boilerplate` | `(task_name, tasks_root=..., gym_id=None, env_cfg_class=None)` | gym.register + MDP stubs |
| `generate_leisaac_env` | `(scene_usda_path, task_name, ..., cosmos_sensors=False, ...)` | Full env_cfg.py generation |
| `generate_isaaclab_env` | `(scene_config, task_name, output_dir=None)` | Standalone IsaacLab export (no leisaac) |
| `pipeline` | `(scene_usda_path, task_name, robot_prim_path=..., robot_usd_path=None, cosmos_sensors=False)` | Full orchestration of above 5 steps |

**Helpers:** `_cam_field()`, `_light_field()`, `_discover_subassets()`, `_obs_cam_field()`, `_robot_articulation_field()`, `_subasset_field()`

**Constant:** `LEISAAC_ROOT = "/home/may33/projects/ml_portfolio/robotics/leisaac"`

---

## vbti/utils/robot_utils.py

USD robot manipulation.

| Function | Signature | Notes |
|----------|-----------|-------|
| `fix_articulation_base` | `(robot_path, save_path=None, unfix=False)` | Sets kinematicEnabled |
| `set_drives` | `(robot_path, save_path=None, stiffness=17.8, damping=0.60, max_force=10.0, max_velocity=10.0)` | Iterates PhysicsRevoluteJoint prims |
| `extract_robot_config` | `(robot_path)` | → dict (joints with axis, limits, drive params) |
| `make_ready` | `(robot_path, save_path=None, ...)` | fix_base + set_drives |
| `inspect_robot` | `(robot_path, joint_info=True)` | Colored prim tree + drive warnings |

---

## vbti/utils/clean_mesh.py

Interactive Polyscope GUI mesh cleaner.

**Class:** `MeshCleaner`

| Method | Purpose |
|--------|---------|
| `__init__` | Load PLY, compute normals, sRGB→linear for display |
| Connected component filter | Remove small floating blobs (min_ratio=0.01) |
| SOR filter | Statistical outlier removal (nb=20, std=2.0) |
| OBB crop | Interactive bounding box (rotation + center + half-extents) |
| Preview | Apply all filters without saving |
| Save | Output `mesh_cleaned.ply` |

---

## vbti/utils/cosmos_transfer.py

Cosmos Transfer data augmentation.

| Function | Signature | Notes |
|----------|-----------|-------|
| `extract` | `(episode, dataset_file=..., depth_min=0.01, depth_max=2.0)` | HDF5 → PNG per camera per modality |
| `process` | `(episode, fps=30)` | PNG → MP4 + Canny edges |
| `config` | `(episode, camera="side_cam", prompt="...", variant="default", vis_weight=0.7, depth_weight=0.3, edge_weight=0.3)` | Generate spec JSON |
| `transfer` | `(config_file, api_key="", ...)` | ⚠ Calls Transfer **1** API, not 2.5 |
| `reassemble` | `(episode, variant, dataset_file)` | ⚠ **NOT IMPLEMENTED** |
| `prepare_cosmos` | `(dataset_file, output_dir, prompt, cameras, controls, ...)` | Batch all episodes |

**Constants:** `CAMERAS`, `MODALITIES`, `FRAME_SIZE=(640,480)`, `BASE_DIR`, `Canny=(50,150)`

---

## vbti/utils/dataset_utils.py

Dataset loading & splitting.

| Function | Signature | Returns |
|----------|-----------|---------|
| `load_and_split_dataset` | `(repo_id, root=None, train_ratio=0.8, random_seed=42, ...)` | `(full, train, val)` LeRobotDataset |
| `create_dataloaders` | `(train, val, batch_size=8, num_workers=4)` | `(train_loader, val_loader)` |

---

## vbti/utils/datasets/inspect_dataset.py

Dataset inspection & reports.

### LeRobot Functions
| Function | Purpose |
|----------|---------|
| `lerobot_info(path)` | meta/info.json |
| `lerobot_stats(path)` | Normalization stats |
| `lerobot_features(path)` | Feature schema |
| `lerobot_parquet_schema(path)` | Column types/shapes |
| `lerobot_action_stats(path)` | Min/max/mean/std per joint + unit detection |
| `lerobot_state_stats(path)` | observation.state stats |
| `lerobot_episode_lengths(path)` | Episode metadata |
| `lerobot_video_info(path)` | Video file counts/sizes |
| `lerobot_action_state_correspondence(path)` | Check action ≈ state(t) or delta |

### HDF5 Functions
| Function | Purpose |
|----------|---------|
| `hdf5_tree(path, episode)` | Full episode structure |
| `hdf5_episode_list(path)` | {name, frames} per episode |
| `hdf5_action_stats(path)` | Action statistics + unit detection |
| `hdf5_state_stats(path)` | State statistics |
| `hdf5_image_stats(path)` | dtype/shape/range per camera |
| `hdf5_action_state_correspondence(path)` | Action-to-state mapping |
| `report_lerobot(path)` | Full inspection report |
| `report_hdf5(path)` | Full inspection report |

---

## vbti/utils/inference/run_smolvla_inference.py

SmolVLA inference in Isaac Sim.

| Function | Purpose |
|----------|---------|
| `load_policy(checkpoint, device)` | → (policy, preprocessor, postprocessor) |
| `obs_to_policy_input(obs_dict)` | Convert Isaac obs → policy format (unit conversion + image transform) |
| Main loop | Load env → get obs → preprocess → select_action → postprocess → degrees→radians → env.step |

**CLI args:** `--checkpoint`, `--task`, `--num_episodes`, `--action_horizon`, `--step_hz`, `--max_steps`, `--plot_actions`, `--save_video`, `--enable_cameras`

---

## vbti/utils/train/train_smolvla_custom.py

SmolVLA training script.

| Component | Value |
|-----------|-------|
| Dataset | `eternalmay33/lift_cube_3cams` |
| Steps | 10000 |
| Batch size | 4 |
| LR | 1e-5 |
| Chunk size | 50 |
| Validate every | 500 steps |
| Checkpoint every | 1000 steps |

---

## vbti/scripts/3d/

| Script | Purpose |
|--------|---------|
| `reconstruct_mesh.py` | GS PLY → Poisson mesh + color transfer → GLB |
| `fix_manifold.py` | PyMeshLab repair: duplicates, non-manifold, holes |
| `create_deformable_cube.py` | Programmatic deformable surface in Isaac |
| `convert_duck_deformable.py` | In-editor: clean mesh → PhysX deformable (tet mesh, Young's 5e5) |
| `dump_soft_cube_schema.py` | Introspect SoftCube prim schema |
| `tune_deformable.py` | Regex parameter tuning for USDA deformable bodies |
| `manifold_check.py` | Poisson reconstruction + watertight check |

## vbti/scripts/cameras/

| Script | Purpose |
|--------|---------|
| `check_usb.py` | RealSense USB speed, usbfs memory, firmware |
| `reset_camera.py` | Hardware reset by serial or all cameras |
| `view_cameras.py` | Live 2×N grid of RealSense feeds |

## vbti/scripts/servos/

| Script | Purpose |
|--------|---------|
| `scan_all.py` | Scan Feetech bus, display IDs 1-6 with position/voltage/temp |
| `change_id.py` | Safely change servo ID (EEPROM unlock, collision check) |
| `factory_reset_motors.py` | Reset motor to factory, handle ID collisions gracefully |

---

## Config Files

### vbti/data/ready_export_sov1/config/vbti_so_v1_env_cfg.py

| Class | Purpose |
|-------|---------|
| `VbtiSoV1SceneCfg` | Scene USD, robot pose, 3× TiledCamera (640×480@30Hz, 18.1mm), DomeLight + DistantLight |
| `ObservationsCfg` | joint_pos, joint_vel, joint_pos_rel, joint_vel_rel, last_action, cam_top, cam_right, cam_left, wrist |
| `TerminationsCfg` | time_out (25s) |
| `VbtiSoV1EnvCfg` | Full RL env config (decimation=1, PhysX bounce threshold 0.01) |

**DR Status:** Commented out but wired (`leisaac.utils.domain_randomization` imports exist).

### vbti/data/so_v1/scene_config.json

Robot pose, camera positions/orientations/focal lengths, joint limits/drive params, light config, HDRI path.
