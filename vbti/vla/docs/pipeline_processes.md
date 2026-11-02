# Pipeline Processes — Code-to-Knowledge Map

Step-by-step process flows for every major pipeline in VBTI, with exact function calls, data formats, and gotchas.

---

## Table of Contents

1. [Video → Mesh Reconstruction](#1-video--mesh-reconstruction)
2. [Scene Composition → LeIsaac Task](#2-scene-composition--leisaac-task)
3. [Teleoperation & Data Collection](#3-teleoperation--data-collection)
4. [Cosmos Transfer Augmentation](#4-cosmos-transfer-augmentation)
5. [SmolVLA Training](#5-smolvla-training)
6. [SmolVLA Inference in Isaac Sim](#6-smolvla-inference-in-isaac-sim)
7. [Robot USD Preparation](#7-robot-usd-preparation)
8. [Interactive Asset Creation](#8-interactive-asset-creation)

---

## 1. Video → Mesh Reconstruction

**Entry point:** `python -m vbti.logic.reconstruct.master video_processing` → `gs_reconstruction` → `ply_to_usda`

### Step 1a: Frame Extraction

```
video_utils.extract_frames(video_path, output_dir, mode="count", value=200)
```

| What happens | Code | Notes |
|---|---|---|
| Fix phone rotation metadata | `video_utils.fix_rotation()` → ffmpeg `-c:v libx264 -crf 18` | Phone cameras store rotation as EXIF; OpenCV ignores it |
| Extract best frames | `sharp_frame_extractor <video> -o <dir> --count N` (subprocess) | Quality-based selection, not uniform sampling |
| Modes | "count" (N frames), "every" (seconds), "percentage" (0-1) | |

**Output:** `output_dir/*.png` (8-bit RGB frames)

### Step 1b: COLMAP Reconstruction

```
colmap_utils.process_colmap(frames_dir, output_dir, matching_method="exhaustive")
```

| What happens | Code | Notes |
|---|---|---|
| Run SfM | `colmap_utils.run_colmap()` → `ns-process-data images` (subprocess) | Nerfstudio wrapper around COLMAP |
| Validate models | `colmap_utils.validate_models()` → scores by images.bin size | **COLMAP creates multiple models** — model 0 isn't always best |
| Swap best to 0 | Backs up `sparse/0/` → `sparse/0_original/`, moves best to `sparse/0/` | |
| Undistort images | `colmap_utils.undistort()` → `colmap image_undistorter` | OPENCV → PINHOLE camera model (removes lens distortion) |

**Gotchas:**
- For <500 images, use `exhaustive` matching (no vocab tree needed)
- If vocab tree fails ("Failed to read faiss index"), replace `~/.local/share/nerfstudio/vocab_tree.fbow` with faiss version from COLMAP 3.11.1 release

**Output:** `output_dir/undistorted/{images/, sparse/0/}` (PINHOLE cameras, undistorted PNGs)

### Step 1c: MILo GS Training + Mesh Extraction

```
gs_milo_utils.reconstruct_mesh(source_dir, model_dir, config_path=None, ...)
```

| What happens | Code | Notes |
|---|---|---|
| Train Gaussian Splatting | `gs_milo_utils.train_gs()` → `python train.py` in MILo dir | Env: gcc-14, CUDA 12.9, TORCH_CUDA_ARCH_LIST=8.9 |
| Extract mesh from SDF | `gs_milo_utils.extract_mesh()` → `python mesh_extract_sdf.py` | Learnable SDF, not Poisson reconstruction |

**Config fields** (interactive via `create_config`):
- `mesh_config`: verylowres/lowres/default/highres/veryhighres
- `imp_metric`: indoor/outdoor
- `iterations`: 18000 (fast) or 30000 (quality)
- `data_device`: cpu (saves VRAM) or cuda
- `refine_iter`: 1000 (fast) or 2000-3000 (quality)
- `remove_oof`: remove out-of-view vertices

**Output:** `model_dir/mesh_learnable_sdf.ply` (vertex-colored triangle mesh)

### Step 1d: Mesh → USD

```
format_utils.mesh_to_usd(mesh_path, output_path, static_friction=0.7, ...)
```

| What happens | Code | Notes |
|---|---|---|
| Load PLY | open3d TriangleMesh | Vertex colors + normals |
| sRGB→Linear conversion | `_srgb_to_linear()` — IEC 61966-2-1 gamma | **Without this, mesh renders white in Isaac Sim** |
| COLMAP→USD coordinate transform | `COLMAP_TO_USD` 4×4 matrix: x→-x, y→-z, z→-y | Hardcoded constant |
| PCA alignment + center | `_align_and_center()` — PCA computes "up" direction, centers at origin | Bottom at Z=0 |
| Write USD | pxr: Mesh + VertexColorMaterial + PhysicsMaterial + MeshCollisionAPI | Convex decomposition for physics |

**USD structure:**
```
/World (Xform)
  /World/Environment/Mesh (UsdGeom.Mesh + collision)
  /World/Environment/VertexColorMaterial (UsdPreviewSurface)
  /World/Environment/PhysicsMaterial (static_friction, dynamic_friction, restitution)
```

---

## 2. Scene Composition → LeIsaac Task

**Entry point:** `python -m vbti.logic.reconstruct.master scene_composition` or `python -m vbti.logic.reconstruct.robot_utils pipeline`

### Prerequisites

Scene must be composed in Isaac Sim GUI first:
1. Import MILo mesh USD (from step 1d)
2. Import collision mesh (invisible physics body)
3. Place SO-101 robot (set `kinematicEnabled=True`)
4. Position cameras (side_cam, table_cam, gripper_cam)
5. Add HDRI dome light + distant light
6. Import interactive objects (duck, cup as GLB→USD)
7. Save as `scene.usda`

### Automated Pipeline

```
isaac_cfg_utils.pipeline(scene_usda_path, task_name, robot_prim_path, robot_usd_path, cosmos_sensors)
```

| Step | Function | Output |
|---|---|---|
| 1. Strip robot + lights | `create_no_robot_scene()` | `{scene}_no_robot.usda` |
| 2. Extract config | `extract_scene_config()` | `scene_config.json` (robot pose, camera positions, light params) |
| 3. Register scene asset | `generate_scene_asset()` | `leisaac/assets/scenes/{task_name}.py` |
| 4. Create task folders | `create_task_boilerplate()` | `leisaac/tasks/{task_name}/__init__.py` (gym.register) |
| 5. Generate env config | `generate_leisaac_env()` | `{task_name}_env_cfg.py` (SceneCfg, ObsCfg, TermCfg, EnvCfg) |

**Generated code structure:**
- **SceneCfg**: scene USD + cameras (spawn=None for scene cameras, spawned for gripper) + lights
- **ObservationsCfg**: `joint_pos` + images from each camera. If `cosmos_sensors=True`: adds `depth_to_camera` + `instance_segmentation_fast`
- **TerminationsCfg**: time_out (25s episode)
- **EnvCfg**: robot position, `parse_usd_and_create_subassets()` for auto-discovering physics objects

**Parallel path — standalone IsaacLab export:**
```
isaac_cfg_utils.generate_isaaclab_env(scene_config, task_name, output_dir)
```
Produces self-contained task folder (copies all assets, no leisaac dependency).

**Gotchas:**
- `LEISAAC_ROOT` is hardcoded to `/home/may33/projects/ml_portfolio/robotics/leisaac`
- Default camera resolution: 640×480
- Default robot prim path: `/World/so101_simready_follower_leisaac`
- Robot joint init always zero (hardcoded)
- DR is generated as placeholder (TODO comments in code)

---

## 3. Teleoperation & Data Collection

**Entry point:** `python teleop_se3_agent.py --task <task> --teleop_device so101leader --enable_cameras`

### Physical Setup

| Component | Config |
|---|---|
| Leader arm | `/dev/ttyACM2`, 1MHz baud, reads joint positions |
| Follower arm | `/dev/ttyACM1`, 1MHz baud, receives position commands |
| Cameras | 4× RealSense D405 @ 640×480 @15fps (USB 2.1 limit) |

### Data Flow

```
Leader arm joints (GroupSyncRead @ 60Hz)
    ↓
LeIsaac teleop_se3_agent.py
    ↓
Isaac Sim env.step(actions)
    ↓
HDF5 writer: obs (RGB + depth + seg + joint_pos) + actions
```

### Controls

- **B** — Start recording episode
- **N** — Mark success, save, reset env
- **R** — Discard episode, reset env

### HDF5 Output Schema

```
data/demo_NNN/
  obs/
    side_cam_rgb       (T, 480, 640, 3)  uint8
    side_cam_depth     (T, 480, 640)     float32 (meters)
    side_cam_seg       (T, 480, 640, 4)  uint8 RGBA
    table_cam_rgb      (T, 480, 640, 3)  uint8
    table_cam_depth    (T, 480, 640)     float32
    table_cam_seg      (T, 480, 640, 4)  uint8
    wrist_rgb          (T, 480, 640, 3)  uint8
    wrist_depth        (T, 480, 640)     float32
    wrist_seg          (T, 480, 640, 4)  uint8
    joint_pos          (T, 6)            float32 (radians)
    joint_vel          (T, 6)            float32
  actions              (T, 6)            float32 (radians)
```

### Datasets Collected

| Dataset | Episodes | Size | Notes |
|---------|----------|------|-------|
| `vbti_so_v1_mix_v1.hdf5` | 217 | 154 GB | Full sim dataset with DR |
| `vbti_table_v2_cosmos/raw.hdf5` | 3 | ~85 GB | Cosmos Transfer prep |
| `may33/so101_pick_place` (HuggingFace) | 51 | ~300 MB | Real-world BC baseline |

### Direct Teleop (No Sim)

`raw_teleop.py` — Direct servo-to-servo at 60Hz, 1MHz baud. Reads leader positions via `GroupSyncRead`, writes to follower via `GroupSyncWrite`. No simulation involved. Torque enabled only on follower.

---

## 4. Cosmos Transfer Augmentation

**Entry point:** `python -m vbti.logic.reconstruct.cosmos_transfer <command>`

### ⚠ STATUS NOTE

- `transfer()` calls **Cosmos Transfer 1 API** (not 2.5) — endpoint: `nvidia/cosmos-transfer1-7b`
- `reassemble()` is **NOT IMPLEMENTED** (TODO in code)
- The Cosmos Transfer 2.5 deployment on RunPod is done via the cosmos-transfer2.5 repo directly, not this script

### Step-by-Step

| Command | Function | What it does | Output |
|---------|----------|--------------|--------|
| `extract` | `extract(episode, ...)` | Read HDF5 frames → PNG per camera per modality | `cosmos/captures/episode_NNN/<cam>/{rgb,depth,seg}/*.png` + `depth_raw/*.npy` |
| `process` | `process(episode, fps=30)` | PNG → MP4 + Canny edge detection | `cosmos/processed/episode_NNN/<cam>/{rgb,depth,edge,seg}.mp4` |
| `config` | `config(episode, ...)` | Generate Cosmos spec JSON | `cosmos/configs/episode_NNN_<cam>_<variant>.json` |
| `transfer` | `transfer(config_file, ...)` | HTTP POST to Cosmos API (base64 video) | Output MP4 (decoded) |
| `reassemble` | `reassemble(...)` | **NOT IMPLEMENTED** | — |
| `prepare` | `prepare_cosmos(...)` | Batch all episodes | All videos + configs |

### Depth Processing

```python
# HDF5 depth is float32 meters
depth_normalized = np.clip(depth, depth_min=0.01, depth_max=2.0)
depth_uint8 = ((depth_normalized - 0.01) / (2.0 - 0.01) * 255).astype(uint8)
```

### Constants

- `CAMERAS = ["side_cam", "table_cam", "wrist"]`
- `FRAME_SIZE = (640, 480)` (W × H)
- `Canny thresholds = (50, 150)`
- `BASE_DIR = ./datasets/vbti_table_v2_cosmos`

---

## 5. SmolVLA Training

**Entry point:** `python -m vbti.logic.train.train_smolvla_custom`

### Config

| Parameter | Value |
|-----------|-------|
| Dataset | `eternalmay33/lift_cube_3cams` |
| Output | `outputs/train/smolvla_lift_cube_3cams_v2` |
| Steps | 10000 |
| Batch size | 4 |
| Learning rate | 1e-5 |
| Grad clip | 10.0 |
| Warmup | 500 steps |
| Chunk size | 50 (predict 50 future actions) |
| n_obs_steps | 1 (current observation only) |
| freeze_vision_encoder | True |
| train_expert_only | True |

### Training Loop

1. Load dataset via `load_and_split_dataset()` → 80/20 train/val by episode (not frame)
2. Create DataLoaders via `create_dataloaders()` → batch_size=4, shuffle train
3. Each batch: `preprocessor(batch)` → `policy.forward(batch)` → loss → backward → grad clip → step
4. Validate every 500 steps (50 batches sampled)
5. Save best model (lowest val loss) + checkpoints every 1000 steps

### Output

```
checkpoint_dir/
  config.json               # SmolVLAConfig
  model.safetensors         # Policy weights
  optimizer.pt              # Optimizer state
  policy_preprocessor.json  # Input normalization (mean/std)
  policy_postprocessor.json # Output denormalization (mean/std)
```

**Critical:** All three JSON files required for inference. Missing postprocessor was the root cause of the first inference failure.

---

## 6. SmolVLA Inference in Isaac Sim

**Entry point:** `python -m vbti.logic.inference.run_smolvla_inference --checkpoint=... --task=... --enable_cameras`

### Unit Conversion Chain (CRITICAL)

```
Isaac Sim observation
  ├─ joint_pos: RADIANS → × (180/π) → DEGREES
  ├─ images: uint8 NHWC → permute → NCHW float32 [0,1]
  └─ task prompt: string
       ↓
  preprocessor(obs)           # normalize with degree-based stats
       ↓
  policy.select_action()      # outputs NORMALIZED actions
       ↓
  postprocessor(actions)      # denormalize → DEGREES
       ↓
  × (π/180) → RADIANS
       ↓
  env.step(radians)
```

### Image Conversion

```python
# Isaac returns: (N, H, W, C) uint8 [0-255]
# Policy expects: (N, C, H, W) float32 [0-1]
img = img.permute(0, 3, 1, 2).float() / 255.0
```

### Camera Mapping

| Isaac Obs Key | Policy Input Key |
|---|---|
| `front` | `observation.images.front` |
| `front_cam_cfg` | `observation.images.third_person` |
| `gripper_cam_cfg` | `observation.images.gripper` |

### Normalization Stats (degrees)

```
Action mean: [2.55, 13.68, -19.39, 76.21, 4.79, 23.37]
Action std:  [8.72, 24.19, 33.97, 18.91, 11.25, 16.16]
State mean:  [2.55, 15.93, -17.21, 75.45, 4.75, 28.40]
State std:   [8.69, 24.58, 33.84, 19.99, 11.22, 10.80]
```

---

## 7. Robot USD Preparation

**Entry point:** `python -m vbti.logic.reconstruct.robot_utils <command>`

### Commands

| Command | Function | Purpose |
|---------|----------|---------|
| `inspect` | `inspect_robot(path)` | Print colored prim hierarchy, joint drives, ArticulationRoot |
| `make_ready` | `make_ready(path)` | fix_articulation_base + set_drives in one call |
| `fix_base` | `fix_articulation_base(path)` | Set `kinematicEnabled=True` on ArticulationRoot |
| `set_drives` | `set_drives(path, stiffness, damping, ...)` | Configure joint drive parameters |
| `extract_config` | `extract_robot_config(path)` | Export joint config as JSON |

### Drive Defaults

```
stiffness=17.8, damping=0.60, max_force=10.0, max_velocity=10.0
```

These match LeIsaac `ImplicitActuatorCfg` defaults.

---

## 8. Interactive Asset Creation

### Object Pipeline (GLB from SAM3D)

```
format_utils.glb_to_usd(input_path, output_path, collision="convexDecomposition", rigid_body=True)
```

Loads GLB → preserves materials/UVs → creates USD with collision.

### Deformable Object Pipeline (Duck)

**Scripts:** `vbti/scripts/3d/convert_duck_deformable.py`

```
object_0.ply → reconstruct_mesh.py → object_0.glb
  → fix_manifold.py → object_0_fixed.glb
  → convert_duck_deformable.py (Isaac Sim in-editor)
  → object_1_soft.usda (PhysX tet mesh, Young's 5e5, Poisson 0.499)
```

### Mesh Cleaning

**Script:** `vbti/logic/reconstruct/clean_mesh.py` (Polyscope GUI)

Operations:
1. Connected component filter (min_ratio=0.01)
2. Statistical outlier removal (neighbors=20, std=2.0)
3. OBB cropping (interactive rotation + half-extents)

All operations preview live before committing. Output: `mesh_cleaned.ply`.

---

## Dataset Inspection

**Entry point:** `python -m vbti.logic.dataset.inspect_dataset`

### Unit Detection Logic

```python
if abs_max <= 1.1:        "normalized [-1, 1]"
elif abs_max <= 2π + 0.5: "radians"
elif abs_max <= 360:       "degrees"
else:                      "unknown"
```

### Reports

- `report_lerobot(path)` — Full LeRobot dataset inspection (parquet schema, stats, video info)
- `report_hdf5(path)` — Full HDF5 inspection (tree, episode lengths, action/state stats, images)
- `hdf5_action_state_correspondence()` — Checks if action ≈ state(t), state(t+1), or delta
