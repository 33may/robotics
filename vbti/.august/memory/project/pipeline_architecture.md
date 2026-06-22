# Pipeline Architecture

## robot_utils.py — Scene-to-Task Pipeline

**File**: `vbti/utils/robot_utils.py`

Generates complete leisaac task configs from scene USD files. Entry point: `pipeline()`.

### Pipeline Steps
1. `create_no_robot_scene()` — strips robot, cameras, lights, /Render from scene USD
2. `generate_scene_asset()` — creates leisaac asset bridge file (scenes/{task_name}.py)
3. `create_task_boilerplate()` — creates task folder + gym registration
4. `generate_env_cfg()` — generates SceneCfg, ObsCfg, TermCfg, EnvCfg

### Key Functions
- `extract_scene_config()` — reads USD, extracts robot pos/orient, camera configs, lights, joints
- `_cam_field()` — generates TiledCameraCfg with spawn config (line 553)
- `_obs_cam_field()` — generates ObsTerm for camera image (line 631)
- `_light_field()` — generates AssetBaseCfg for lights
- `_discover_subassets()` — finds RigidBody/Articulation prims for domain randomization
- `parse_usd_and_create_subassets()` — called at runtime to create scene objects

### Camera Config Generation (current)
```python
# _cam_field() generates:
data_types=["rgb"]  # ONLY RGB — no depth/seg

# _obs_cam_field() generates:
ObsTerm(func=mdp.image, params={"data_type": "rgb"})  # ONLY RGB obs
```

### TODO: Add Cosmos sensor data to recording
To capture depth + segmentation during data collection (avoiding replay):
- Add `distance_to_camera`, `instance_segmentation_fast` to `data_types` in `_cam_field()`
- Add corresponding ObsTerms in `_obs_cam_field()` or separate obs group
- This way HDF5 dataset contains all modalities needed for Cosmos Transfer

---

## Cosmos Transfer Pipeline

**File**: `vbti/utils/cosmos_transfer.py`

5-step pipeline for data augmentation via NVIDIA Cosmos Transfer:

1. **capture** — replay episode in Isaac Sim with depth+seg cameras → frames
2. **process** — frames → mp4 videos + Canny edge maps
3. **config** — generate Cosmos Transfer config JSON
4. **transfer** — run Cosmos inference (TODO)
5. **reassemble** — output video → frames → HDF5 (TODO)

### Key Decision: State vs Action Replay
- `replay_mode="state"` (default) — feeds recorded joint positions as PD targets
- `replay_mode="action"` — re-simulates from raw actions (non-deterministic!)
- State mode needed because PhysX GPU is non-deterministic for contact (grasps fail)

### Future Direction
Instead of replaying, collect ALL sensor data during recording by modifying
`generate_env_cfg()` to include depth/seg in camera data_types. This eliminates
the replay problem entirely.

---

## Isaac Sim Replay Non-Determinism

**NVIDIA officially documents** that replay is non-deterministic:
- PhysX contact solver state is NOT serializable
- GPU atomic operations cause ULP-level floating-point divergence per step
- Grasps are bifurcation points — tiny differences → success/failure
- `enable_enhanced_determinism=False` by default (perf cost to enable)
- `solve_articulation_contact_last=False` by default (not optimized for gripping)

**NVIDIA's recommendation**: collect more demos than needed, accept replay failures.

---

## HDF5 Dataset Structure (Isaac Lab format)

**File**: `datasets/vbti_table_v1/vbti_dataset_v1.hdf5` (85 GB)

```
data/demo_N/
  actions: (N_frames, 6) float32
  processed_actions: (N_frames, 6) float32
  initial_state/
    articulation/robot/{joint_position, joint_velocity, root_pose, root_velocity}
    rigid_object/{object_0, object_01}/{root_pose, root_velocity}
  obs/
    side_cam: (N_frames, 480, 640, 3) uint8     ← 3 cameras = most of 85GB
    table_cam: (N_frames, 480, 640, 3) uint8
    wrist: (N_frames, 480, 640, 3) uint8
    joint_pos, joint_vel, joint_pos_rel, joint_vel_rel: (N_frames, 6) float32
    actions: (N_frames, 6) float32
  states/...
```

- 110 episodes, ~46K total frames, LZF compression
- 85GB because raw uint8 images (129GB uncompressed, LZF only gets 66%)
- `demo_6` is broken (only has initial_state, no obs)
- Episode index ≠ demo number (sorted alphabetically: demo_10 before demo_2)

---

## VBTI Pipeline CLI Architecture (2026-02-23)

**Decision**: Phase-based Python CLI with checkpoints. Single entry point, modular utils per phase.

### Structure
```
vbti/
  pipeline.py              # single CLI entry point (subcommands per phase)
  utils/
    video_utils.py          # Phase 1: video stats, frame extraction
    colmap_utils.py         # Phase 2: COLMAP runner (future)
    milo_utils.py           # Phase 2: MILo training (future)
    create_scene_usd.py     # Phase 2/4: mesh → USDA conversion (exists)
    robot_utils.py          # Phase 4: scene → leisaac config (exists)
    cosmos_transfer.py      # Phase 5: data augmentation (exists)
```

### Key Principles
- Each phase is a subcommand: `vbti pipeline phase1 --video ...`
- Phases have manual checkpoints between them (review frames, check mesh quality)
- Shared config/state via directory structure convention
- External tools (COLMAP, MILo) called via `subprocess.run()` with live logs
- `sharp-frame-extractor` (cansik) installed in `vbti/libs/`, patched to Python >=3.11

### Libs
- `vbti/libs/sharp-frame-extractor/` — forked, `requires-python` patched from >=3.12 to >=3.11
- Env: `gsplat-pt25` (Python 3.11, PyTorch 2.5.1+cu124)

---

## Domain Randomization During Replay

- Object positions: randomized on reset but **overwritten** by recorded state
- Light rotation: **persists** (not a rigid body, not in state dict)
- Scale randomization: NOT present, planned for future (would need metadata in dataset)
