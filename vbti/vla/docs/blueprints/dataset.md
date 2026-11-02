# Dataset And Depth Modules

## Scope

Dataset code lives in:

- `logic/dataset/`
- `logic/depth/`

The dataset layer is the contract between collection, training, inference, and evaluation.

## LeRobot Dataset Contract

Core fields:

```text
action
observation.state
observation.images.top
observation.images.left
observation.images.right
observation.images.gripper
observation.images.gripper_depth   # optional depth-as-image stream
task
```

The six-dimensional robot state/action order is:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Camera names and order matter because SmolVLA builds its input interface from dataset metadata and rename maps.

## Important Files

| File | Purpose |
|---|---|
| `logic/dataset/__init__.py` | `resolve_dataset_path()` and destructive `delete_dataset()`. |
| `logic/dataset/check_utils.py` | List/report datasets, camera checks, action/state plots. |
| `logic/dataset/hdf5_utils.py` | HDF5 tree/report/viewer. |
| `logic/dataset/convert_utils.py` | HDF5 to LeRobot conversion, verification, recalibration/link helpers. |
| `logic/dataset/loading_utils.py` | Dataset loading, splitting, aggregation. |
| `logic/dataset/subsample.py` | Deterministic episode-stride dataset slicing. |
| `logic/dataset/add_video_features.py` | Bake UVA/SigLIP future-window teacher features. |
| `logic/dataset/strip_feature.py` | Create copy without a feature. |
| `logic/dataset/add_zero_depth_turbo.py` | Add black `gripper_depth` feature for merge compatibility. |
| `logic/dataset/depth_viewer.py` | View packed/baked depth. |
| `logic/depth/*` | Capture, colorize, bake, compare, and prepare depth. |

## Dataset Inspection Commands

```bash
python -m vla.logic.dataset.check_utils ls
python -m vla.logic.dataset.check_utils info eternalmay33/duck_cup_v020_all
python -m vla.logic.dataset.check_utils cameras eternalmay33/duck_cup_v020_all
python -m vla.logic.dataset.check_utils report eternalmay33/duck_cup_v020_all
python -m vla.logic.dataset.check_utils plot_actions eternalmay33/duck_cup_v020_all --save=/tmp/actions.png
```

Compare action distributions:

```bash
python -m vla.logic.dataset.check_utils compare_actions --datasets='{"real":"/path/real","sim":"/path/sim"}'
```

HDF5 inspection:

```bash
python -m vla.logic.dataset.hdf5_utils info /path/to/data.hdf5
python -m vla.logic.dataset.hdf5_utils report /path/to/data.hdf5
python -m vla.logic.dataset.hdf5_utils view /path/to/data.hdf5 0 all --sensor all
```

## HDF5 To LeRobot

Discover schema:

```bash
python -m vla.logic.dataset.convert_utils discover /path/to/data.hdf5
```

Convert:

```bash
python -m vla.logic.dataset.convert_utils convert \
  /path/to/data.hdf5 \
  eternalmay33/my_dataset \
  "Pick up the duck and place it in the cup"
```

With camera map:

```bash
python -m vla.logic.dataset.convert_utils convert \
  /path/to/data.hdf5 \
  eternalmay33/my_dataset \
  "Pick up the duck and place it in the cup" \
  --camera_map='{"cam_top":"top","cam_left":"left"}'
```

Verify:

```bash
python -m vla.logic.dataset.convert_utils verify ~/.cache/huggingface/lerobot/eternalmay33/my_dataset
```

Other helpers:

```bash
python -m vla.logic.dataset.convert_utils to_delta <source> <output>
python -m vla.logic.dataset.convert_utils recalibrate <source> <old_calib> <new_calib>
python -m vla.logic.dataset.convert_utils link <dataset_path> <repo_id>
python -m vla.logic.dataset.convert_utils ls
```

## Dataset Aggregation And Slicing

Aggregate datasets:

```bash
python -m vla.logic.dataset.loading_utils aggregate \
  --datasets="eternalmay33/ds1,eternalmay33/ds2" \
  --output="eternalmay33/mix_name"
```

Subsample every Nth episode:

```bash
python -m vla.logic.dataset.subsample \
  --src=eternalmay33/duck_cup_v020_all \
  --stride=4 \
  --dst=eternalmay33/duck_cup_v020_every4 \
  --start=0
```

`stride` must be at least `2`. This reindexes episodes/frames and creates a smaller dataset artifact.

## Depth Workflows

Depth is represented as an image feature, usually:

```text
observation.images.gripper_depth
```

The preferred model interface is turbo RGB depth with the same shape as normal RGB images. Training and inference must use the same clipping/colorization logic.

### Capture Real D405 Depth Sample

```bash
python -m vla.logic.depth.capture_gripper_sample \
  --serial 123622270367 \
  --seconds 5 \
  --width 640 --height 480 --fps 30 \
  --out /path/to/d405_gripper_sample
```

Outputs:

- `depth_uint16.npy`
- `color_uint8.npy`
- preview PNGs
- `histogram.png`

### Bake Packed Depth To Turbo RGB

```bash
python -m vla.logic.depth.bake_packed_depth \
  --repo_id=eternalmay33/04_05_06_07_merged_may-sim_depth \
  --out-repo-id=eternalmay33/04_05_06_07_merged_may-sim_depth_turbo \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 \
  --clip-max-m=0.20 \
  --depth-scale-m=1e-4 \
  --overwrite
```

Important: if `--out-repo-id` is omitted, the script may do an in-place temp/swap. Project preference is to create new artifacts unless in-place is explicitly intended.

### Add Estimated Gripper Depth

```bash
python -m vla.logic.depth.add_gripper_depth \
  --src eternalmay33/01_02_03_merged_may-sim_detection \
  --dst eternalmay33/01_02_03_merged_may-sim_detection_gripper-depth \
  --gripper-key observation.images.gripper \
  --depth-key observation.images.gripper_depth \
  --mode per-frame-norm
```

Default model is Depth Anything V2 Metric Indoor Small. Caveat: DA-V2 absolute depth scale differs from D405; `per-frame-norm` is useful for visual structure but loses absolute distance.

### Add Zero Depth Feature For Merge Compatibility

```bash
python -m vla.logic.dataset.add_zero_depth_turbo \
  --src eternalmay33/no_depth_dataset \
  --dst eternalmay33/no_depth_dataset_padded \
  --reference eternalmay33/depth_dataset \
  --feature observation.images.gripper_depth \
  --height 480 --width 640
```

This creates a black “no depth signal” feature so schemas match for merging/training.

### View Depth

```bash
python -m vla.logic.dataset.depth_viewer raw /path/to/packed.png --out=/tmp/depth.png
python -m vla.logic.dataset.depth_viewer frame eternalmay33/depth_dataset --episode 0 --frame 50 --out=/tmp/frame.png
python -m vla.logic.dataset.depth_viewer replay eternalmay33/depth_dataset --episode 0
```

### Compare Real Vs Estimated Depth

```bash
python -m vla.logic.depth.compare_real_vs_estimated \
  --d405-dir /path/to/d405_sample \
  --src-dataset eternalmay33/01_02_03_merged_may-sim_detection \
  --n-est-samples 12 \
  --n-candidates 60 \
  --max-median-depth-m 0.85 \
  --out /path/to/results
```

Outputs: `summary.json`, histogram plots, and candidate panels.

## UVA / Future Feature Baking

Bake teacher SigLIP/UVA features:

```bash
python -m vla.logic.dataset.add_video_features \
  --dataset eternalmay33/duck_cup_v020_all \
  --teacher /path/to/pretrained_model \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --batch-size 8 \
  --dtype fp16 \
  --rewrite-chunk-rows 256 \
  --output /absolute/new_dataset
```

Remote evidence used:

```bash
python -m vla.logic.dataset.add_video_features \
  --dataset eternalmay33/duck_cup_v020_all \
  --root /home/vbti/anton/data/eternalmay33/duck_cup_v020_all \
  --teacher /home/vbti/anton/data/uva_teacher_v020_150k \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --batch-size 32 \
  --dtype fp16 \
  --output /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva
```

## Pitfalls

- Do not change dataset schema casually; training and inference depend on matching feature names.
- Remote training uses `lerobot-train`, so custom local DataLoader code is ignored remotely.
- Bake transforms into datasets or patch LeRobot for remote training.
- Prefer new dataset artifacts over destructive in-place edits.
- Depth model comparisons must distinguish real D405 depth from estimated DA-V2 depth.
