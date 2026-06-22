# Dataset Inspection

## List Datasets

```bash
python -m vbti.logic.dataset.check_utils ls
```

Lists available local LeRobot datasets from the default cache.

## Basic Info

```bash
python -m vbti.logic.dataset.check_utils info <repo_id_or_path>
```

Use this before training to check:

- dataset path;
- episode count;
- frame count;
- FPS;
- feature keys;
- robot/action/state dimensions.

## Camera Keys

```bash
python -m vbti.logic.dataset.check_utils cameras <repo_id_or_path>
```

Use this to verify the dataset has the same image keys that the config/inference will use.

## Full Report

```bash
python -m vbti.logic.dataset.check_utils report <repo_id_or_path>
```

Full report checks metadata, schema, stats, camera keys, action/state ranges, episode lengths, and video information.

## Action Plots

```bash
python -m vbti.logic.dataset.check_utils plot_actions <repo_id_or_path> --save=/tmp/actions.png
```

Use this to detect:

- rest-position clusters;
- action saturation;
- missing joint movement;
- real/sim distribution mismatch;
- bad trims or idle-heavy episodes.

## Compare Action Distributions

```bash
python -m vbti.logic.dataset.check_utils compare_actions --datasets='{"real":"/path/real","sim":"/path/sim"}'
```

Use before mixing datasets. Similar schema does not mean similar action distribution.

## HDF5 Reports

```bash
python -m vbti.logic.dataset.hdf5_utils info /path/to/file.hdf5
python -m vbti.logic.dataset.hdf5_utils report /path/to/file.hdf5
python -m vbti.logic.dataset.hdf5_utils view /path/to/file.hdf5 0 all --sensor all
```

The HDF5 report is used for simulation episodes before conversion to LeRobot.

Typical HDF5 schema:

```text
data/demo_NNN/
  obs/
    side_cam_rgb
    side_cam_depth
    side_cam_seg
    table_cam_rgb
    table_cam_depth
    table_cam_seg
    wrist_rgb
    wrist_depth
    wrist_seg
    joint_pos
    joint_vel
  actions
```

## Depth Viewer

```bash
python -m vbti.logic.dataset.depth_viewer raw /path/to/packed.png --out=/tmp/depth.png
python -m vbti.logic.dataset.depth_viewer frame <repo_id> --episode 0 --frame 50 --out=/tmp/frame.png
python -m vbti.logic.dataset.depth_viewer replay <repo_id> --episode 0
```

Defaults:

- `depth-key=observation.images.gripper_depth`;
- `depth-scale-m=1e-4`;
- clip `0.05-0.20 m`.

## Inspection Checklist

Before training, write down:

- repo/path;
- episode/frame count;
- FPS;
- action/state dimension;
- camera keys;
- presence/absence of `gripper_depth`;
- task text(s);
- whether detection-state augmentation is present;
- whether UVA video features are present;
- action/state distribution sanity.
