# Dataset Schema Contract

## LeRobot Layout

LeRobot v3 datasets are usually stored under:

```text
~/.cache/huggingface/lerobot/<org>/<repo>
```

Typical layout:

```text
meta/
  info.json
  stats.json
  episodes/*.parquet
data/
  chunk-*/file-*.parquet
videos/
  observation.images.<camera>/chunk-*/file-*.mp4
```

## Core Feature Keys

Standard real duck-cup datasets use:

```text
action
observation.state
observation.images.top
observation.images.left
observation.images.right
observation.images.gripper
task
```

Depth-extended datasets add:

```text
observation.images.gripper_depth
```

UVA-extended datasets add a video-feature key such as:

```text
observation.video_features.siglip_output_4x4
```

Detection-augmented datasets expand `observation.state` from 6D to 22D.

## Joint Order

The six-dimensional action/state order is:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

Docs and code often shorten this to:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

## Camera Contract

Camera keys must match the training config and inference camera names.

Example v020-style image list:

```yaml
dataset:
  cameras:
    names:
      - top
      - left
      - right
      - gripper
      - gripper_depth
```

Changing camera order or names changes the observation contract. Do not rename cameras casually.

## Depth Contract

Depth is not a numeric model branch. It is represented as an image stream:

```text
observation.images.gripper_depth
```

The training dataset and live inference must use the same clipping/colorization convention. Current task-relevant close-range clip is usually:

```text
min = 0.05 m
max = 0.20 m
scale = 1e-4 m per uint16 unit
```

## Detection-State Contract

Detection-augmented state is:

```text
6 joints + 16 detection values = 22 values
```

Detection values are ordered as:

```text
[left, right, top, gripper] x [duck, cup] x [cx, cy]
```

This order is fixed and is not automatically derived from arbitrary camera order.

## Resolve Function

Use `resolve_dataset_path()` from `logic/dataset/__init__.py` when code accepts a dataset repo ID or path. Resolution order is:

1. explicit root/path if valid;
2. existing path;
3. Hugging Face LeRobot cache path;
4. suggestions/error.

## Destructive Operations

`delete_dataset()` exists and can remove dataset artifacts. Do not use it unless explicitly approved.
