# UVA / Teacher Video Features

## Purpose

`add_video_features.py` bakes future-window teacher features into a LeRobot dataset. This supports SmolVLA-UVA-style auxiliary prediction losses without requiring the remote dataloader to run custom extraction logic.

## Command

```bash
python -m vbti.logic.dataset.add_video_features \
  --dataset <repo_id> \
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

Important args:

| Arg | Meaning |
|---|---|
| `--dataset` | Source LeRobot repo/path. |
| `--root` | Optional explicit dataset root. |
| `--teacher` | Pretrained policy/model used as feature extractor. |
| `--layer` | Feature layer, e.g. `siglip_output`. |
| `--spatial-size` | Spatial grid size, commonly `4`. |
| `--t-future` | Future window length. |
| `--target-camera` | Camera image key to extract from. |
| `--batch-size` | Teacher extraction batch size. |
| `--dtype` | Feature dtype, often `fp16`. |
| `--output` | New dataset path. |

## Remote Bake Example

Observed remote workflow:

```bash
python -m vbti.logic.dataset.add_video_features \
  --dataset eternalmay33/duck_cup_v020_all \
  --root /home/vbti/anton/data/eternalmay33/duck_cup_v020_all \
  --teacher /home/vbti/anton/data/uva_teacher_v020_150k \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --batch-size 32 \
  --dtype fp16 \
  --rewrite-chunk-rows 256 \
  --output /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva
```

## Output Feature

Typical output feature:

```text
observation.video_features.siglip_output_4x4
```

Shape concept:

```text
(t_future, spatial_size, spatial_size, hidden_dim)
```

Future frames are clamped by episode boundaries.

## Why Bake

Remote training runs `lerobot-train`. It does not execute local custom dataloader code. Baking features into the dataset makes the feature available to the remote LeRobot training path.

## Verification

After baking:

```bash
python -m vbti.logic.dataset.check_utils report <new_dataset>
```

Check that:

- the new video feature exists;
- frame/episode counts match source;
- camera feature keys remain unchanged;
- the dataset path is accessible on the remote machine if training remotely.
