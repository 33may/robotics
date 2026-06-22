# Dataset Transforms

## Rule: New Artifact First

Dataset transforms should normally write a new dataset. Do not rename-over-source or delete source datasets unless explicitly requested.

## Aggregate Datasets

```bash
python -m vbti.logic.dataset.loading_utils aggregate \
  --datasets="eternalmay33/ds1,eternalmay33/ds2" \
  --output="eternalmay33/mix_name"
```

Use only when schemas match. If one dataset has depth and the other does not, add a placeholder depth feature first or train separately.

## Subsample Episodes

```bash
python -m vbti.logic.dataset.subsample \
  --src=eternalmay33/duck_cup_v020_all \
  --stride=4 \
  --dst=eternalmay33/duck_cup_v020_every4 \
  --start=0
```

Defaults:

- `start=0`;
- `output_dir=None`;
- `stride >= 2`.

Used for data-efficiency sweeps, e.g. every 16th/8th/4th/2nd episode.

## Strip Feature

```bash
python -m vbti.logic.dataset.strip_feature \
  --src=<repo_id> \
  --dst=<new_repo_id> \
  --feature=observation.images.gripper_depth
```

Optional flags:

```bash
--overwrite
--no-verify
```

Use when creating an RGB-only comparison dataset from a depth dataset.

## Add Zero Depth Placeholder

```bash
python -m vbti.logic.dataset.add_zero_depth_turbo \
  --src=<no_depth_repo> \
  --dst=<padded_repo> \
  --reference=<depth_repo> \
  --feature=observation.images.gripper_depth \
  --height=480 --width=640
```

Purpose: make a no-depth dataset schema-compatible with a depth dataset. The feature is black, meaning “no depth signal”.

## Detection/Phase Augmentation

Generate detections:

```bash
python -m vbti.logic.detection.process_dataset <repo_id_or_path>
```

Generate phases:

```bash
python -m vbti.logic.detection.phases <repo_id_or_path>
```

Bake features:

```bash
python -m vbti.logic.dataset.augment <repo_id_or_path> \
  --augmentations detection phase \
  -o <output_repo_id> \
  --cameras left right top gripper
```

Options:

```bash
--drop top_duck
--include-confidence
--detection-parquet detection_results_hold.parquet
--root /optional/root
--dry-run
```

Warning: detection augmentation changes `observation.state` from 6D to 22D. Only use it for checkpoints/configs expecting 22D state.

## `augment_all.py` Status

`augment_all.py` appears stale because it calls `augment_dataset()` with arguments that no longer match the current signature. Prefer explicit `augment.py` commands unless this file is repaired.

## Transform Verification

After any transform:

```bash
python -m vbti.logic.dataset.check_utils report <new_repo_id>
python -m vbti.logic.dataset.check_utils cameras <new_repo_id>
```

For state-changing transforms, inspect action/state dimensions and stats.
