# Dataset Transforms

## Rule: New Artifact First

Dataset transforms should normally write a new dataset. Do not rename-over-source or delete source datasets unless explicitly requested.

## Aggregate Datasets

```bash
python -m vla.logic.dataset.loading_utils aggregate \
  --datasets="eternalmay33/ds1,eternalmay33/ds2" \
  --output="eternalmay33/mix_name"
```

Use only when schemas match. If one dataset has depth and the other does not, add a placeholder depth feature first or train separately.

## Subsample Episodes

```bash
python -m vla.logic.dataset.subsample \
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
python -m vla.logic.dataset.strip_feature \
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
python -m vla.logic.dataset.add_zero_depth_turbo \
  --src=<no_depth_repo> \
  --dst=<padded_repo> \
  --reference=<depth_repo> \
  --feature=observation.images.gripper_depth \
  --height=480 --width=640
```

Purpose: make a no-depth dataset schema-compatible with a depth dataset. The feature is black, meaning “no depth signal”.

## Transform Verification

After any transform:

```bash
python -m vla.logic.dataset.check_utils report <new_repo_id>
python -m vla.logic.dataset.check_utils cameras <new_repo_id>
```

For state-changing transforms, inspect action/state dimensions and stats.
