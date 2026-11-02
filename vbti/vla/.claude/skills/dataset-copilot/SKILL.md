---
name: dataset-copilot
description: Dataset copilot for the vla framework. Use when inspecting LeRobot or HDF5 datasets, converting sim HDF5 to LeRobot, checking cameras, trimming rest frames, comparing action distributions, subsampling, aggregating, or editing dataset features. Triggers on dataset, camera, convert, LeRobot, HDF5, parquet, episodes, actions, trim, rest bias, sim-to-real.
---

# Dataset Copilot

Inspect, convert, transform and verify LeRobot datasets for the SO-101 duck/cup setup. Paths are relative to `vbti/vla`; commands run there in the `vla` conda env (`pip install -e vbti/vla`).

## Session start

1. `python -m vla.logic.dataset.check_utils ls`
2. If `$ARGUMENTS` names a dataset, inspect it: `info`, `cameras`, `report`.
3. Otherwise ask which dataset.

## References

- [cli_reference.md](cli_reference.md): every dataset command and its arguments.
- [conversion_knowledge.md](conversion_knowledge.md): joint units, sim-to-real conversion, LeRobot v3.0 layout.
- `docs/modules/dataset/`, `docs/modules/depth/`, `docs/runbooks/dataset_preparation.md`, `docs/tutorial/04_dataset.md`.

## Workflows

### Inspect

```bash
python -m vla.logic.dataset.check_utils info <repo_id> [--root=R]
python -m vla.logic.dataset.check_utils cameras <repo_id> [--root=R]
python -m vla.logic.dataset.check_utils report <repo_id_or_path>
python -m vla.logic.dataset.check_utils plot_actions <repo_id> --save=/tmp/actions.png
python -m vla.logic.dataset.trim_utils trajectories <dataset_path> --save=/tmp/traj.png
```

### Rest bias

Real demos start and end at the rest pose. Models trained on them return to rest. Preview, then write a new trimmed dataset:

```bash
python -m vla.logic.dataset.trim_utils preview <dataset_path> --save=/tmp/trim.png
python -m vla.logic.dataset.trim_utils trim <repo_id> <new_repo_id> [--dist_thresh=10.0]
```

Trimming alone did not remove rest bias in v003 (predictions sat on the rest pose, not the trimmed mean). Look at the distribution and coverage, not only the rest frames.

### Convert sim HDF5 to LeRobot

```bash
python -m vla.logic.dataset.hdf5_utils info /path/to/data.hdf5
python -m vla.logic.dataset.convert_utils discover /path/to/data.hdf5
python -m vla.logic.dataset.convert_utils convert /path/to/data.hdf5 <repo_id> "<task text>" \
    --camera_map='{"cam_top":"top","cam_left":"left","cam_right":"right","wrist":"gripper"}'
python -m vla.logic.dataset.convert_utils verify ~/.cache/huggingface/lerobot/<repo_id>
python -m vla.logic.dataset.check_utils report <repo_id>
```

### Compare datasets

```bash
python -m vla.logic.dataset.check_utils compare_actions --datasets='{"real":"/path/real","sim":"/path/sim"}' --save=/tmp/cmp.png
python -m vla.logic.dataset.trim_utils compare_trimmed --datasets='{"real":"/path/real","sim":"/path/sim"}' --save=/tmp/cmp_trim.png
```

### Build training sets

```bash
python -m vla.logic.dataset.loading_utils aggregate --datasets="<repo_a>,<repo_b>" --output=<out_repo>
python -m vla.logic.dataset.subsample <src> <stride> <dst>
python -m vla.logic.dataset.strip_feature --src <src> --dst <dst> --feature <key>
```

Depth baking and UVA feature commands: `docs/modules/depth/commands.md`, `docs/modules/dataset/uva_features.md`.

## Rules

- Write a new dataset; never mutate or delete a source dataset without explicit approval. `check_utils delete` and `trim_utils drop_episodes --inplace` are destructive.
- Run `cameras` before a dataset goes into a training config. Camera names and order are part of the observation contract.
- Plot distributions; summary stats hide rest bias and coverage gaps.
- Check the dataset's calibration profile. Joint values are only valid for the calibration they were recorded with.
- Depth is an image stream. Inference must use the same clip range and colorization as the baked dataset.
- Remote training runs `lerobot-train`. Local loader changes do not reach it unless baked into the dataset.
- Use the CLI for anything it covers; no ad-hoc inline Python for those operations.
- Joint units differ by source: real recordings use degrees for body joints and `[0, 100]` for the gripper; `convert_utils` writes sim data as `[-100, 100]` body / `[0, 100]` gripper. Check before mixing sim and real.
