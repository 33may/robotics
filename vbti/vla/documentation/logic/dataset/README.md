# `logic/dataset`

## Purpose

`logic/dataset` owns LeRobot/HDF5 dataset inspection, conversion, loading, aggregation, subsampling, replay, feature editing, and training-target feature preparation.

It is the main contract layer between data collection, training, inference, and evaluation.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | `resolve_dataset_path()` and destructive `delete_dataset()`. |
| `check_utils.py` | LeRobot dataset listing, metadata, camera/schema/stats reports, action plots. |
| `hdf5_utils.py` | HDF5 tree/report/viewer for sim datasets. |
| `convert_utils.py` | HDF5 to LeRobot conversion and related helpers. |
| `loading_utils.py` | Dataset loading, train/val splitting, aggregation. |
| `replay_utils.py` | Robot/dataset replay and pose helpers. |
| `trim_utils.py` | Dataset trimming/episode curation utilities. |
| `viewer.py` | Augmented dataset viewer. |
| `augment.py` | Bake detection/phase features into `observation.state`. |
| `augment_all.py` | Batch augmentation helper; appears stale against current `augment.py`. |
| `subsample.py` | Deterministic episode-stride dataset slicing. |
| `strip_feature.py` | Create a dataset copy without a selected feature. |
| `add_zero_depth_turbo.py` | Add black depth-image feature for schema compatibility. |
| `add_video_features.py` | Bake teacher/UVA future video features. |
| `depth_transform.py` | Packed depth to turbo RGB transform for dataset reading. |
| `depth_viewer.py` | Packed/baked depth viewer. |
| `target_extractors/` | Teacher feature extraction hooks. |

## Docs

- `schema.md` - LeRobot schema and dataset contract.
- `inspection.md` - inspection/report/viewer commands.
- `conversion.md` - HDF5 to LeRobot conversion.
- `transforms.md` - subsample, aggregate, strip, zero-depth, augmentation.
- `uva_features.md` - `add_video_features.py` workflow.

## Critical Rules

- Prefer new dataset artifacts over in-place mutation.
- Camera keys and state dimensions are model interface, not metadata trivia.
- Remote training uses `lerobot-train`; custom loader/runtime transforms are ignored unless baked into the dataset or patched into LeRobot.
