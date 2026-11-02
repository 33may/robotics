# 4. Dataset management

**What you have:** a LeRobot dataset from block 3. Datasets live in the LeRobot cache, `~/.cache/huggingface/lerobot/<user>/<dataset_name>/` (`meta/` info + episodes + stats, `data/` parquet frames, `videos/` per-camera mp4), and can be pushed to the Hugging Face Hub under the same repo id. Training and evaluation refer to datasets only by this repo id.

## Inspect

`python -m vla.logic.dataset.<tool>`:

| Tool | What it shows |
|---|---|
| `check_utils ls \| info \| cameras \| report` | Datasets in the cache, metadata, camera keys, full report |
| `check_utils plot_actions \| compare_actions` | Per-joint action plots; two datasets side by side |
| `trim_utils trajectories \| distribution` | Per-episode trajectories, joint distributions |
| `replay_utils show \| replay \| goto` | Inspect an episode; replay it on the follower |
| `hdf5_utils` | Simulation HDF5 datasets |
| `depth_viewer` | Dataset viewer with the depth stream |

## Transform

Every tool writes a new dataset; the source is never modified.

| Tool | What it does |
|---|---|
| `trim_utils preview \| trim \| drop_episodes \| compare_trimmed` | Cut rest frames, remove episodes |
| `subsample` | Deterministic episode subsets |
| `strip_feature` | Remove a feature |
| `convert_utils` | Simulation HDF5 → LeRobot |
| `add_zero_depth_turbo`, `depth_transform`, `logic/depth/*` | Depth as an extra image stream |
| `add_video_features` | Bake teacher visual features (UVA aux loss) |

## Agent-driven

Dataset work is meant to run through an agent — `docs/agents/dataset_agent.md` holds its instructions (sources to read, code scope, safe-operation rules). Full reference: `docs/modules/dataset/`.
