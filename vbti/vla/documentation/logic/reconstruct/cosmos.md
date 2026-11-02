# Cosmos Transfer Prep

## File

```text
logic/reconstruct/cosmos_transfer.py
```

## Status

- `transfer()` uses Cosmos Transfer 1 hosted API, not Cosmos Transfer 2.5.
- `reassemble()` is not implemented.
- Actual Cosmos Transfer 2.5 work uses the external deployment/repo, not this script.

## Commands

Extract episode frames:

```bash
python -m vbti.logic.reconstruct.cosmos_transfer extract \
  --episode 0 \
  --dataset_file ./datasets/vbti_table_v2_cosmos/raw.hdf5
```

Process PNGs to videos/edges:

```bash
python -m vbti.logic.reconstruct.cosmos_transfer process --episode 0 --fps 30
```

Generate config:

```bash
python -m vbti.logic.reconstruct.cosmos_transfer config \
  --episode 0 \
  --camera side_cam \
  --prompt "photorealistic scene, natural lighting" \
  --variant default
```

Call API:

```bash
python -m vbti.logic.reconstruct.cosmos_transfer transfer \
  --config_file ./datasets/vbti_table_v2_cosmos/cosmos/configs/episode_000_side_cam_default.json
```

Batch prepare:

```bash
python -m vbti.logic.reconstruct.cosmos_transfer prepare \
  --dataset_file ./datasets/vbti_table_v2_cosmos/raw.hdf5 \
  --output_dir ./datasets/vbti_table_v2_cosmos/cosmos/cosmos_ready
```

## Data Layout

Input HDF5 expected shape:

```text
data/<episode>/obs/<camera>
data/<episode>/obs/<camera>_depth
data/<episode>/obs/<camera>_seg
```

Extracted output:

```text
cosmos/captures/episode_000/<cam>/rgb/*.png
cosmos/captures/episode_000/<cam>/depth/*.png
cosmos/captures/episode_000/<cam>/depth_raw/*.npy
cosmos/captures/episode_000/<cam>/seg/*.png
```

Processed output:

```text
cosmos/processed/episode_000/<cam>/rgb.mp4
cosmos/processed/episode_000/<cam>/depth.mp4
cosmos/processed/episode_000/<cam>/seg.mp4
cosmos/processed/episode_000/<cam>/edge.mp4
```

## Constants

- cameras: `side_cam`, `table_cam`, `wrist`;
- frame size: `640x480`;
- depth normalization: `0.01-2.0 m`;
- Canny thresholds: `50`, `150`.
