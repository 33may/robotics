# `logic/depth`

## Purpose

`logic/depth` owns real D405 depth capture, depth colorization/packing, offline depth baking, estimated depth insertion, real-vs-estimated comparison, and realtime inference preparation.

Depth is a dataset/inference modality represented as an image stream, not a custom model branch.

## Files

| File | Purpose |
|---|---|
| `colorize.py` | Clip/colorize depth, pack/unpack uint16 into RGB. |
| `capture_gripper_sample.py` | Capture real gripper D405 RGB/depth sample. |
| `bake_packed_depth.py` | Convert packed depth feature into turbo RGB image feature. |
| `add_gripper_depth.py` | Add estimated depth from Depth Anything V2. |
| `realtime_prepare.py` | Runtime D405 uint16 to turbo RGB parity function. |
| `compare_real_vs_estimated.py` | Compare real D405 and estimated depth distributions. |

## Docs

- `commands.md` - all depth CLI commands.
- `train_inference_parity.md` - how to keep offline and realtime depth identical.

## Contract

Depth feature key:

```text
observation.images.gripper_depth
```

Canonical task clip:

```text
0.05 m to 0.20 m
```

Canonical D405 scale:

```text
1e-4 m per uint16 unit
```
