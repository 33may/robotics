# `logic/` Module Documentation

This folder mirrors the actual source layout under `/home/may33/projects/ml_portfolio/robotics/vbti/logic`.

Use this tree when you need to work with code. The older `documentation/modules/*.md` files are broad domain summaries; this folder is the readable module-by-module reference.

## Module Index

| Source module | Documentation folder | What it owns |
|---|---|---|
| `logic/cameras` | `cameras/` | RealSense/OpenCV presets, live view, capture, calibration overlay, USB/reset utilities. |
| `logic/servos` | `servos/` | SO-101 Feetech scanning, calibration profiles, rest pose, EEPROM recovery. |
| `logic/dataset` | `dataset/` | LeRobot/HDF5 inspection, conversion, loading, aggregation, curation, feature transforms. |
| `logic/depth` | `depth/` | D405 capture, packed depth, turbo colorization, offline baking, realtime parity. |
| `logic/detection` | `detection/` | Grounding DINO/student detection, phase labels, detection-state features, distillation. |
| `logic/train` | `train/` | Config schema, local engine, remote `lerobot-train`, chains, experiment folders. |
| `logic/inference` | `inference/` | Real inference, protocol evaluation, protocol generation, session analysis/rendering. |
| `logic/reconstruct` | `reconstruct/` | Video/COLMAP/MILo reconstruction, USD conversion, Isaac/LeIsaac generation, Cosmos prep. |

## Reading Order

1. For a quick system-level overview, read `../SYSTEM_TEXTBOOK.md`.
2. For implementation work, open the folder matching the `logic/<module>` source folder.
3. For command usage, read each module's `commands.md` or workflow-specific file.
4. For agents, use `../agents/*.md`; they now point to this module tree.

## Important Cross-Module Contracts

- Cameras define dataset/inference image keys.
- Servos define calibration and joint meaning for `action` and `observation.state`.
- Dataset schema defines what training and inference must provide.
- Training configs define expected camera/state schema and checkpoint layout.
- Evaluation sessions are the only quantitative real-robot comparison evidence.
- Reconstruction/simulation data must convert back into the same LeRobot schema before training.
