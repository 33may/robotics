# `logic/detection`

## Purpose

`logic/detection` owns object detection labels, task phase labels, detection-state augmentation, and detector distillation/export.

It started as a detection-state augmentation experiment and remains useful for analysis, debugging, and explicit state augmentation experiments. The main final VLA direction avoids relying on detector coordinates as the primary interface unless explicitly testing that setup.

## Files

| File | Purpose |
|---|---|
| `detect.py` | Grounding DINO teacher, ONNX backend, student detector support. |
| `process_dataset.py` | Run detector over LeRobot dataset videos and write dense/interpolated parquet. |
| `phases.py` | Detect task phases from motion/state. |
| `distill/` | Filter labels, train student detector, export ONNX. |

## Docs

- `workflow.md` - dataset detection/phase/augmentation workflow.
- `state_augmentation.md` - 22D state contract and inference requirements.
- `distillation.md` - student detector/distillation commands.

## Current Detector Status

Current code uses Grounding DINO as the teacher detector. Older memory mentioning OWLv2 is stale for current code.
