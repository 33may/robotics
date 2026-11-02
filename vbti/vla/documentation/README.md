# Robot Learning Pipeline Documentation

This folder is the current source of truth for the robot learning pipeline.

Start here:

- `SYSTEM_TEXTBOOK.md` - consolidated textbook for the full system.
- `logic/README.md` - readable documentation mirroring the actual `logic/` code folders.
- `logic/cameras/` - RealSense/OpenCV presets, viewing, capture, USB/reset, integration.
- `logic/servos/` - SO-101 scan, calibration profiles, rest, EEPROM recovery.
- `logic/dataset/` - LeRobot/HDF5 schema, inspection, conversion, transforms, UVA features.
- `logic/depth/` - D405 capture, packed/baked depth, train/inference parity.
- `logic/detection/` - Grounding DINO, phase labels, detection-state augmentation, distillation.
- `logic/train/` - config schema, local engine, remote training, chains, experiment history.
- `logic/inference/` - real inference, protocol eval, protocol schemas, session analysis.
- `logic/reconstruct/` - video/COLMAP/MILo, USD/Isaac, scripts, Cosmos prep.
- `runbooks/README.md` - executable scenarios for collection, training, remote runs, chains, and evaluation.
- `agents/` - specialist agent definitions for each module.
- `evidence/terminal_and_session_evidence.md` - command patterns found in zsh history and Claude sessions.

The broad files under `modules/` are overview summaries. For actual code work, prefer `logic/`.

Older docs under `docs/` are retained as historical notes.
