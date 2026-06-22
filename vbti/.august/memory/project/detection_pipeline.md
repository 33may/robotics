---
name: detection_pipeline
description: OWLv2 duck/cup detection pipeline + phase detection + dataset augmentation — full architecture and file locations
type: project
originSessionId: 0201284c-83d0-427d-9c40-1b23652763be
---
## Detection Pipeline (built 2026-04-16)

### Modules
- `vbti/logic/detection/detect.py` — DuckDetector (OWLv2 fp16, batch=8, threshold=0.08)
- `vbti/logic/detection/process_dataset.py` — Process LeRobot datasets: ffmpeg→OWLv2→parquet
- `vbti/logic/detection/phases.py` — Velocity-based phase detection (reach/pregrasp/grasp/transport/release)
- `vbti/logic/dataset/augment.py` — Merge detection+phase into observation.state (6d→23d)
- `vbti/logic/dataset/augment_all.py` — Convenience: augment all 3 datasets at once

### Augmented State Layout (23d, fits SmolVLA max_state_dim=32)
```
[0:6]   original joints
[6:18]  detection cx/cy for left/right/top × duck/cup
[18:23] phase one-hot (reach, pregrasp, grasp, transport, release)
```

### Detection Quality Notes
- Cup: 97-100% left/right, 77% top
- Duck: 33-35% left/right during reach/pregrasp, drops to 0 when gripper occludes
- Top camera weakest for duck (2.9%) — viewing angle issue
- Duck occlusion during grasp/transport is expected — interpolated values + confidence=0 encode this

### Phase Detection
- 244/244 episodes across all 3 datasets — 100% success
- Algorithm: velocity-based state machine on gripper signal
- Phase distribution varies by dataset (01 has release, 03 has none — recording protocol difference)

### 3D Position: NOT FEASIBLE
- No depth data in datasets (all cameras RGB-only, video.is_depth_map=False)
- No camera extrinsics calibrated
- calibrate.py is a visual alignment tool, not geometric calibration
- 2D detection from 3 cameras implicitly encodes 3D (model can learn triangulation)

**Why:** These modules are the building blocks for augmented training. Knowing the layout and quality characteristics prevents re-investigating.

**How to apply:** When setting up training, use augment_all.py to create augmented datasets. The 23d state works with SmolVLA's existing max_state_dim=32 — no model code changes needed. For inference, need to add live OWLv2 + phase detection to run_real_inference.py.
