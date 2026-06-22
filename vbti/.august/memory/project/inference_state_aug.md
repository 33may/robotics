---
name: inference_state_aug
description: Detection-augmented state vector layout for v014+ SmolVLA models — required at inference, ordering rules, RGB pitfall
type: project
originSessionId: 649e2b5b-6dba-46cd-bf13-5766608383d4
---
## v014+ models use a 22-d augmented `observation.state`

Dataset `eternalmay33/01_02_03_merged_may-sim_detection` (and any model trained on it, including v014 step_020000) ships an augmented state, not the raw 6-d joint vector.

Layout (verified against the preprocessor stats safetensors and dataset `meta/info.json`):

```
[0:6]    joints — degrees: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
[6:22]   detection — normalized cx/cy in [0, 1], 4 cams × 2 objects × 2 coords:
         left_duck, left_cup, right_duck, right_cup,
         top_duck,  top_cup,  gripper_duck, gripper_cup
```

**Camera and object order is canonical to the dataset, NOT the user's `camera_names` ordering** — re-ordering silently mis-aligns the normalizer's mean/std subtraction. Constants live in `vbti/logic/inference/run_real_inference.py` as `DETECTION_AUG_CAM_ORDER` / `DETECTION_AUG_OBJ_ORDER`.

Use `center_norm` from `StudentDetector.detect()`. Zero-fill for not-found objects and missing cameras (matches how `detection_results.parquet` was written).

**Why:** v014 was trained with detection cx/cy concatenated into observation.state. Inference without augmentation crashes in `normalize_processor._apply_transform` with `tensor a (6) must match tensor b (22)`.

**How to apply:**
- Inference: pass `--detection=true` to both `run_real_inference.py run` and `eval_engine.py run`. The flag is now wired to both build the 16-d aug vector and draw the overlay (one flag, both behaviors).
- New v0XX models that share this dataset: same flag works, no code changes.
- If a future model uses a different aug layout, update the two `DETECTION_AUG_*_ORDER` constants — everything else flows from those.

## RGB pitfall fixed 2026-04-29

`_capture_frames` returns RGB (RealSense raw + OpenCV cameras converted in cameras.py:138). The old `_run_detection_overlay` did an extra `cv2.cvtColor(frame, COLOR_BGR2RGB)` on already-RGB frames, channel-swapping before passing to the detector. That meant live inference fed the detector BGR-as-RGB while the dataset (built via `ffmpeg -pix_fmt rgb24`) was true RGB — train/inference distribution mismatch on the augmentation values. Now removed.

**How to apply:** if adding new live detection paths, frames from `_capture_frames` are already RGB — pass them directly to `StudentDetector.detect()`.

## Hold-last-good required for v014+ (fixed 2026-04-29)

Symptom that triggered this: v014 (with detection) drove the arm toward the action-mean / "mid-range" pose every trial, while v013 (no detection) worked. Closed-loop sensitivity probe showed the model is highly sensitive to the 16 detection dims — zeroing them shifts predictions by ~9° MAE; randomizing them by ~15°. Within the training distribution it's <1° MAE.

Dataset (`01_02_03_merged_may-sim/detection_results.parquet`) was built by `process_dataset.py` with `StudentDetector(run="m1_baseline")` AND `apply_confidence_hold` (process_dataset.py:174): when a detection's conf < threshold, the row's `cx/cy` get **replaced with the previous good frame's** (and `conf := 0`). Verified: 153/153 conf=0 frames in episode 58 match the immediately prior frame's cx/cy exactly.

Live inference must do the same: **`DetectionStateHolder` in `run_real_inference.py`** keeps `last[(cam, obj)] = (cx, cy)` per trial, emits the held value when the detector returns not-found, and zero-fills only before any successful detection. `eval_engine.run` calls `state_holder.reset()` at the start of each trial (mirrors per-episode reset).

Detector identity check (sanity-tested 2026-04-29 against samples 10000/60000/100000): live `StudentDetector(run="m1_baseline")` outputs match the parquet to ≤0.001 across all 8 cam×obj pairs when conf>0 — same detector built the dataset.

**How to apply:** any new detection-augmented model trained on the same pipeline → use the holder, not the stateless `_detection_state_vector` (which is kept only for tests). If you switch detectors / re-process the dataset with a different `--student-run` or threshold, re-run the live-vs-parquet sanity diff before trusting inference.
