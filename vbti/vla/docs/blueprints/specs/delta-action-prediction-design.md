# Delta Action Prediction — Design Spec

## Problem

SmolVLA trained on duck-cup pick-place achieves ~90% in-distribution but <10% out-of-distribution. The model predicts absolute joint positions, anchoring the policy to training-time spatial coordinates. Switching to delta (relative) actions should decouple learned manipulation logic from absolute position.

## Decision: Step-wise Delta, Offline Conversion

**Step-wise delta**: `delta_action[t] = absolute_action[t] - observation.state[t]`

- Each action is the displacement from the current state to the next target
- Works cleanly with any action horizon at inference (horizon=1 means zero compounding error, larger horizons drift linearly but self-correct on re-observation)
- Can be precomputed offline — every frame has the same semantic meaning, so dataset-level MEAN_STD normalization is valid
- Gripper (index 5) stays absolute — binary/discrete open/close, delta doesn't apply

**Why not chunk-wise delta**: chunk-wise requires knowing the chunk start at training time (data-loader dependent), making offline precomputation impossible. Stats would also be invalid — position 0 and position 49 in a chunk have fundamentally different magnitude distributions, but SmolVLA applies one mean/std to the entire action feature. pi0 handles this because flow-matching has its own internal normalization; SmolVLA does not.

**Why not on-the-fly transform**: SmolVLA's normalization stats (`stats.json`) are computed at dataset creation time. On-the-fly delta would mean the stats don't match what the model sees. You'd need to precompute delta stats anyway (full dataset pass), so you might as well save the result.

## Components

### 1. Dataset Conversion Utility

**Location**: `vbti/logic/dataset/convert_utils.py` — new `convert_to_delta()` function alongside existing `convert()`.

**Input**: path to an existing absolute-action LeRobot dataset.

**Output**: new standalone LeRobot dataset at a user-specified path. Full copy (including videos) — no symlinks, no dependency on source dataset. Can be edited, trimmed, shared independently.

**Transform (per frame)**:
```
delta_action[t][0:5] = absolute_action[t][0:5] - observation.state[t][0:5]
delta_action[t][5]   = absolute_action[t][5]   # gripper stays absolute
```

**What gets written**:
- New parquet files with transformed action column
- Copied video files (unchanged)
- New `meta/stats.json` computed over delta actions
- Updated `meta/info.json` (same structure, new path)
- Copied `meta/episodes/`, `meta/tasks.parquet`

**CLI**: `python vbti/logic/dataset/convert_utils.py to_delta <source_path> <output_path>`

### 2. Inference Delta Mode

**Location**: `vbti/logic/inference/run_real_inference.py`

**Flag**: `--delta_actions` CLI argument (default: off).

**Logic when enabled**:
```python
target = current_state + predicted_delta       # joints 0-4
target[5] = predicted_delta[5]                 # gripper absolute
target = np.clip(target, joint_min, joint_max) # safety clamp
```

**Default behavior (flag off)**: unchanged — absolute action applied directly.

### 3. Training

No changes. Point SmolVLA config at the delta dataset (`repo_id` or local path). The dataset has correct stats baked in. Model architecture unchanged — SmolVLA already receives `observation.state` as input.

## What Does NOT Change

- Model architecture (SmolVLA)
- Training code / backend
- Data collection pipeline
- Video encoding
- Existing absolute datasets (untouched)

## Files Modified

| File | Change |
|------|--------|
| `vbti/logic/dataset/convert_utils.py` | Add `convert_to_delta()` function |
| `vbti/logic/inference/run_real_inference.py` | Add `--delta_actions` flag and conditional logic |

## Validation Plan

1. Convert duck_cup_130eps dataset to delta
2. Inspect delta stats — joint means should be near zero, gripper stats similar to original
3. Train SmolVLA on delta dataset with same hyperparameters as v010
4. In-distribution eval: should match v010 (~90%)
5. Out-of-distribution eval: duck at unseen positions — the key test
