# LeIsaac SmolVLA Training & Inference Debug Session

**Date:** 2026-02-05
**Duration:** ~3 hours
**Goal:** Train SmolVLA on lift_cube_3cams dataset and run inference in Isaac Sim

---

## Executive Summary

Successfully trained SmolVLA policy on custom lift cube dataset and deployed it in Isaac Sim simulation. Encountered and resolved **11 major issues** spanning package imports, data format mismatches, and unit conversions.

**Key Insight:** When bridging training data (LeRobot format) with simulation (Isaac Sim), careful attention to unit conversions is critical. The data pipeline involves multiple transformations that must be correctly reversed during inference.

---

## Table of Contents

1. [Environment Setup Issues](#1-environment-setup-issues)
2. [Policy Loading Issues](#2-policy-loading-issues)
3. [Data Format Issues](#3-data-format-issues)
4. [Normalization & Unit Issues](#4-normalization--unit-issues)
5. [Complete Data Pipeline](#5-complete-data-pipeline)
6. [Files Modified](#6-files-modified)
7. [Lessons Learned](#7-lessons-learned)

---

## 1. Environment Setup Issues

### Issue 1.1: leisaac Package Import Failure

**Error:**
```
ModuleNotFoundError: No module named 'leisaac.assets.robots.lerobot'
```

**Root Cause:**
Missing `__init__.py` files in leisaac subpackages:
- `leisaac/assets/robots/`
- `leisaac/assets/scenes/`
- `leisaac/assets/`

**Fix:**
Created missing `__init__.py` files:

```python
# leisaac/assets/robots/__init__.py
"""Robot asset configurations."""
from .lerobot import SO101_FOLLOWER_CFG

# leisaac/assets/scenes/__init__.py
"""Scene asset configurations."""
from .simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from .kitchen import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_HAMBURGER_CFG
from .bedroom import LIGHTWHEEL_BEDROOM_CFG
from .toyroom import LIGHTWHEEL_TOYROOM_CFG

# leisaac/assets/__init__.py
"""Asset configurations for leisaac."""
from . import robots
from . import scenes
```

**Location:** `leisaac/source/leisaac/leisaac/assets/*/`

---

### Issue 1.2: Template Task Import Failure

**Error:**
```
ModuleNotFoundError: No module named 'leisaac.assets.robots.lerobot'
(during template task import)
```

**Root Cause:**
The `template` task in leisaac has broken imports and shouldn't be loaded during normal operation.

**Fix:**
Added "template" to the blacklist in `leisaac/tasks/__init__.py`:

```python
# Before
_BLACKLIST_PKGS = ["utils", ".mdp"]

# After
_BLACKLIST_PKGS = ["utils", ".mdp", "template"]
```

**Location:** `leisaac/source/leisaac/leisaac/tasks/__init__.py:15`

---

### Issue 1.3: Assets Path Not Found

**Error:**
```
pxr.Tf.ErrorException: Failed to open layer @/home/.../assets/scenes/table_with_cube/scene.usd@
```

**Root Cause:**
`LEISAAC_ASSETS_ROOT` environment variable not set. The code defaults to git root + `/assets`, but assets are in `leisaac/assets/`.

**Fix:**
Set environment variable in inference script before imports:

```python
import os
from pathlib import Path

_robotics_root = Path(__file__).parents[3]
_leisaac_assets = _robotics_root / "leisaac" / "assets"
if _leisaac_assets.exists():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(_leisaac_assets)
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:25-27`

---

### Issue 1.4: leisaac Source Path Not in sys.path

**Error:**
```
ModuleNotFoundError: No module named 'leisaac.tasks'
```

**Root Cause:**
Python editable install doesn't properly expose nested subpackages.

**Fix:**
Added leisaac source to sys.path:

```python
import sys
_leisaac_src = _robotics_root / "leisaac" / "source" / "leisaac"
if _leisaac_src.exists() and str(_leisaac_src) not in sys.path:
    sys.path.insert(0, str(_leisaac_src))
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:21-23`

---

### Issue 1.5: Missing Camera Flag

**Error:**
```
RuntimeError: A camera was spawned without the --enable_cameras flag.
```

**Root Cause:**
Isaac Sim requires explicit flag to enable camera rendering.

**Fix:**
Run with `--enable_cameras` flag:

```bash
python -m vbti.utils.inference.run_smolvla_inference \
    --checkpoint=... \
    --task=LeIsaac-SO101-LiftCube-v0 \
    --enable_cameras
```

---

## 2. Policy Loading Issues

### Issue 2.1: PreTrainedPolicy Abstract Class

**Error:**
```
TypeError: Can't instantiate abstract class PreTrainedPolicy with abstract methods...
```

**Root Cause:**
Attempted to use `PreTrainedPolicy.from_pretrained()` to get preprocessor, but it's an abstract base class.

**Fix:**
Use `PolicyProcessorPipeline.from_pretrained()` instead:

```python
# Before (wrong)
from lerobot.policies.pretrained import PreTrainedPolicy
preprocessor = PreTrainedPolicy.from_pretrained(checkpoint_path).preprocessor

# After (correct)
from lerobot.processor import PolicyProcessorPipeline
preprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=checkpoint_path,
    config_filename="policy_preprocessor.json",
)
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:69-82`

---

### Issue 2.2: Wrong Import Path for PolicyProcessorPipeline

**Error:**
```
ModuleNotFoundError: No module named 'lerobot.policies.processor'
```

**Root Cause:**
Incorrect import path.

**Fix:**
```python
# Before (wrong)
from lerobot.policies.processor import PolicyProcessorPipeline

# After (correct)
from lerobot.processor import PolicyProcessorPipeline
```

---

## 3. Data Format Issues

### Issue 3.1: Image Dtype Mismatch

**Error:**
```
RuntimeError: "upsample_bilinear2d_out_frame" not implemented for 'Byte'
```

**Root Cause:**
Isaac Sim returns images as `uint8` (0-255), but policy expects `float32` (0-1).

**Fix:**
```python
# Convert images from uint8 to float
if img.dtype == torch.uint8:
    img = img.float() / 255.0
```

**Also:** Convert NHWC → NCHW:
```python
if img.dim() == 4 and img.shape[-1] == 3:
    img = img.permute(0, 3, 1, 2)  # NHWC → NCHW
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:127-131`

---

## 4. Normalization & Unit Issues

### Issue 4.1: Missing Postprocessor (CRITICAL)

**Symptom:**
Robot moves erratically, actions don't match expected behavior.

**Root Cause:**
SmolVLA outputs **normalized** actions (scaled by mean/std from training). The postprocessor that denormalizes actions was never loaded or applied.

**Training saves 3 components:**
1. `config.json` - model configuration
2. `policy_preprocessor.json` - normalizes inputs (state, images)
3. `policy_postprocessor.json` - **denormalizes outputs (actions)** ← WAS MISSING

**Fix:**
Load and apply postprocessor:

```python
def load_policy(checkpoint_path, device):
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)

    # Normalizes inputs
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, config_filename="policy_preprocessor.json"
    )

    # CRITICAL: Denormalizes outputs
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, config_filename="policy_postprocessor.json"
    )

    return policy, preprocessor, postprocessor

# In inference loop:
actions_normalized = policy.select_action(policy_obs)
actions_dict = {"action": actions_normalized}
actions_denorm = postprocessor(actions_dict)["action"]  # Denormalize!
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:67-91, 235-237`

---

### Issue 4.2: Action Unit Mismatch - Degrees vs Radians (CRITICAL)

**Symptom:**
Robot moves but actions are completely wrong scale.

**Debug output showed:**
```
Denormalized: [-4.7, -7.7, -1.3, 5.96, 1.5, 10.7]  ← degrees
Current joints: [-0.16, -0.008, -0.13, 0.17, 0.17, 0.17]  ← radians
```

**Root Cause:**
- Training data conversion script converts radians → degrees: `joint_pos = joint_pos / np.pi * 180`
- Model outputs degrees (after denormalization)
- Isaac Sim expects radians

**Fix:**
```python
# Convert degrees to radians for Isaac Sim
import math
actions_rad = actions_denorm * (math.pi / 180.0)
env.step(actions_rad)
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:240-241`

---

### Issue 4.3: State Input Unit Mismatch - Radians vs Degrees (CRITICAL) - FIXED SAWTOOTH

**Symptom:**
Robot shows sawtooth pattern in actions, jumps back periodically every ~50 steps.

**Root Cause:**
- Training data has state in **degrees**
- Inference passes state in **radians** from Isaac Sim
- Preprocessor normalizes using degree-based statistics
- Model receives incorrect state → predicts wrong actions → appears to "reset"

**Example:**
```
Joint at 0.5 radians = 28.6 degrees
Passed as: 0.5 → normalized: (0.5 - 2.55) / 8.69 = -0.24 (WRONG)
Should be: 28.6 → normalized: (28.6 - 2.55) / 8.69 = 3.0 (CORRECT)
```

**Why this caused sawtooth pattern:**
The model couldn't track its actual position because the state values were ~57x smaller than expected (radians vs degrees). Every time a new action chunk was generated, the model thought the robot was near the starting position (small state values) and predicted "from start" actions.

**Fix:**
```python
# State - convert from radians (Isaac) to degrees (training data)
if "joint_pos" in obs_dict:
    import math
    state_rad = obs_dict["joint_pos"]
    state_deg = state_rad * (180.0 / math.pi)  # radians → degrees
    policy_input["observation.state"] = state_deg
```

**Location:** `vbti/utils/inference/run_smolvla_inference.py:115-119`

**Result:** Sawtooth pattern eliminated after this fix

---

## 5. Complete Data Pipeline

### Training Pipeline
```
Isaac Sim (data collection)
    ↓ joint positions in RADIANS
Conversion Script (isaaclab2lerobot_3cam.py)
    ↓ joint_pos = joint_pos / np.pi * 180  # → DEGREES
LeRobot Dataset
    ↓ stored in DEGREES, images as video
Training
    ↓ preprocessor normalizes: (degrees - mean) / std
Model learns NORMALIZED actions
    ↓
Checkpoint saves: config.json, policy_preprocessor.json, policy_postprocessor.json
```

### Inference Pipeline (Correct)
```
Isaac Sim (observation)
    ↓ joint_pos in RADIANS, images uint8 NHWC
obs_to_policy_input()
    ├─ state: radians * (180/π) → DEGREES
    ├─ images: uint8→float, /255, NHWC→NCHW
    └─ task: "Pick up the cube..."
preprocessor()
    ↓ normalizes state using degree-based stats
policy.select_action()
    ↓ outputs NORMALIZED actions
postprocessor()
    ↓ denormalizes: normalized * std + mean → DEGREES
degrees→radians conversion
    ↓ degrees * (π/180) → RADIANS
env.step(radians)
```

---

## 6. Files Modified

| File | Changes |
|------|---------|
| `vbti/utils/inference/run_smolvla_inference.py` | Main inference script - sys.path, env var, preprocessor/postprocessor loading, unit conversions, debug output, action plotting |
| `vbti/utils/datasets/check_converted_dataset.py` | Added image dtype checking via LeRobot API |
| `leisaac/.../assets/__init__.py` | Created - package init |
| `leisaac/.../assets/robots/__init__.py` | Created - robot configs export |
| `leisaac/.../assets/scenes/__init__.py` | Created - scene configs export |
| `leisaac/.../tasks/__init__.py` | Added "template" to blacklist |

---

## 7. Lessons Learned

### 7.1 Unit Conversion Chain
When training data undergoes unit conversion:
1. **Document all conversions** in the pipeline
2. **Reverse ALL conversions** during inference
3. **Both input AND output** need conversion (easy to forget one)

### 7.2 Normalization Pipeline
SmolVLA (and similar policies) have TWO processors:
- **Preprocessor**: normalizes inputs before model
- **Postprocessor**: denormalizes outputs after model

Both must be loaded and applied correctly.

### 7.3 Debug Strategy
When model doesn't work:
1. **Print actual values** at each pipeline stage
2. **Compare with training data** statistics
3. **Check units** - often the culprit
4. **Plot actions over time** - reveals patterns like chunking issues

### 7.4 Isaac Sim Specifics
- Requires `--enable_cameras` for camera rendering
- Joint positions are in **radians**
- Images are **uint8 NHWC**

### 7.5 LeRobot Specifics
- `PolicyProcessorPipeline.from_pretrained()` loads processors
- `select_action()` returns ONE action (uses internal queue)
- Action queue has `n_action_steps` (typically = chunk_size) actions

---

## Appendix: Quick Reference

### Run Inference
```bash
cd /home/may33/projects/ml_portfolio/robotics
python -m vbti.utils.inference.run_smolvla_inference \
    --checkpoint=outputs/train/smolvla_lift_cube_3cams/checkpoint_005000 \
    --task=LeIsaac-SO101-LiftCube-v0 \
    --enable_cameras \
    --plot_actions
```

### Check Dataset
```bash
python -m vbti.utils.datasets.check_converted_dataset \
    --repo_id=eternalmay33/lift_cube_3cams
```

### Unit Conversion Formulas
```python
# Radians ↔ Degrees
degrees = radians * (180 / math.pi)
radians = degrees * (math.pi / 180)

# Normalization ↔ Denormalization
normalized = (raw - mean) / std
raw = normalized * std + mean
```

### Training Data Stats (for reference)
```
Action mean (deg): [2.55, 13.68, -19.39, 76.21, 4.79, 23.37]
Action std (deg):  [8.72, 24.19, 33.97, 18.91, 11.25, 16.16]
State mean (deg):  [2.55, 15.93, -17.21, 75.45, 4.75, 28.40]
State std (deg):   [8.69, 24.58, 33.84, 19.99, 11.22, 10.80]
```
