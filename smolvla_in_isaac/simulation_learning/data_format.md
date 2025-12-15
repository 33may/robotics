# Isaac Lab to LeRobot Dataset Conversion Guide

This document explains the complete data transformation pipeline for collecting robot demonstrations in Isaac Lab and converting them to LeRobot format for training imitation learning policies (like ACT).

## Table of Contents
- [Overview](#overview)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Isaac Lab HDF5 Format](#isaac-lab-hdf5-format)
- [LeRobot Dataset Format](#lerobot-dataset-format)
- [Critical Transformations](#critical-transformations)
- [Inference: LeRobot back to Isaac Lab](#inference-lerobot-back-to-isaac-lab)
- [Common Issues](#common-issues)

---

## Overview

The workflow consists of three main stages:

1. **Data Collection**: Collect demonstrations in Isaac Lab using teleoperation (saves to HDF5)
2. **Conversion**: Transform HDF5 data to LeRobot format using `isaaclab2lerobot.py`
3. **Training**: Train policies (e.g., ACT) using LeRobot's standard pipeline
4. **Inference**: Deploy trained policies back in Isaac Lab simulation

**Critical Point**: Data must be normalized during conversion and **denormalized** during inference, or the model will output wrong actions!

---

## Data Flow Pipeline

```
Isaac Lab Teleoperation
         ↓
    HDF5 Dataset
  (joint positions in radians, images as uint8)
         ↓
  Conversion Script (isaaclab2lerobot_3cam.py)
         ↓
   LeRobot Dataset
  (normalized joint positions in degrees, images as compressed video)
         ↓
     Training (ACT/Diffusion/etc.)
         ↓
  Trained Policy Checkpoint
         ↓
   Inference Script (test_act_policy.py)
         ↓
    Isaac Lab Simulation
  (denormalized actions back to radians)
```

---

## Isaac Lab HDF5 Format

### Location
`/home/may33/projects/ml_portfolio/robotics/leisaac/datasets/lift_cube.hdf5`

### Structure
```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions           # Shape: [T, 6], dtype: float32, units: radians
    │   ├── obs/
    │   │   ├── joint_pos     # Shape: [T, 6], dtype: float32, units: radians
    │   │   ├── front         # Shape: [T, H, W, 3], dtype: uint8 (0-255)
    │   │   ├── wrist         # Shape: [T, H, W, 3], dtype: uint8 (0-255)
    │   │   └── gripper_cam_cfg # Shape: [T, H, W, 3], dtype: uint8 (0-255)
    │   └── attrs/
    │       └── success       # bool
    ├── demo_1/
    └── ...
```

### Key Characteristics
- **Actions & Joint Positions**: Stored in **radians** with robot-specific limits
- **Images**: Raw RGB uint8 (0-255), typically 480x640x3
- **Episodes**: Each demonstration is a separate group
- **Metadata**: Success flag to filter failed demonstrations

### Joint Position Ranges (Isaac Lab)
```python
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),   # shoulder_pan (degrees, converted from radians)
    (-100.0, 100.0),   # shoulder_lift
    (-100.0, 90.0),    # elbow_flex
    (-95.0, 95.0),     # wrist_flex
    (-160.0, 160.0),   # wrist_roll
    (-10, 100.0),      # gripper (0 = closed, 100 = open)
]
```

**Note**: These ranges represent the physical limits of the SO-101 robot arm. Actions outside these ranges will be clipped or cause errors in simulation.

---

## LeRobot Dataset Format

### Location
`~/.cache/huggingface/lerobot/eternalmay33/pick_place_test/`

### Structure
```
pick_place_test/
├── meta/
│   └── info.json          # Dataset metadata
├── videos/               # Compressed video files (AV1 codec)
│   ├── chunk-000/
│   │   ├── observation.images.front_episode_000000.mp4
│   │   ├── observation.images.third_person_episode_000000.mp4
│   │   └── observation.images.gripper_episode_000000.mp4
│   └── ...
└── data/
    └── chunk-000/
        └── episode_000000.parquet  # Actions and states
```

### Data Schema (Single Arm)
```python
SINGLE_ARM_FEATURES = {
    "action": {
        "shape": (6,),
        "dtype": "float32",
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    },
    "observation.state": {
        "shape": (6,),
        "dtype": "float32",
        # Same joint names
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "video.codec": "av1",
        "video.fps": 30.0,
    },
    "observation.images.third_person": { ... },
    "observation.images.gripper": { ... },
}
```

### Key Characteristics
- **Actions & States**: Normalized to **standard ranges** in degrees
- **Images**: Compressed as video files (AV1 codec), decoded on-the-fly during training
- **Timestamps**: Delta timestamps define temporal relationships between observations and actions
- **Episodes**: Stored as separate video/parquet files indexed by episode number

### Joint Position Ranges (LeRobot Normalized)
```python
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),   # All joints except gripper
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),      # gripper (always positive: 0=closed, 100=open)
]
```

**Why normalize?**
- Standardized ranges help neural networks learn more effectively
- Prevents one joint from dominating gradients due to larger magnitude
- Makes it easier to apply the same architecture to different robots

---

## Critical Transformations

### 1. Joint Position Normalization (Isaac → LeRobot)

**Location**: `leisaac/scripts/convert/isaaclab2lerobot.py:177-185`

```python
def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Convert Isaac Lab joint positions (radians) to LeRobot format (normalized degrees)."""

    # Step 1: Convert radians to degrees
    joint_pos = joint_pos / np.pi * 180

    # Step 2: Normalize each joint from Isaac range to LeRobot range
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]

        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min

        # Linear remapping: [isaac_min, isaac_max] → [lerobot_min, lerobot_max]
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min

    return joint_pos
```

**Example for shoulder_pan (joint 0):**
```
Input (Isaac):  -1.92 radians → -110 degrees
                ↓
Normalize:      (-110 - (-110)) / 220 * 200 + (-100) = -100
                ↓
Output (LeRobot): -100
```

### 2. Image Normalization

**During Conversion** (`isaaclab2lerobot.py:214-215`):
```python
front_images = np.array(demo_group["obs/front"])  # Shape: [T, 480, 640, 3], uint8 (0-255)
# Images are stored as-is in HDF5, then compressed to video during LeRobot conversion
```

**During Training** (`train_act_with_eval.py:141-150`):
```python
# LeRobot automatically:
# 1. Decodes video frames
# 2. Converts to float32 and normalizes to [0, 1]
# 3. Applies any configured image augmentations
```

**During Inference** (`test_act_policy.py:141-150`):
```python
# Manual normalization required:
front_img = policy_obs["front"].cpu().numpy()[0].astype(np.float32) / 255.0
# Shape: [480, 640, 3], float32 (0.0-1.0)

# Then permute to PyTorch format: [3, 480, 640]
observation["observation.images.front"] = torch.from_numpy(front_img).permute(2, 0, 1)
```

### 3. First 5 Frames Skipped

**Location**: `isaaclab2lerobot.py:209`

```python
for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
    # Skip first 5 frames to avoid initialization artifacts
```

**Why?** The first few frames often contain:
- Robot settling into starting position
- Camera exposure adjustment
- Environment initialization glitches

---

## Inference: LeRobot back to Isaac Lab

When deploying a trained policy, you must **reverse** all transformations!

### 1. Observation Preprocessing (Isaac → LeRobot)

**Location**: `test_act_policy.py:95-104`

```python
def preprocess_isaac_to_lerobot(joint_pos: np.ndarray) -> np.ndarray:
    """Convert joint positions from Isaac Lab (radians) to LeRobot format (normalized degrees)."""

    joint_pos = joint_pos / np.pi * 180  # rad to deg

    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min

        joint_pos[i] = (joint_pos[i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min

    return joint_pos
```

### 2. Action Postprocessing (LeRobot → Isaac)

**Location**: `test_act_policy.py:107-116`

```python
def postprocess_lerobot_to_isaac(joint_pos: np.ndarray) -> np.ndarray:
    """Convert joint positions from LeRobot (normalized degrees) back to Isaac Lab (radians)."""

    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min

        # Reverse the normalization
        joint_pos[i] = (joint_pos[i] - lerobot_min) / lerobot_range * isaac_range + isaaclab_min

    # Note: Conversion to radians happens in Isaac Lab's action processing
    # joint_pos = joint_pos / 180 * np.pi  # deg to rad (commented out - handled by environment)

    return joint_pos
```

**Critical**: The degrees-to-radians conversion may be handled by the environment's action processing, depending on the action space configuration!

### 3. Complete Inference Loop

```python
# 1. Get observation from Isaac Lab (radians, uint8 images)
obs_dict, _ = env.reset()

# 2. Convert observation to LeRobot format
joint_pos = obs_dict["policy"]["joint_pos"].cpu().numpy()[0]  # [6], radians
joint_pos_lerobot = preprocess_isaac_to_lerobot(joint_pos)    # [6], normalized degrees

front_img = obs_dict["policy"]["front"].cpu().numpy()[0]      # [480, 640, 3], uint8
front_img = front_img.astype(np.float32) / 255.0              # [480, 640, 3], float32 [0, 1]

observation = {
    "observation.state": torch.from_numpy(joint_pos_lerobot).float().unsqueeze(0).to(device),
    "observation.images.front": torch.from_numpy(front_img).permute(2, 0, 1).unsqueeze(0).to(device),
    # ... other cameras
}

# 3. Get action from policy
with torch.no_grad():
    action_lerobot = policy.select_action(observation)  # [1, 6], normalized degrees

# 4. Convert action back to Isaac Lab format
action_np = action_lerobot.cpu().numpy()[0]                   # [6], normalized degrees
action_isaac = postprocess_lerobot_to_isaac(action_np)        # [6], Isaac degrees

# 5. Execute in environment
action_tensor = torch.from_numpy(action_isaac).float().unsqueeze(0).to(env.device)
obs_dict, reward, terminated, truncated, info = env.step(action_tensor)
```

---

## Common Issues

### Issue 1: Model produces invalid actions in Isaac Lab

**Symptom**: Robot moves erratically, actions seem random, joints go to extreme positions.

**Cause**: Missing or incorrect denormalization of policy outputs.

**Solution**: Verify `postprocess_lerobot_to_isaac()` is called on every action before sending to environment.

```python
# WRONG
action = policy.select_action(observation)
env.step(action)  # ❌ Action is still in LeRobot normalized format!

# CORRECT
action_lerobot = policy.select_action(observation)
action_isaac = postprocess_lerobot_to_isaac(action_lerobot.cpu().numpy()[0])
env.step(torch.from_numpy(action_isaac).unsqueeze(0))  # ✓
```

### Issue 2: Images look wrong to the policy

**Symptom**: Policy trained successfully but performs poorly in simulation.

**Cause**: Mismatch in image preprocessing (uint8 vs float32, value range 0-255 vs 0-1).

**Solution**: Ensure images are normalized to [0, 1] and converted to float32:

```python
# During inference
front_img = obs["policy"]["front"].cpu().numpy()[0]
front_img = front_img.astype(np.float32) / 255.0  # ✓ Normalize to [0, 1]
```

### Issue 3: Joint limits exceeded

**Symptom**: Isaac Lab throws warnings about clipped actions or joint limit violations.

**Cause**: Incorrect min/max ranges in normalization constants.

**Solution**: Verify `ISAACLAB_JOINT_POS_LIMIT_RANGE` matches your robot's actual limits. Check with:

```python
# Print joint limits from environment
print(env.unwrapped.robot.joint_limits)
```

### Issue 4: Actions in radians vs degrees confusion

**Symptom**: Actions are orders of magnitude wrong (e.g., 1.5 rad interpreted as 1.5 deg).

**Cause**: Missing radians ↔ degrees conversion.

**Solution**: Track units carefully:
- **Isaac Lab**: Radians
- **Conversion intermediate**: Degrees
- **LeRobot**: Normalized degrees
- **Inference intermediate**: Isaac degrees
- **Isaac Lab actions**: May be degrees or radians depending on action space config

Check your environment's action space:
```python
print(env.action_space)  # Shows expected units
```

### Issue 5: First episode always fails

**Symptom**: First inference episode has zero reward, subsequent episodes work fine.

**Cause**: Model expecting specific initial state distribution that doesn't match environment reset.

**Solution**:
- Ensure environment resets to similar states as training data
- Consider adding random initial state perturbations during training
- Skip first episode in evaluation metrics

### Issue 6: Training loss is good but validation/inference is poor

**Symptom**: Training loss decreases normally, but policy doesn't generalize.

**Cause**: Overfitting to specific camera positions, lighting, or object placements.

**Solution**:
- Collect more diverse demonstrations
- Add data augmentation (color jitter, random crops)
- Use domain randomization during data collection
- Check if validation set is truly held-out

---

## Data Statistics

From the `pick_place_test` dataset:
- **Total episodes**: 39
- **Total frames**: 8,159
- **Average episode length**: ~209 frames (~7 seconds at 30 fps)
- **Training split**: 31 episodes (80%)
- **Validation split**: 8 episodes (20%)
- **Image resolution**: 480x640x3
- **Action dimensionality**: 6 (5 joints + 1 gripper)
- **FPS**: 30 Hz

---

## References

- **Conversion Script**: `/home/may33/projects/ml_portfolio/robotics/leisaac/scripts/convert/isaaclab2lerobot.py`
- **Training Script**: `/home/may33/projects/robotics/smolvla_in_isaac/simulation_learning/train_act_with_eval.py`
- **Inference Script**: `/home/may33/projects/robotics/smolvla_in_isaac/simulation_learning/test_act_policy.py`
- **Check Dataset**: `/home/may33/projects/robotics/smolvla_in_isaac/simulation_learning/check_dataset.py`
- **LeRobot Version**: v0.3.3 (https://github.com/huggingface/lerobot/tree/v0.3.3)

---

## Quick Reference: Min/Max Ranges

| Joint | Isaac Min | Isaac Max | LeRobot Min | LeRobot Max | Isaac Range | LeRobot Range |
|-------|-----------|-----------|-------------|-------------|-------------|---------------|
| shoulder_pan | -110° | 110° | -100 | 100 | 220° | 200 |
| shoulder_lift | -100° | 100° | -100 | 100 | 200° | 200 |
| elbow_flex | -100° | 90° | -100 | 100 | 190° | 200 |
| wrist_flex | -95° | 95° | -100 | 100 | 190° | 200 |
| wrist_roll | -160° | 160° | -100 | 100 | 320° | 200 |
| gripper | -10° | 100° | 0 | 100 | 110° | 100 |

**Note**: All Isaac values are in degrees (after converting from radians). LeRobot values are normalized.

---

## Summary

The key to successful sim-to-sim deployment:

1. **Understand both formats**: Isaac uses radians with robot-specific limits; LeRobot uses normalized degrees
2. **Always normalize during conversion**: Apply `preprocess_joint_pos()` to all actions/states
3. **Always denormalize during inference**: Apply `postprocess_lerobot_to_isaac()` to policy outputs
4. **Handle images consistently**: uint8 [0-255] in Isaac → float32 [0-1] for policy
5. **Track your units**: Radians vs degrees confusion is the #1 source of bugs
6. **Test transformations**: Verify round-trip conversion (Isaac → LeRobot → Isaac) preserves values

When in doubt, add debug prints and check value ranges at each step!
