# Common Utilities

Shared utilities for Isaac Lab <-> LeRobot integration.

## Modules

### `constants.py`
Centralized constants for the entire project:
- Dataset paths and configuration
- Joint position ranges (Isaac Lab vs LeRobot)
- Camera configuration
- Training/inference defaults

### `transformations.py`
Data transformation functions:
- `load_normalization_stats()` - Load mean/std from dataset
- `preprocess_isaac_to_lerobot()` - Convert observations: Isaac → Policy
- `postprocess_lerobot_to_isaac()` - Convert actions: Policy → Isaac
- `prepare_observation()` - Format full observation dict for policy
- `preprocess_joint_pos_batch()` - Batch conversion for dataset creation

### `isaac_utils.py`
Isaac Lab environment setup:
- `setup_isaac_environment()` - Configure environment variables
- `create_app_launcher()` - Initialize Isaac Sim AppLauncher
- `create_lift_cube_environment()` - Create configured environment
- `print_environment_info()` - Debug info about environment
- `safe_close_environment()` - Cleanup resources

## Usage Example

```python
from common import (
    # Setup
    setup_isaac_environment,
    create_app_launcher,
    create_lift_cube_environment,
    # Transformations
    load_normalization_stats,
    prepare_observation,
    postprocess_lerobot_to_isaac,
    # Constants
    DEFAULT_DATASET_REPO_ID,
)

# 1. Setup Isaac environment
setup_isaac_environment()

# 2. Load normalization stats
stats = load_normalization_stats(DEFAULT_DATASET_REPO_ID)

# 3. Create Isaac Lab app
app_launcher, simulation_app = create_app_launcher(headless=False)

# 4. Create environment
env = create_lift_cube_environment(num_envs=1)

# 5. Run inference loop
obs_dict, _ = env.reset()

# Prepare observation
observation = prepare_observation(
    obs_dict, device,
    stats['obs_state_mean'],
    stats['obs_state_std']
)

# Get action from policy
action = policy.select_action(observation)

# Convert to Isaac format
action_isaac = postprocess_lerobot_to_isaac(
    action.cpu().numpy()[0],
    stats['action_mean'],
    stats['action_std']
)
```

## Benefits

✅ **No code duplication** - Functions used by both ACT and SmolVLA
✅ **Single source of truth** - Constants defined once
✅ **Easy to maintain** - Update in one place
✅ **Type hints** - Better IDE support
✅ **Documentation** - Docstrings for all functions

## Migration Guide

### Before (duplicated code):
```python
# In test_act_policy.py
ISAACLAB_JOINT_POS_LIMIT_RANGE = [(-110.0, 110.0), ...]
LEROBOT_JOINT_POS_LIMIT_RANGE = [(-100, 100), ...]

def preprocess_isaac_to_lerobot(joint_pos):
    # 30 lines of code...
```

### After (using common):
```python
from common import (
    ISAACLAB_JOINT_POS_LIMIT_RANGE,
    LEROBOT_JOINT_POS_LIMIT_RANGE,
    preprocess_isaac_to_lerobot,
)
```

**Reduction**: 70% less code, 100% less duplication!
