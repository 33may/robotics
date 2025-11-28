# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Isaac Lab extension project for developing a robotics task called "Collect Cubes". It is built as an isolated extension outside the core Isaac Lab repository, allowing independent development while leveraging Isaac Lab and Isaac Sim infrastructure.

**Framework Stack:**
- Isaac Sim (5.0.0) - NVIDIA's robotics simulation platform
- Isaac Lab - Python framework for robot learning built on Isaac Sim
- skrl - Reinforcement learning library for training agents
- Gymnasium - Standard RL environment interface

## Environment Setup

The project depends on Isaac Lab and Isaac Sim being installed separately. Environment paths are configured in `.env`:

```bash
PYTHONPATH=/home/may33/isaac/IsaacLab/source:/home/may33/isaac/isaac-sim-standalone-5.0.0-linux-x86_64/...
```

Before running scripts, ensure Isaac Lab is properly installed and the Python environment has access to it.

## Installation

Install this extension in editable mode:

```bash
python -m pip install -e source/collect_cubes
```

Or if Isaac Lab is not in a venv/conda:

```bash
/path/to/isaaclab.sh -p -m pip install -e source/collect_cubes
```

## Common Commands

### List Available Environments

```bash
python scripts/list_envs.py
```

This lists all registered tasks with the "Template-" prefix (currently: `Template-Collect-Cubes-Direct-v0`).

### Test Environment with Dummy Agents

Zero-action agent (outputs zeros):
```bash
python scripts/zero_agent.py --task=Template-Collect-Cubes-Direct-v0
```

Random-action agent:
```bash
python scripts/random_agent.py --task=Template-Collect-Cubes-Direct-v0
```

### Train RL Agent

Train using skrl PPO agent:
```bash
python scripts/skrl/train.py --task=Template-Collect-Cubes-Direct-v0
```

With custom parameters:
```bash
python scripts/skrl/train.py --task=Template-Collect-Cubes-Direct-v0 --num_envs=64 --max_iterations=1000 --seed=42
```

Multi-GPU training:
```bash
python scripts/skrl/train.py --task=Template-Collect-Cubes-Direct-v0 --distributed
```

### Play Trained Policy

```bash
python scripts/skrl/play.py --task=Template-Collect-Cubes-Direct-v0 --checkpoint=/path/to/checkpoint
```

### Code Formatting

```bash
pre-commit run --all-files
```

## Architecture

### Directory Structure

```
source/collect_cubes/collect_cubes/
├── tasks/
│   └── direct/
│       └── collect_cubes/
│           ├── __init__.py              # Gym environment registration
│           ├── collect_cubes_env.py     # DirectRLEnv implementation
│           ├── collect_cubes_env_cfg.py # Environment configuration
│           └── agents/
│               └── skrl_ppo_cfg.yaml    # PPO hyperparameters
├── __init__.py
└── ui_extension_example.py              # Optional Omniverse UI extension
```

### Environment Registration

Environments are registered in `tasks/direct/collect_cubes/__init__.py` using Gymnasium:

```python
gym.register(
    id="Template-Collect-Cubes-Direct-v0",
    entry_point="collect_cubes.tasks.direct.collect_cubes.collect_cubes_env:CollectCubesEnv",
    kwargs={
        "env_cfg_entry_point": "...:CollectCubesEnvCfg",
        "skrl_cfg_entry_point": "agents:skrl_ppo_cfg.yaml",
    },
)
```

### Environment Implementation

The task follows Isaac Lab's `DirectRLEnv` pattern:

**`collect_cubes_env_cfg.py`**: Defines environment configuration
- Simulation parameters (dt, decimation, episode length)
- Action/observation space dimensions
- Robot configuration (currently Franka Panda)
- Scene setup (number of envs, spacing)

**`collect_cubes_env.py`**: Implements the RL environment
- `_setup_scene()`: Creates robot, ground plane, lighting, clones environments
- `_pre_physics_step()`: Receives actions from policy
- `_apply_action()`: Applies actions to robot (currently zero effort)
- `_get_observations()`: Returns state observations to policy
- `_get_rewards()`: Computes reward signal
- `_get_dones()`: Determines episode termination
- `_reset_idx()`: Resets specific environments

**Current Status**: The environment is a minimal stub that spawns a Franka Panda arm with zero control. Observations return zeros, rewards are constant (1.0), and episodes never terminate early. This is a starting point for implementing the actual cube collection task.

### Robot Configuration

The environment uses `FRANKA_PANDA_CFG` from `isaaclab_assets.robots.franka`, configured with:
- Fixed base
- 7 DOF action space (controllable joints)
- 14-dimensional observation space (joint positions + velocities)

### RL Training Configuration

Training uses skrl library with PPO. Configuration in `agents/skrl_ppo_cfg.yaml`:
- Policy/Value networks: 2 hidden layers, 32 units each, ELU activation
- PPO hyperparameters: rollouts=32, learning_epochs=8, mini_batches=8
- Learning rate: 5e-4 with KL adaptive scheduling
- Default timesteps: 4800 (for testing; increase for real training)

The training script (`scripts/skrl/train.py`) uses Hydra for configuration management, allowing command-line overrides and parameter sweeps.

## Development Notes

### Task Naming Convention

Task IDs should start with "Template-" to be discovered by `list_envs.py`. Update the search pattern in that script if changing the naming convention.

### Isaac Lab Integration

This project imports Isaac Lab as an external dependency. Key imports:
- `isaaclab.envs.DirectRLEnv` - Base RL environment class
- `isaaclab.assets.Articulation` - Robot articulation wrapper
- `isaaclab.sim` - Simulation utilities
- `isaaclab_tasks` - Must be imported to register Isaac Lab tasks
- `isaaclab_rl.skrl` - Wrapper for skrl integration

### Omniverse Extension (Optional)

The project can be loaded as an Omniverse extension:
1. Add `source/` directory to Extension Manager search paths
2. Enable "collect_cubes" extension under Third Party
3. Custom UI defined in `ui_extension_example.py` will load

### Python Requirements

- Python >= 3.10
- Isaac Sim 4.5.0 or 5.0.0 compatibility
- Dependencies: psutil (minimal; Isaac Lab provides most dependencies)
