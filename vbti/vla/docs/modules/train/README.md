# `logic/train`

## Purpose

`logic/train` manages experiment configs, local/custom training, remote `lerobot-train`, sequential sweeps, and experiment folder state.

## Files

| File | Purpose |
|---|---|
| `config_utils.py` | Dataclass config schema, default configs, schema/show/create CLI. |
| `engine.py` | Local engine, version resolution, local train loop, LeRobot command builder. |
| `remote.py` | SSH/rsync/tmux remote training wrapper. |
| `chain.py` | Local polling orchestrator for sequential remote runs. |
| `chain_remote.py` | Generates one remote script for sequential chains. |
| `experiment_utils.py` | Experiment folder/version/status/checkpoint helpers. |
| `monitor.py` | Status/log monitoring helpers. |
| `backends/base.py` | Backend interface. |
| `backends/smolvla.py` | Local SmolVLA backend path. |
| `backends/groot.py` | GR00T backend path. |

## Docs

- `config_schema.md` - YAML fields and defaults.
- `local_engine.md` - local engine and LeRobot dry-run usage.
- `remote_training.md` - remote training workflow.
- `chains.md` - sequential run workflows.
- `experiments.md` - duck-cup version history and artifact layout.

## Critical Rule

Remote training launches `lerobot-train`. It does not execute local `SmolVLABackend` code. Dataset transforms must be baked, or the LeRobot fork must be patched.
