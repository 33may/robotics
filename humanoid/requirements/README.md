# Python environments

This repo deliberately uses multiple conda environments — different parts of the
humanoid stack have incompatible Python/ABI constraints. Each environment has
its own `requirements/<env>.txt` here.

| Env | Python | File | Purpose |
|---|---|---|---|
| `hum` | 3.12 | [hum.txt](hum.txt) | Oli corpus extraction, MCP tooling, repo-side docs work |
| `limx` | 3.8 | [limx.txt](limx.txt) | Runtime for any process importing `limxsdk` (sim + RL deploy) |

Adjacent envs (defined outside this repo but referenced for context):

| Env | Python | Where it lives | Purpose |
|---|---|---|---|
| `isaac` | 3.11 | Isaac Sim install | Isaac Sim / IsaacLab development |
| `lerobot` | 3.12 | `vbti/` project | LeRobot training, datasets, SmolVLA |
| `groot` | 3.10 | `vbti/` project | NVIDIA GR00T N1.6 training |

## Why multiple envs

Several constraints are mutually incompatible inside one Python:

- **`limxsdk` wheel is hard-locked to Python 3.8** (embeds `libpython3.8.so.1.0`,
  `_robot.so` is cp38). Cannot install on 3.9+.
- **Isaac Sim ships its own Python 3.11**. Cannot be swapped.
- **LeRobot tracks recent torch** (2.7+), which dropped 3.8 a long time ago.

So each runtime gets its own env, and they communicate via files on disk
(ONNX policies, LeRobot datasets) or over the wire (`limxsdk` MROS bus).
See [`docs/vendor/humanoid-rl-deploy-python.md`](../docs/vendor/humanoid-rl-deploy-python.md)
for the sim ↔ policy ↔ real wire contract.

## Creating an env

`hum`:

```bash
conda create -n hum -c conda-forge python=3.12 -y
conda activate hum
pip install -r requirements/hum.txt
```

`limx` (read the header of `limx.txt` for ABI notes before bumping pins):

```bash
conda create -n limx -c conda-forge python=3.8.18 -y
conda activate limx
pip install vendor/humanoid-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-4.0.1-py3-none-any.whl
pip install -r requirements/limx.txt
```

Smoke test for `limx`:

```bash
python -c "import limxsdk, mujoco, onnxruntime, scipy.spatial.transform, numpy; print('ok')"
```
