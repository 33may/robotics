---
name: resolve_dataset_path shared function
description: Canonical dataset path resolver in vbti/logic/dataset/__init__.py — resolves repo_id, filesystem path, or lerobot cache
type: project
originSessionId: 1a4916d7-6577-4f75-b1aa-dc5c75803a7c
---
## resolve_dataset_path

**Location**: `vbti/logic/dataset/__init__.py`

Shared function that resolves dataset references to filesystem paths. Resolution order:
1. If `root` kwarg given, use it directly
2. If path exists on disk, use it
3. Try as repo_id in `~/.cache/huggingface/lerobot/` (e.g. `enea-c/VBTI-Align-v2A`)
4. Raise with suggestions

**Why:** Previously duplicated in check_utils.py, trim_utils.py, and missing from replay_utils.py. Now centralized.

**How to apply:** All dataset scripts use lazy import `from vbti.logic.dataset import resolve_dataset_path` inside their resolver functions. The `vbti` package is importable because a `.pth` file adds the project root to sys.path in the lerobot conda env.

### .pth file
`/home/may33/miniconda3/envs/lerobot/lib/python3.12/site-packages/vbti.pth` contains:
```
/home/may33/projects/ml_portfolio/robotics
```
This makes `vbti` importable from anywhere when using the lerobot env.
