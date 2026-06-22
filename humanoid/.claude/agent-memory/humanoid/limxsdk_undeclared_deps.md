---
name: limxsdk-undeclared-deps
description: limxsdk 4.0.1 wheel METADATA omits runtime deps that submodules import (click, possibly others); add to limx env explicitly
metadata:
  type: project
---

The `limxsdk-4.0.1-py3-none-any.whl` METADATA declares `onnxruntime, pyyaml, numpy<1.26.4,>1.21.0, pygame, scipy, pandas, mujoco>3.2.2` — but several SDK submodules `import` packages that aren't in that list. Pip install succeeds; runtime fails with `ModuleNotFoundError` only when the relevant code path executes.

Known missing deps:

| Module | Imports | Symptom |
|---|---|---|
| `limxsdk/ability/cli.py` | `click` | `python -m limxsdk.ability.cli load/switch` → `ModuleNotFoundError: No module named 'click'`; deploy `main.py` then SIGSEGVs as a downstream effect. |

**Why:** Hit 2026-06-19 on first launcher run. Click is now pinned in `humanoid/requirements/limx.txt` under an "Undeclared SDK runtime deps" section.

**How to apply:** When a new `limxsdk.<module>` first runs and explodes with `ModuleNotFoundError`, add the missing package to `requirements/limx.txt` (same section) and update this list. Do not file upstream until we have a few examples to send at once.

Related: [[vendor-humanoid-mujoco-sim]], [[vendor-humanoid-rl-deploy-python]].
