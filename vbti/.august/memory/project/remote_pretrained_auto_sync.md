---
name: Remote training auto-syncs local pretrained models
description: remote.py auto-detects local pretrained paths in config, rsyncs to remote, rewrites --policy.path
type: project
---

## Feature Added (2026-03-30)

`vbti/logic/train/remote.py` now auto-detects when `config.model.pretrained` is a local directory path (not a HF repo ID). On `train`:

1. Detects local path exists via `Path(cfg.model.pretrained).exists()`
2. Rewrites `--policy.path` to `{remote_version_dir}/pretrained_base`
3. Rsyncs the local checkpoint dir to that remote path before launching

**Priority:** `resume_from` flag > local pretrained path > HF repo ID (passthrough)

**Why:** Previously had to use `--resume_from` with symlinks to fine-tune from a checkpoint in a different version. Now just set `pretrained:` to the local checkpoint path in config.yaml.

**How to apply:** In config.yaml, set `model.pretrained` to an absolute local path like:
```
pretrained: /home/may33/.../v008/lerobot_output_r1/checkpoints/025000/pretrained_model
```
`remote.py train` handles the rest automatically.
