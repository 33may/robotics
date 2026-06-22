---
name: Remote Training Machine (5090)
description: Remote 5090 machine setup, env quirks, and remote training workflow via vbti/logic/train/remote.py
type: reference
originSessionId: 2f1e08d0-4ca8-4d5a-8dd5-1c4ea7f1d98d
---
## Remote Machine

- **Host**: `vbti@10.11.100.156` (DHCP-assigned, changes — was .151 until 2026-05-11, .101.240 until 2026-05-18; .100.156 confirmed reachable 2026-05-26 and matches canonical ED25519 fingerprint; .101.240 was also visible with the same fingerprint)
- **Hostname**: `vbti-MS-7E66`
- **Password**: `vbti25robot`
- **Network**: /23 subnet (10.11.100.0/23), so .100.x and .101.x are same L2 segment
- **GPU**: RTX 5090 (Blackwell, sm_120)
- **Python env**: `/home/vbti/anton/env` (uv managed, NOT conda)
- **PyTorch**: nightly 2.12.0.dev+cu128 (sm_120 requires nightly)
- **CUDA toolkit**: 12.8 (`/usr/local/cuda-12.8`)
- **Data dir**: `/home/vbti/anton/data`
- **Experiments dir**: `/home/vbti/anton/experiments`

## Remote Training CLI

`python -m vbti.logic.train.remote {train|status|logs|pull|push_data}`

Config in `vbti/remote.yaml`. Reads active experiment/version from `state.json`.

## Key Env Quirks (2026-03-26)

- `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64` required in train scripts
- torchcodec must match nightly PyTorch (install from nightly index)
- Isaac Sim is installed on this machine but bundles CUDA 11.8 — don't rely on it
- Policy type: local lerobot uses `internvla`, standard lerobot uses `smolvla` — checkpoint configs must be patched when pushing to remote and reverted after
- `num_workers: 0` causes GPU starvation on 5090 — use 8+
- Dataset `eternalmay33/08-merged` is local-only (not on HF Hub) — all data must be rsynced, don't set HF_HUB_OFFLINE=1 as it blocks base model downloads
- Remote lerobot requires `--policy.repo_id` arg (validation check) — added to `_build_lerobot_command` using job_name as placeholder
