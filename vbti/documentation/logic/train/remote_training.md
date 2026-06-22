# Remote Training

## Purpose

`logic/train/remote.py` launches training on the remote machine through SSH, rsync, tmux, and `lerobot-train`.

## Remote Config

Config file:

```text
remote.yaml
```

It contains host, user, password, Python/env paths, remote data root, and remote experiment root. It includes plaintext secrets; do not copy them into docs or commits.

## Commands

Push dataset:

```bash
python -m vbti.logic.train.remote push-data --repo_id=eternalmay33/duck_cup_v020_all
```

Launch one version:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r1
```

Dry run:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r1 \
  --dry_run
```

Resume:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r2 \
  --resume_from=lerobot_output_r1/checkpoints/030000
```

Status/logs:

```bash
python -m vbti.logic.train.remote status --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.remote logs --experiment=duck_cup_smolvla --version=v020 --lines=100
```

Pull:

```bash
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=step_080000
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=step_080000 --pretrained_only=true
```

## What `train` Does

1. Resolve experiment/version/config.
2. Build `lerobot-train` command using `engine._build_lerobot_command()`.
3. Rewrite dataset root to remote path.
4. Sync dataset if missing/incomplete.
5. Sync local pretrained model path if needed.
6. Sync resume checkpoint if needed.
7. Write remote training script.
8. Launch tmux session.
9. Save local `remote_session.json` receipt.

## Remote Receipt

Saved under:

```text
experiments/duck_cup_smolvla/vNNN/remote_session.json
```

Typical fields:

- `run_name`;
- `tmux_session`;
- `remote_version_dir`;
- `remote_run_dir`;
- `remote_ckpt_dir`;
- `remote_data_root`;
- `job_name`;
- `repo_id`;
- `host`.

## Remote Environment Assumptions

- remote Python/env activation under `/home/vbti/anton/env`;
- remote data root under `/home/vbti/anton/data`;
- remote experiments under `/home/vbti/anton/experiments`;
- patched editable LeRobot fork under `/home/vbti/anton/lerobot`;
- CUDA 12.8 library path;
- optional `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Known Pitfalls

- `status()`/`logs()` may look at a different log path than newer launch scripts write. If logs look empty, inspect `remote_session.json` and remote version dir.
- SSH/rsync disable strict host key checking.
- Remote path bypasses local backend classes.
- Remote-only baked datasets, such as UVA datasets, may not exist locally.
