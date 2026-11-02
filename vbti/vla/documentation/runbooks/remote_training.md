# Runbook: Remote Training

Use for main SmolVLA training runs.

## 1. Create/Verify Version

Run `create_experiment_version.md` first.

Minimum checks:

```bash
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/vNNN/config.yaml
python -m vbti.logic.train.engine train-lerobot --experiment=duck_cup_smolvla --version=vNNN --dry_run
```

## 2. Ensure Dataset Is Available

If dataset is local and should be pushed:

```bash
python -m vbti.logic.train.remote push-data --repo_id=<dataset_repo_id>
```

`remote.py train` can also auto-sync when it detects missing/incomplete data.

## 3. Launch Training

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --run_name=lerobot_output_r1
```

Non-streaming launch for autonomous/background workflows:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --run_name=lerobot_output_r1 \
  --stream=false
```

## 4. Resume From Checkpoint

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --run_name=lerobot_output_r2 \
  --resume_from=lerobot_output_r1/checkpoints/030000
```

## 5. Monitor

```bash
python -m vbti.logic.train.remote status --experiment=duck_cup_smolvla --version=vNNN
python -m vbti.logic.train.remote logs --experiment=duck_cup_smolvla --version=vNNN --lines=100
```

If these do not show logs, inspect `remote_session.json` and the remote version directory; current code may look at a different log path than the launcher writes.

## 6. Pull Outputs

All outputs:

```bash
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=vNNN
```

Specific checkpoint:

```bash
python -m vbti.logic.train.remote pull \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --checkpoint=step_080000
```

Pretrained model only:

```bash
python -m vbti.logic.train.remote pull \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --checkpoint=step_080000 \
  --pretrained_only=true
```

## 7. Record Training Notes

Update:

```text
experiments/duck_cup_smolvla/vNNN/notes.md
```

Include:

- command;
- run name;
- tmux session;
- dataset;
- start/end time;
- checkpoint steps;
- W&B URL if used;
- errors or restarts.

## 8. Next Step

Run `checkpoint_pull.md`, then `evaluation.md`.

## Pitfalls

- Remote uses `lerobot-train`, not local backend classes.
- Do not expose remote password from `remote.yaml`.
- Dataset path/root must exist on remote.
- Patched LeRobot fork is required for project-specific features.
- If using UVA or custom features, confirm baked dataset exists on remote.
