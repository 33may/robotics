# Runbook: Local Training Or Dry Run

Use local training mainly for smoke tests, backend development, or LeRobot command generation. Main large SmolVLA runs usually use remote training.

## 1. Inspect Active Status

```bash
python -m vbti.logic.train.engine status
```

## 2. Show Config

```bash
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/vNNN/config.yaml
```

## 3. Generate LeRobot Command

```bash
python -m vbti.logic.train.engine train-lerobot \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --dry_run
```

Use this before remote training to validate command translation.

## 4. Local Engine Training

```bash
python -m vbti.logic.train.engine train \
  --experiment=duck_cup_smolvla \
  --version=vNNN
```

Resume:

```bash
python -m vbti.logic.train.engine train --resume
python -m vbti.logic.train.engine train --resume --reset_lr
```

## 5. When Not To Use Local Engine

Do not use local engine to validate remote-only behavior. Remote training uses `lerobot-train` and the patched LeRobot fork.

If testing dataset transforms for remote training, bake them into the dataset and dry-run `train-lerobot` instead.

## 6. Outputs

Local engine writes under the experiment version, typically:

```text
metrics/training_log.jsonl
metrics/summary.json
checkpoints/
run.md
```

Remote LeRobot output uses `lerobot_output_rX/` instead.
