# Runbook: Create A New Experiment Version

Use when creating `experiments/duck_cup_smolvla/vNNN/` for a new training run.

## 1. Pick Version And Hypothesis

Define:

```text
version: vNNN
hypothesis: what is different and why
dataset: repo/path
expected comparison: which earlier version/protocol
```

Examples:

- “v030 tests RGB-only v020 data without depth.”
- “v031 tests 25% data with UVA aux weight 0.003.”
- “v032 tests unfrozen vision on a new targeted failure-region dataset.”

## 2. Copy A Close Config

```bash
cd /home/may33/projects/ml_portfolio/robotics/vbti
mkdir -p experiments/duck_cup_smolvla/vNNN
cp experiments/duck_cup_smolvla/v020/config.yaml experiments/duck_cup_smolvla/vNNN/config.yaml
```

Choose the source config that is closest to the intended run:

- v020-style full strong baseline;
- v021-v024-style data-efficiency config;
- v025-v029-style UVA config;
- v017/v018-style RGB/depth A/B config.

## 3. Edit Config

Edit:

```text
experiments/duck_cup_smolvla/vNNN/config.yaml
```

Check at minimum:

- `model.type`;
- `model.pretrained`;
- `model.freeze_vision_encoder`;
- `model.aux_weight` if UVA;
- `dataset.sources[0].repo_id`;
- `dataset.sources[0].root` if needed;
- `dataset.sources[0].episodes` if slicing by config;
- `dataset.cameras.names`;
- `training.steps` or `training.epochs`;
- `training.batch_size`;
- `training.lr`;
- `training.vision_lr_scale`;
- `logging.save_freq`;
- W&B settings if enabled.

## 4. Inspect Config

```bash
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/vNNN/config.yaml
```

## 5. Verify Dataset Schema Against Config

```bash
python -m vbti.logic.dataset.check_utils report <dataset_repo_or_path>
python -m vbti.logic.dataset.check_utils cameras <dataset_repo_or_path>
```

Ensure camera names in config match dataset features.

## 6. Dry-Run Training Command

```bash
python -m vbti.logic.train.engine train-lerobot \
  --experiment=duck_cup_smolvla \
  --version=vNNN \
  --dry_run
```

Inspect generated `lerobot-train` command for:

- correct dataset;
- correct pretrained path;
- correct steps;
- correct batch size;
- correct camera rename map;
- `--use_policy_training_preset=false`;
- W&B/job name.

## 7. Add Notes

Create:

```text
experiments/duck_cup_smolvla/vNNN/notes.md
```

Template:

```markdown
# vNNN Notes

## Hypothesis

## Dataset

## Config Differences

## Expected Comparison

## Training Command

## Evaluation Plan

## Results
```

## 8. Ready For Training

If dry run and dataset check pass, continue with:

- `remote_training.md`; or
- `training_chains.md` for sweeps.
