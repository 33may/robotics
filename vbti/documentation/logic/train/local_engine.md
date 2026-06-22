# Local Engine And LeRobot Command Builder

## Status

The local engine exists and can train through backend classes, but the main large SmolVLA runs use the remote `lerobot-train` path.

## Commands

```bash
python -m vbti.logic.train.engine status
python -m vbti.logic.train.engine train --config=/path/to/config.yaml --experiment=duck_cup_smolvla
python -m vbti.logic.train.engine train --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.engine train --resume
python -m vbti.logic.train.engine train --resume --reset_lr
```

## LeRobot Dry Run

```bash
python -m vbti.logic.train.engine train-lerobot --config=/path/to/config.yaml --experiment=duck_cup_smolvla --dry_run
python -m vbti.logic.train.engine train-lerobot --experiment=duck_cup_smolvla --version=v020 --dry_run
python -m vbti.logic.train.engine train-lerobot
```

This prints/builds the `lerobot-train` command used by remote training.

## Generated LeRobot Flags

Important mappings include:

- `--policy.path=<model.pretrained>`
- `--dataset.repo_id=<repo_id>`
- `--dataset.root=<root>`
- `--dataset.episodes=<json>`
- `--dataset.image_transforms.enable=true`
- `--batch_size=<training.batch_size>`
- `--steps=<training.steps or derived epochs>`
- `--num_workers=<training.num_workers>`
- `--use_policy_training_preset=false`
- optimizer/scheduler flags;
- SmolVLA policy flags;
- camera `rename_map`;
- W&B flags.

## Critical Flag

`--use_policy_training_preset=false` is required so LeRobot does not override project optimizer/scheduler settings with policy defaults.

## Epoch-Derived Steps

If config sets:

```yaml
training:
  epochs: 12
```

the engine computes steps approximately as:

```text
steps = epochs * num_frames / batch_size
```

This matters for data-efficiency sweeps where dataset size changes.

## When To Use Local Engine

Use local engine for:

- backend development;
- small smoke tests;
- dry-running command generation;
- understanding config translation.

Do not assume backend edits affect remote training unless remote code path is changed.
