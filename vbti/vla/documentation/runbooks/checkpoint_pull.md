# Runbook: Pull Checkpoints And Prepare For Evaluation

## 1. Identify Version And Run

Example:

```text
experiment: duck_cup_smolvla
version: v020
run_name: lerobot_output_r12
checkpoint: step_150000
```

Check receipt:

Open:

```text
experiments/duck_cup_smolvla/v020/remote_session.json
```

Verify `run_name`, `remote_run_dir`, `remote_ckpt_dir`, and host.

## 2. Pull Checkpoint

```bash
python -m vbti.logic.train.remote pull \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r12 \
  --checkpoint=step_150000 \
  --pretrained_only=true
```

If pulling all:

```bash
python -m vbti.logic.train.remote pull \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r12
```

## 3. Verify Checkpoint Files

Expected pretrained model folder:

```text
experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model/
```

Expected files:

```text
config.json
model.safetensors
preprocessor.json
postprocessor.json
train_config.json
```

Naming may differ by LeRobot version, but both preprocessor and postprocessor must be present.

## 4. Pick Evaluation Command

For a shorthand checkpoint:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=checkpoint_sweep \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --record=True
```

For explicit path:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=/absolute/path/to/pretrained_model \
  --protocol=checkpoint_sweep \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --record=True
```

## 5. Record In Notes

Update version notes with:

- checkpoint path;
- pull command;
- intended protocol;
- expected comparison.
