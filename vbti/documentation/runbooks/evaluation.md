# Runbook: Protocol Evaluation

Use this for quantitative real-robot comparison.

## 1. Preflight

Run `preflight.md`.

## 2. Choose Protocol

Common protocols:

- `checkpoint_sweep`
- `dual_cup_30`
- `dual_cup_60`
- `id_scale_60`

Render/verify protocol if needed:

```bash
python -m vbti.logic.inference.protocols.protocols render checkpoint_sweep
python -m vbti.logic.inference.protocols.protocols verify checkpoint_sweep
```

## 3. Choose Checkpoint

Examples:

```text
v020-150k
v17-20k
/absolute/path/to/pretrained_model
```

## 4. Run RGB Evaluation

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=checkpoint_sweep \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --port=/dev/ttyACM1 \
  --cameras=realsense \
  --action_horizon=10 \
  --max_steps=600 \
  --record=True
```

## 5. Run Depth Evaluation

Only if checkpoint expects depth:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v18-20k \
  --protocol=dual_cup_30 \
  --experiment=duck_cup_smolvla \
  --version=v018 \
  --port=/dev/ttyACM1 \
  --cameras=opencv_depth \
  --depth=true \
  --action_horizon=10 \
  --max_steps=10000 \
  --record=True
```

## 6. Resume Interrupted Evaluation

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=checkpoint_sweep \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --record=True \
  --resume=True
```

## 7. Operator Controls

- `SPACE`: start trial.
- `S`: mark success.
- `F`: mark failure.
- `V`: save video.
- `Q` / Esc: quit.

## 8. After Evaluation

Session output is saved under:

```text
experiments/duck_cup_smolvla/vNNN/eval_sessions/chkpt_step_<step>_ah_<horizon>_pr_<protocol>_<timestamp>/
```

Immediately inspect:

```bash
python -m vbti.logic.inference.eval_helpers info latest
```

## Validity Rules

- Do not compare different protocols as the same metric.
- Include numerator/denominator, not only percentage.
- Record camera preset, depth/detection flags, and action horizon.
- If hardware changed mid-session, mark the session as compromised.
