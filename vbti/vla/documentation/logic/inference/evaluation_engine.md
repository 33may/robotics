# Protocol Evaluation Engine

## Command

Preferred module form:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=dual_cup_60 \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --port=/dev/ttyACM1 \
  --action_horizon=10 \
  --max_steps=10000 \
  --fps=30 \
  --record=True
```

Important: current Fire binding exposes `run` arguments directly. Some older docs show `eval_engine.py run`; verify before using that form.

## Defaults

| Arg | Default |
|---|---|
| `protocol` | `checkpoint_sweep` |
| `port` | `/dev/ttyACM1` |
| `cameras` | `realsense` |
| `action_horizon` | `10` |
| `max_steps` | `600` |
| `fps` | `30` |
| `max_relative_target` | `10.0` |
| `record` | `False` |
| `resume` | `False` |
| `depth` | `False` |
| `detection` | `False` |

## Operator Flow

1. Engine shows live placement overlay.
2. Operator places duck/cup(s).
3. Press `SPACE` to start trial.
4. Policy runs live.
5. Press `S` to mark success or `F` to mark failure.
6. Press `V` to save video if needed.
7. Robot returns to rest.
8. Next trial starts.

Keys:

- `SPACE`: start trial.
- `S`: success.
- `F`: failure.
- `V`: save video.
- `Q` / Esc: quit early.

## Output

Session folder:

```text
experiments/duck_cup_smolvla/vNNN/eval_sessions/chkpt_step_<step>_ah_<horizon>_pr_<protocol>_<timestamp>/
```

Contents:

```text
session.json
session.md
videos/trial_XX_success.mp4
videos/trial_XX_failure.mp4
```

## Resume

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=dual_cup_60 \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --record=True \
  --resume=True
```

Use when a session crashed/interrupted and should continue from saved trial state.

## Checkpoint Shorthand

Examples:

```text
v020-150k
v17-20k
step_080000
```

Always verify resolved checkpoint path before long evaluation.

## Depth Evaluation

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v18-20k \
  --protocol=dual_cup_30 \
  --cameras=opencv_depth \
  --depth=true \
  --record=True
```

Depth requires RealSense-backed gripper camera.
