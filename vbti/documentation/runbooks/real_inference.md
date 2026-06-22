# Runbook: Real Inference

Use for manual policy debugging, not quantitative comparison.

## 1. Preflight

Run `preflight.md`.

## 2. Pick Checkpoint

Example explicit checkpoint:

```text
/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model
```

Verify pre/postprocessor files are present.

## 3. Preview Cameras

```bash
python -m vbti.logic.inference.run_real_inference preview
```

## 4. Run RGB Policy

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/absolute/path/to/pretrained_model \
  --port=/dev/ttyACM1 \
  --cameras=realsense \
  --task="Pick up the duck and place it in the red cup" \
  --action_horizon=10 \
  --max_relative_target=10.0 \
  --fps=30 \
  --max_steps=500
```

## 5. Run Depth Policy

Only if trained with `gripper_depth`:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/absolute/path/to/pretrained_model \
  --port=/dev/ttyACM1 \
  --cameras=opencv_depth \
  --depth=true \
  --task="Pick up the duck and place it in the red cup"
```

## 6. Run Detection-State Policy

Only if trained with 22D detection state:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/absolute/path/to/pretrained_model \
  --port=/dev/ttyACM1 \
  --detection=true \
  --task="Pick up the duck and place it in the red cup"
```

## 7. After Run

Move to rest if needed:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1
```

Record notes:

- checkpoint;
- prompt;
- camera preset;
- observed behavior;
- failures;
- whether this needs formal protocol evaluation.

## Stop Conditions

- Robot moves unexpectedly or too aggressively.
- Camera freezes.
- Depth stream missing for depth model.
- Detection overlay/state missing for detection model.
- Calibration seems wrong.
