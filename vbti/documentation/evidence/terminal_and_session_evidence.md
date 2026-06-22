# Terminal And Session Evidence Summary

This appendix summarizes repo-relevant commands found in zsh history and Claude session logs. It intentionally does not copy private history wholesale.

## Training Commands Seen

Remote training was repeatedly launched through:

```bash
python -m vbti.logic.train.remote train --experiment=duck_cup_smolvla --version=v017 --run_name=lerobot_output_r1
python -m vbti.logic.train.remote train --experiment=duck_cup_smolvla --version=v018 --run_name=lerobot_output_r1
python -m vbti.logic.train.remote train --run_name=lerobot_output_r1 --dry_run
python -m vbti.logic.train.remote status --experiment=duck_cup_smolvla --version=v018
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=step_080000
```

Sequential UVA/data-efficiency work used:

```bash
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name lerobot_output_r2
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name=lerobot_output_r4_aux0003 --wait_for_session uva_bake
```

Repeated session conclusion: remote training uses `lerobot-train`; local `SmolVLABackend` changes do not affect remote runs.

## Depth Commands Seen

Packed depth baking:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=eternalmay33/04_05_06_07_merged_may-sim_depth \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 --clip-max-m=0.20 --overwrite
```

Preferred new-artifact bake:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=eternalmay33/04_05_06_07_merged_may-sim_depth \
  --out-repo-id=eternalmay33/04_05_06_07_merged_may-sim_depth_turbo \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 --clip-max-m=0.20 --overwrite
```

Estimated gripper depth:

```bash
python -m vbti.logic.depth.add_gripper_depth \
  --src eternalmay33/01_02_03_merged_may-sim_detection \
  --dst eternalmay33/01_02_03_merged_may-sim_detection_gripper-depth \
  --gripper-key observation.images.gripper \
  --depth-key observation.images.gripper_depth \
  --mode per-frame-norm
```

Real D405 sample capture:

```bash
python -m vbti.logic.depth.capture_gripper_sample \
  --serial 130523070141 \
  --seconds 5 \
  --out /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v016/data/d405_gripper_sample
```

## Evaluation Commands Seen

Direct protocol evaluation:

```bash
python vbti/logic/inference/eval_engine.py --checkpoint=v17-20k --protocol=dual_cup_30 --experiment=duck_cup_smolvla --record=True
python vbti/logic/inference/eval_engine.py --checkpoint=v17-20k --protocol=dual_cup_30 --experiment=duck_cup_smolvla --record=True --resume=True
python vbti/logic/inference/eval_engine.py --checkpoint=v18-20k --protocol=dual_cup_30 --depth=true
python vbti/logic/inference/eval_engine.py --checkpoint=v18-20k --protocol=dual_cup_30 --port=/dev/ttyACM1 --cameras=opencv --depth=true --max_steps=10000 --action_horizon=10 --record=True
```

Session analysis:

```bash
python -m vbti.logic.inference.eval_helpers ls
python -m vbti.logic.inference.eval_helpers info latest
python -m vbti.logic.inference.eval_helpers info latest --group_by=pointing_at_cup
python -m vbti.logic.inference.eval_helpers play latest 3
```

The sessions and docs repeatedly use fixed protocols such as `dual_cup_30`, `dual_cup_60`, and `checkpoint_sweep` as the comparison basis.

## Real Inference Commands Seen

```bash
python vbti/logic/inference/run_real_inference.py preview
python vbti/logic/inference/run_real_inference.py run \
  --checkpoint=vbti/experiments/duck_cup_smolvla/v001/checkpoints/best \
  --port=/dev/ttyACM0 \
  --task="pick up the object"
```

## Dataset/UVA Commands Seen

Remote UVA feature baking:

```bash
python -m vbti.logic.dataset.add_video_features \
  --dataset eternalmay33/duck_cup_v020_all \
  --root /home/vbti/anton/data/eternalmay33/duck_cup_v020_all \
  --teacher /home/vbti/anton/data/uva_teacher_v020_150k \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --batch-size 32 \
  --dtype fp16 \
  --output /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva
```

Local examples also used `--teacher vbti/experiments/duck_cup_smolvla/v020/.../pretrained_model`, `--target-camera observation.images.gripper`, and `--spatial-size 4`.

## Teleoperation / Recording Evidence

History includes repeated `lerobot-record` workflows with:

- `--robot.type=so101_follower`
- `--robot.port=/dev/ttyACM1`
- `--robot.id=may-sim` or `frodeo-test`
- OpenCV camera map for top/left/right/gripper
- `--teleop.type=so101_leader`
- `--teleop.port=/dev/ttyACM2`
- `--dataset.repo_id=...`
- `--dataset.single_task="Pick up the duck and place it in the cup"`
- `--dataset.streaming_encoding=true`

Modern docs should prefer stable `/dev/cam_*` symlinks/presets over raw numeric indices unless reproducing an old run exactly.

## Session Lessons Captured

- Remote training bypasses local backend code.
- Dataset transforms should normally bake into new artifacts.
- Depth must be represented identically in dataset and live inference.
- Use real protocol sessions, not subjective demo impressions, for comparisons.
- Camera/servo hardware state should be checked before model debugging.
