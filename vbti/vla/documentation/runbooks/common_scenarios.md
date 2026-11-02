# Common Scenarios Quick Reference

## Scenario: I Want To Record A New Dataset

1. `preflight.md`
2. `data_collection.md`
3. `dataset_preparation.md`
4. create notes with dataset ID and known issues.

## Scenario: I Want To Train A New Model Version

1. `dataset_preparation.md`
2. `create_experiment_version.md`
3. `remote_training.md`
4. `checkpoint_pull.md`
5. `evaluation.md`
6. `evaluation_analysis.md`

## Scenario: I Want A Fast Config Dry Run

```bash
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/vNNN/config.yaml
python -m vbti.logic.train.engine train-lerobot --experiment=duck_cup_smolvla --version=vNNN --dry_run
```

## Scenario: I Want To Launch A Remote Run

```bash
python -m vbti.logic.train.remote train --experiment=duck_cup_smolvla --version=vNNN --run_name=lerobot_output_r1
```

## Scenario: I Want To Run A Data-Efficiency Sweep

1. Create/verify v021-v024-style configs.
2. Dry-run each config.
3. Launch chain:

```bash
python -m vbti.logic.train.chain --versions v021,v022,v023,v024 --run_name=lerobot_output_r1
```

## Scenario: I Want To Run A UVA Sweep

1. Bake UVA features:

```bash
python -m vbti.logic.dataset.add_video_features --dataset <source> --teacher <teacher> --layer siglip_output --spatial-size 4 --t-future 4 --target-camera observation.images.gripper --output <uva_dataset>
```

2. Verify dataset.
3. Create v025-v029-style configs.
4. Launch remote chain:

```bash
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name=lerobot_output_r4_aux0003 --wait_for_session uva_bake
```

## Scenario: I Want To Evaluate A Checkpoint

1. `preflight.md`
2. Pull checkpoint if needed:

```bash
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=vNNN --checkpoint=step_XXXXX --pretrained_only=true
```

3. Run eval:

```bash
python -m vbti.logic.inference.eval_engine --checkpoint=vNNN-XXk --protocol=checkpoint_sweep --experiment=duck_cup_smolvla --version=vNNN --record=True
```

4. Analyze:

```bash
python -m vbti.logic.inference.eval_helpers info latest
```

## Scenario: I Want To Debug A Model Without Formal Evaluation

```bash
python -m vbti.logic.inference.run_real_inference preview
python -m vbti.logic.inference.run_real_inference run --checkpoint=/path/to/pretrained_model --port=/dev/ttyACM1 --task="Pick up the duck and place it in the cup"
```

Then run formal protocol evaluation before making a model-quality claim.

## Scenario: I Want To Add Depth

1. Capture/check real gripper depth if needed:

```bash
python -m vbti.logic.depth.capture_gripper_sample --serial <gripper_serial> --seconds 5 --out /tmp/d405_sample
```

2. Bake depth dataset:

```bash
python -m vbti.logic.depth.bake_packed_depth --repo_id=<src> --out-repo-id=<dst> --depth-key=observation.images.gripper_depth --clip-min-m=0.05 --clip-max-m=0.20
```

3. Train with `gripper_depth` in camera names.
4. Evaluate with:

```bash
python -m vbti.logic.inference.eval_engine --checkpoint=<ckpt> --protocol=<protocol> --depth=true --cameras=opencv_depth --record=True
```

## Scenario: Something Fails On The Robot

Run in this order:

```bash
python -m vbti.logic.servos.scan_all
python -m vbti.logic.cameras.view_cameras --preset realsense
python -m vbti.logic.cameras.check_usb
python -m vbti.logic.dataset.check_utils report <training_dataset>
```

Only after hardware and dataset checks should model code/config be blamed.
