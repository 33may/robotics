# Runbook: Dataset Preparation After Recording

Use after recording or after creating a transformed dataset.

## 1. Inspect Dataset

```bash
python -m vbti.logic.dataset.check_utils report <repo_id_or_path>
python -m vbti.logic.dataset.check_utils cameras <repo_id_or_path>
python -m vbti.logic.dataset.check_utils plot_actions <repo_id_or_path> --save=/tmp/actions.png
```

Write down:

- episodes;
- frames;
- FPS;
- state/action dimension;
- camera keys;
- task text;
- action distribution issues.

## 2. Decide If Dataset Needs Curation

Look for:

- failed demonstrations;
- excessive rest frames;
- wrong camera mapping;
- wrong calibration profile;
- missing depth/camera features;
- inconsistent task text.

## 3. Convert HDF5 Simulation Dataset If Needed

Inspect:

```bash
python -m vbti.logic.dataset.hdf5_utils report /path/to/data.hdf5
python -m vbti.logic.dataset.convert_utils discover /path/to/data.hdf5
```

Convert:

```bash
python -m vbti.logic.dataset.convert_utils convert \
  /path/to/data.hdf5 \
  eternalmay33/<converted_dataset> \
  "Pick up the duck and place it in the cup"
```

Verify:

```bash
python -m vbti.logic.dataset.convert_utils verify eternalmay33/<converted_dataset>
python -m vbti.logic.dataset.check_utils report eternalmay33/<converted_dataset>
```

## 4. Add/Bake Depth If Needed

Preferred packed-depth bake:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=<source_depth_dataset> \
  --out-repo-id=<output_turbo_dataset> \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 \
  --clip-max-m=0.20 \
  --depth-scale-m=1e-4
```

Add black placeholder depth for schema compatibility:

```bash
python -m vbti.logic.dataset.add_zero_depth_turbo \
  --src=<no_depth_dataset> \
  --dst=<padded_dataset> \
  --reference=<depth_dataset>
```

## 5. Subsample For Data-Efficiency Experiments

```bash
python -m vbti.logic.dataset.subsample \
  --src=<source_dataset> \
  --stride=4 \
  --dst=<output_dataset> \
  --start=0
```

Common strides:

- `16`: about 6.25%;
- `8`: about 12.5%;
- `4`: about 25%;
- `2`: about 50%.

## 6. Detection/Phase Augmentation If Needed

Only for detection-state experiments:

```bash
python -m vbti.logic.detection.process_dataset <dataset>
python -m vbti.logic.detection.phases <dataset>
python -m vbti.logic.dataset.augment <dataset> --augmentations detection phase -o <augmented_dataset>
```

Verify `observation.state` dimension after augmentation.

## 7. UVA Feature Bake If Needed

```bash
python -m vbti.logic.dataset.add_video_features \
  --dataset <source_dataset> \
  --teacher <teacher_pretrained_model> \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --output <output_dataset>
```

For remote training, bake where the remote can access the result or sync it afterward.

## 8. Final Verification

```bash
python -m vbti.logic.dataset.check_utils report <final_dataset>
python -m vbti.logic.dataset.check_utils cameras <final_dataset>
```

Do not create a training version until the final dataset report is sane.
