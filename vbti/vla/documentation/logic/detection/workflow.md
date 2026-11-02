# Detection And Phase Workflow

## 1. Generate Detection Results

```bash
python -m vbti.logic.detection.process_dataset <repo_id_or_path> \
  --stride 30 \
  --gripper-stride 10 \
  --batch-size 4 \
  --cameras left right top gripper \
  --threshold 0.1 \
  --conf-hold-threshold 0.15 \
  --device cuda
```

Output:

```text
<dataset_root>/detection_results.parquet
```

The script decodes videos, runs detection sparsely, and interpolates/holds detections to create dense frame-level results.

## 2. Generate Phase Labels

```bash
python -m vbti.logic.detection.phases <repo_id_or_path> --fps 30
```

Output:

```text
<dataset_root>/phase_labels.parquet
```

Phases are motion/state-machine labels such as:

- reach;
- pregrasp;
- grasp;
- transport;
- release.

## 3. Bake Into Dataset State

```bash
python -m vbti.logic.dataset.augment <repo_id_or_path> \
  --augmentations detection phase \
  -o <output_repo_id> \
  --cameras left right top gripper \
  --detection-parquet detection_results.parquet
```

Options:

```bash
--include-confidence
--drop top_duck
--dry-run
```

## 4. Verify

```bash
python -m vbti.logic.dataset.check_utils report <output_repo_id>
```

Check that `observation.state` dimension matches the intended model.

## Use Cases

- Explicit detector-coordinate state augmentation experiments.
- Dataset quality inspection.
- Failure analysis.
- Distillation label generation.
- Debug overlays during inference/eval.

## Non-Use Case

Do not silently add detector coordinates to the main VLA training path. It changes the learning problem and can create an easy-signal shortcut.
