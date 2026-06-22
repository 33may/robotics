# Detector Distillation

## Purpose

Distillation creates lighter detector models or filtered labels from teacher detections. This is separate from the main VLA policy training loop.

## Export Grounding DINO ONNX

```bash
python -m vbti.logic.detection.distill.export_onnx \
  --output ~/.cache/vbti/grounding_dino_base.onnx
```

Also writes a sibling `.json` with fixed prompt/tokenization metadata.

## Stage 1 Labels

```bash
python -m vbti.logic.detection.distill.distill_stage1 \
  --dataset <repo_id> \
  --output /path/detection_labels_stage1.parquet \
  --rgb-cache /path/cache.parquet \
  --refresh-rgb
```

Optional drop controls:

```bash
--drop-all-eps "1,2"
--drop-duck-eps "3"
--drop-cup-eps "4"
--skip-phases
```

## Stage 2 Cleanup

```bash
python -m vbti.logic.detection.distill.distill_stage2 \
  --input /path/stage1.parquet \
  --output /path/stage2.parquet \
  --max-gap 30 \
  --neighbor-k 3
```

## Final Filter And Galleries

```bash
python -m vbti.logic.detection.distill.distill_filter \
  --dataset <repo_id> \
  --output /path/detection_labels_final.parquet \
  --gallery-dir /path/data_analysis \
  --n-samples 24 \
  --seed 42
```

Outputs include filtered parquet, stats, galleries, and findings under the gallery directory.

## Train Student

```bash
python -m vbti.logic.detection.distill.distill cache --force
python -m vbti.logic.detection.distill.distill train --all --run m1_baseline --max-epochs 40 --batch-size 128 --model mobilenet_v3_small
python -m vbti.logic.detection.distill.distill eval --cam left --checkpoint /path/best.pt
```

## Pitfalls

- ONNX export assumes fixed processor image shape; mismatches can fail at runtime.
- Distilled detectors are only useful if validated on the actual camera views.
- Do not train a policy on detector-state augmentation unless live inference can reproduce the same state reliably.
