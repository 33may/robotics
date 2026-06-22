# Detection-State Augmentation

## State Layout

Base state:

```text
6 joints
```

Detection-augmented state:

```text
6 joints + 16 detection values = 22 dims
```

Detection value order:

```text
[left, right, top, gripper] x [duck, cup] x [cx, cy]
```

This order is fixed.

## Training

Bake detection features before training:

```bash
python -m vbti.logic.detection.process_dataset <dataset>
python -m vbti.logic.dataset.augment <dataset> --augmentations detection -o <dataset_detection>
```

Training config must point at the augmented dataset.

## Inference

Use:

```bash
python -m vbti.logic.inference.run_real_inference run --checkpoint=<ckpt> --detection=true
```

Evaluation:

```bash
python -m vbti.logic.inference.eval_engine --checkpoint=<ckpt> --protocol=<protocol> --detection=true
```

## Live Detector Rules

- Live camera frames are already RGB in the inference path.
- Do not BGR-swap before detector input.
- Detection holder uses last-good values to avoid single-frame detector dropouts.
- Missing detection can become a behavioral failure if the checkpoint relies on coordinates.

## When To Use

Use detection-state augmentation only for a controlled experiment where the question is explicitly about coordinate augmentation.

## Why It Was Deprioritized

The project found detector coordinates can become an easy shortcut. If removing detector coordinates hurts more than removing image information, the policy is not learning the intended visual behavior. The main direction keeps object understanding inside image observations.
