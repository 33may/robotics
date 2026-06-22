# Depth Train/Inference Parity

## Why This Matters

Depth is only useful if the model sees the same representation during training and inference. If offline depth is clipped/colorized differently than live D405 depth, the checkpoint receives a shifted image distribution.

## Offline Path

Typical offline path:

```text
packed uint16 depth image in LeRobot dataset
-> unpack / scale to meters
-> clip to task range
-> turbo RGB colorization
-> write as observation.images.gripper_depth
```

Command:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=<source> \
  --out-repo-id=<output> \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 \
  --clip-max-m=0.20 \
  --depth-scale-m=1e-4
```

## Runtime Path

Runtime path:

```text
RealSense gripper D405 z16
-> aligned to gripper RGB
-> uint16 depth frame
-> depth_uint16_to_turbo_rgb()
-> policy input observation.images.gripper_depth
```

Inference examples:

```bash
python -m vbti.logic.inference.run_real_inference run --depth=true --cameras=opencv_depth
python -m vbti.logic.inference.eval_engine --checkpoint=v18-20k --protocol=dual_cup_30 --depth=true --cameras=opencv_depth
```

## Fixed Clip

Current close manipulation task uses:

```text
clip_min_m = 0.05
clip_max_m = 0.20
```

This emphasizes near-gripper geometry rather than the full table depth range.

## Estimated Depth Caveat

Depth Anything V2 metric indoor depth can be useful as a structural signal, but its absolute scale can differ strongly from real D405 depth. If training on estimated depth and deploying with real depth, compare distributions first.

Command:

```bash
python -m vbti.logic.depth.compare_real_vs_estimated --d405-dir <sample> --src-dataset <dataset> --out <results>
```

## Checklist Before Training A Depth Model

- Dataset contains `observation.images.gripper_depth`.
- Feature shape matches RGB image shape expected by LeRobot/SmolVLA.
- Offline transform used fixed task clip.
- Training config includes `gripper_depth` in camera names/rename map.
- Inference/eval will use `--depth=true` and a RealSense-backed gripper camera.
- RGB-only comparison checkpoint is evaluated under the same protocol.
