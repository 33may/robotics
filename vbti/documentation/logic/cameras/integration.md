# Camera Integration

## Dataset Integration

Camera logical names become LeRobot feature keys:

```text
top     -> observation.images.top
left    -> observation.images.left
right   -> observation.images.right
gripper -> observation.images.gripper
```

Depth adds:

```text
gripper_depth -> observation.images.gripper_depth
```

Changing names or ordering changes the model interface. Training, inference, and evaluation must match.

## Inference Integration

`logic/inference/run_real_inference.py` uses camera presets to initialize live streams and build policy observations.

Common patterns:

```bash
python -m vbti.logic.inference.run_real_inference preview
python -m vbti.logic.inference.run_real_inference run --cameras=realsense
python -m vbti.logic.inference.run_real_inference run --cameras=opencv_depth --depth=true
```

If `depth=true`, the gripper camera must be RealSense-backed. The inference code injects a virtual `gripper_depth` image into the observation.

## Evaluation Integration

`logic/inference/eval_engine.py` uses the same camera layer during protocol evaluation.

The top camera is used for placement overlays. If camera placement drifts, protocol pixel placements become less meaningful.

Before serious evaluation:

```bash
python -m vbti.logic.cameras.view_cameras --preset <preset>
python -m vbti.logic.cameras.calibrate --ref_dir <reference_capture_dir> --preset <preset>
```

## Depth Integration

Runtime depth path:

```text
D405 z16 depth
-> aligned to gripper RGB
-> uint16 depth map
-> depth_uint16_to_turbo_rgb()
-> observation.images.gripper_depth
```

Offline dataset depth must be prepared with the same clip/color mapping as runtime depth. See `../depth/`.

## Failure Modes

| Symptom | Likely cause | First check |
|---|---|---|
| Model behaves spatially wrong | Camera logical names swapped | View grid and compare physical positions. |
| Recording crashes mid-motion | Gripper cable/camera disconnect | USB/cable isolation, `check_usb`, camera reset. |
| Depth missing | OpenCV backend used for gripper | Use `realsense_depth` or `opencv_depth`. |
| Protocol overlay wrong | Top camera drifted | Run visual alignment overlay. |
| All cameras open but image names wrong | Preset/udev mismatch | Compare `realsense` vs `opencv` views. |
