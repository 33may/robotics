# Depth Commands

## Capture Real D405 Sample

```bash
python -m vbti.logic.depth.capture_gripper_sample \
  --serial 123622270367 \
  --seconds 5 \
  --width 640 --height 480 --fps 30 \
  --out /path/to/d405_gripper_sample
```

Observed historical example used serial `130523070141`; verify current gripper serial before running.

Outputs:

```text
depth_uint16.npy
color_uint8.npy
preview_*.png
histogram.png
```

## Bake Packed Depth To Turbo RGB

Preferred new artifact form:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=<source_repo> \
  --out-repo-id=<output_repo> \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 \
  --clip-max-m=0.20 \
  --depth-scale-m=1e-4 \
  --workers=1 \
  --overwrite
```

Historical command:

```bash
python -m vbti.logic.depth.bake_packed_depth \
  --repo_id=eternalmay33/04_05_06_07_merged_may-sim_depth \
  --out-repo-id=eternalmay33/04_05_06_07_merged_may-sim_depth_turbo \
  --depth-key=observation.images.gripper_depth \
  --clip-min-m=0.05 --clip-max-m=0.20 --overwrite
```

Warning: omitting `--out-repo-id` may perform an in-place temp/swap. Prefer `--out-repo-id`.

## Add Estimated Gripper Depth

```bash
python -m vbti.logic.depth.add_gripper_depth \
  --src <src_repo> \
  --dst <dst_repo> \
  --gripper-key observation.images.gripper \
  --depth-key observation.images.gripper_depth \
  --mode per-frame-norm \
  --model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
  --device cuda
```

Optional episode slice:

```bash
--episodes 0:10
```

Modes:

- `per-frame-norm`: good visual contrast, not absolute metric scale.
- fixed clipping modes if implemented/selected in code.

## Compare Real Vs Estimated

```bash
python -m vbti.logic.depth.compare_real_vs_estimated \
  --d405-dir /path/to/d405_sample \
  --src-dataset eternalmay33/01_02_03_merged_may-sim_detection \
  --n-est-samples 12 \
  --n-candidates 60 \
  --max-median-depth-m 0.85 \
  --out /path/to/results
```

Outputs:

```text
summary.json
plots/histograms.png
plots/panel_*.png
```

## Runtime API

```python
from vbti.logic.depth.realtime_prepare import depth_uint16_to_turbo_rgb

rgb = depth_uint16_to_turbo_rgb(depth_u16)
```

Used by real inference/evaluation to match the baked training dataset representation.
