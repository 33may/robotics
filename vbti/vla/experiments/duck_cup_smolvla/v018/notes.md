# v018 — RGB + gripper depth (5th camera, packed PNG → turbo RGB at training time)

## Status
Configured. Two pieces of code must land before this trains:

1. **Pre-aggregate** `eternalmay33/04_05_06_07_merged_may-sim_depth` (shared with v017).
2. **DataLoader transform** that decodes `observation.images.gripper_depth` from packed-PNG (uint16 split into RGB high/low/preview bytes) → metric depth (m) → clip to canonical `[0.07, 0.95]` m → linear normalize → turbo colormap → uint8 RGB. Must run per-batch on CPU before SmolVLA's preprocessor.
3. **Inference-time** depth read + colorize in `vbti/logic/inference/run_real_inference.py` — capture D405 depth, apply identical clip + colormap, inject as `observation.images.gripper_depth`.

(1) gates training; (2) gates training quality; (3) gates real-robot eval.

## Hypothesis
Adding gripper depth as a **5th camera** (Q2-B1 path from v016 research) gives the frozen SigLIP encoder explicit close-range 3-D cues the RGB stream can't provide — surface curvature, gripper-to-object distance, and clutter geometry beyond the visible face. If depth helps, v018 lifts MAE / success vs v017 on real robot **with no model code changes** — only the camera list differs.

## A/B against v017
- **v017** — 4 cams: `[top, left, right, gripper]`
- **v018** — 5 cams: `[top, left, right, gripper, gripper_depth]`

Same merged dataset, same hyperparams, same model.

## Dataset
Same as v017: `eternalmay33/04_05_06_07_merged_may-sim_depth` (175 ep / 70 384 frames). The `observation.images.gripper_depth` feature is `dtype: image, shape: [480, 640, 3]` — uint16 metric depth packed into uint8 RGB:
- R = high byte (`(d>>8) & 0xFF`)
- G = low byte (`d & 0xFF`)
- B = preview byte (`d/256`, viewer-only — the encoder mustn't see this raw)

Storage scale 0.0001 m/unit (D405 default). **Canonical clip [0.05, 0.20] m** (uint16 [500, 2000]) — tuned 2026-04-30 to put the entire LUT span inside the gripper workspace; anything farther saturates to the red end of turbo as an explicit "background / not relevant" signal.

## Architecture (Q2-B1, no changes vs v014)
- SmolVLA fills camera slots `camera1, camera2, camera3` with `[top, left, right]` (pretrained vision weights), then pushes `gripper` and `gripper_depth` into `empty_camera_0` and `empty_camera_1` slots (smolvla.py L92, L67-69).
- Frozen SigLIP encoder, expert-only training, state proj trainable.
- No LoRA, no side branch, no cross-attention. Q2-B2 deferred to v019+.

## DataLoader transform (TODO — to implement before training)
**Where:** new `vbti/logic/dataset/depth_transform.py`, wired into `vbti/logic/train/backends/smolvla.py::make_dataloaders` as a wrapping Dataset on `train_dataset` / `val_dataset` when `gripper_depth` is in the camera list.

**What it does (per sample):**
```
rgb_packed (uint8, H, W, 3)         ← from LeRobotDataset
  → unpack_rgb_to_uint16            ← R<<8 | G  (B ignored)
  → cast to float32, scale × 0.0001 ← metric depth in meters
  → np.clip(d, 0.05, 0.20)          ← canonical clip
  → (d - 0.05) / (0.20 - 0.05)      ← normalize to [0, 1]
  → cv2.applyColorMap(× 255, TURBO) ← uint8 BGR
  → cvtColor(BGR2RGB)               ← uint8 RGB
  → back to torch.uint8 CHW          ← what SigLIP preprocessor expects
```
ImageNet stats then apply normally on the turbo RGB. SigLIP sees a smooth, non-banded image whose color encodes distance.

**Why per-batch CPU and not per-frame baked-in:** keeps the dataset lossless and storage tight (uint16 metric in PNG, ~22 GB at 500 ep). v017 and v019+ can re-render with different clips without rebuilding the dataset.

**Why turbo and not raw monochrome:** SmolVLA's frozen SigLIP expects 3-channel imagery; SigLIP was never trained on greyscale-replicated depth, but a colormapped depth at least sits on the natural image manifold.

## Inference (TODO — to wire before real-robot eval)
`vbti/logic/inference/run_real_inference.py` must:
1. On preset selection, mark gripper as a depth-emitting RealSense (already true for `realsense` preset, gripper serial 128422270260).
2. Per step: capture color **and** depth from gripper. The color frame stays as `observation.images.gripper`. The depth frame must be:
   - `read_latest_depth()` → uint16 metric (camera_realsense.py extension already added)
   - the same clip + turbo transform applied at training (pulled from a shared utility — `vbti/logic/depth/colorize.py::colorize_fixed_clip` already implements it).
   - injected as `observation.images.gripper_depth`.
3. Identical clip range (0.05–0.20 m) on both sides of the train/inference boundary.

## Hyperparams (= v017, = v014)
50 000 steps, batch 32, lr 1e-4, cosine decay → 2.5e-6, warmup 1000. bf16. num_workers 8.

## Failure modes to watch for
- **Stat collapse:** if SmolVLA's per-channel RGB stats on the merged dataset are computed *before* our transform, normalization will be wrong. v018 uses `use_imagenet_stats: true` so dataset stats aren't applied to vision — we should be safe, but verify in training logs.
- **Depth encoder waste:** SigLIP slot is treated as another camera. If the transform image is too "alien" (turbo's non-monotonic hue), encoder embeddings may be uninformative. Mitigation: monitor v018 vs v017 loss curves; if they're indistinguishable at 50k, try a monotonic (e.g. viridis) variant in v019.
- **Training/inference distribution gap:** any difference in the depth clip / colormap between train and infer kills the experiment. Both must come from one shared function (`colorize_fixed_clip`).

## Outputs
- `lerobot_output_r1/` — training run
- `evaluation.md` — real-robot eval
- Comparison vs v017 in `v018/evaluation.md` and a summary at the v016 research level.
