# exp02a — Estimated depth preview on may-sim gripper frames

## Method
- **Model**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (fp16, RTX 4070 Ti SUPER). Metric output (meters).
- **Dataset**: `eternalmay33/01_02_03_merged_may-sim_detection` (LeRobot v3.0, 244 eps, 135846 frames, 30 fps).
- **Camera key**: `observation.images.gripper` (480×640 RGB).
- **Sampling**: 3 episodes (10, 120, 230) × 5 evenly-spaced frames = 15 samples.
- **Render**: per-sample 2×4 grid: RGB + raw auto-norm + 6 colormap/clip configs.

## Visual observations
- **Spec'd clips [0.05, 0.5]m and [0.10, 1.0]m saturate everything.** Most pixels map to the far end (deep red in turbo). The [0.10, 1.0]m clip only resolves the closest gripper-finger surface.
- **Data-fit clip [0.30, 2.0]m gives clean contrastive maps** across all 15 samples. Duck, cup, gripper jaws, table plane, and far floor distinguishable.
- **Turbo > Inferno > Viridis > Gray.** Turbo has best perceptual uniformity and object/background separation. Gray fails entirely — duck and cup blur into background. Bad fit for a frozen SigLIP that needs local contrast.
- **Failure modes**: blue duck plastic occasionally reads slightly low; specular/shadow boundaries cause sharp jumps in a few frames. No NaNs, no catastrophic holes.

## Distribution stats (15 samples, 4.6M pixels)

| stat | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean | std |
|------|-----|----|----|-----|-----|-----|-----|-----|-----|------|-----|
| m | 0.22 | 0.35 | 0.44 | 0.74 | 0.93 | 1.21 | 1.78 | 2.30 | 3.38 | 1.00 | 0.41 |

Roughly log-normal, centered ~0.9m, tail to ~3m. **Less than 5% of pixel mass falls below 0.3m.** The originally proposed 0.05–0.5m clip covers almost nothing of the actual estimated-depth distribution.

## Critical finding — scale mismatch
DA-V2-Indoor outputs **0.4–2m** for scenes the D405 will see at **0.05–0.5m**. The model is biased toward room-scale indoor scenes; wrist-mounted close-range is OOD for it.

**Implication**: estimated depth (DA-V2 metric) and real depth (D405) cannot share a fixed colormap clip without one of them being unusable. Two paths:
- **Per-source affine rescale** before colorization (push D405 into DA-V2 range OR vice versa). Preserves relative metric meaning per source but breaks cross-source consistency in absolute terms.
- **Per-frame min-max normalized → [0,1] → turbo.** Throws away absolute distance entirely; both sources get the same visual treatment. Cheap, robust, but the policy can never learn "object is 10cm vs 50cm away" — only relative depth within a frame.

## Colormap recommendation (DA-V2 backfill only — pending real-D405 confirmation)
- **Turbo** colormap
- **Clip [0.30, 2.0]m** if going fixed-clip with per-source rescale
- **OR per-frame [0,1] normalization** if going relative-only (recommended for first pass since it sidesteps the scale mismatch)
- Encode as 3-channel uint8 H.264

**Do NOT commit this to the full v016 backfill until exp02b confirms behavior on real D405 frames.**

## Output files
- `plots/sample_ep<EP>_f<FR>.png` — 15 comparison grids
- `plots/depth_histogram.png` — pooled distribution
- `depth_stats.json` — numeric summary
- `run_preview.py` — the script the subagent ran
