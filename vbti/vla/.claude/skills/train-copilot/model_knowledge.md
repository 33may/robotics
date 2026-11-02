# Model knowledge

Sources: `docs/research/smolvla_training_internals.md`, `groot_training_internals.md`, `camera_handling_comparison.md`; project numbers from `docs/case-study/README.md`.

## SmolVLA (LeRobot)

- SmolVLM2-500M (SigLIP vision) + flow-matching action expert. 450M params, 100M trainable with vision frozen.
- Default finetune: frozen vision encoder, expert only, state projection trained.
- Cameras: `config.image_features` in dict insertion order, one SigLIP block per camera. Order is part of the model contract.
- Missing cameras: `empty_cameras` adds `-1`-filled images with a zero attention mask.
- Images: 512x512 aspect-preserving resize, left/top pad, `[0,1]` to `[-1,1]`.
- Temporal: last frame only (`n_obs_steps=1`), same for state.
- Normalization: images identity, state/action MEAN_STD. Preprocessor: rename, batch dim, tokenize, device, normalize. Postprocessor: unnormalize, CPU. Both saved with the checkpoint.
- `forward()` returns `(loss, loss_dict)`.
- Policy preset optimizer: AdamW lr 1e-4, betas (0.9, 0.95), eps 1e-8, wd 1e-10, grad clip 10, cosine decay to 2.5e-6.
- Prefix is ~305 tokens (256 vision + 48 language + 1 state). Language is a small share and does not steer the vision tokens: stage-aware prompts changed per-patch cosine distance by 0.0001-0.0003.
- With SigLIP frozen, masked or edited images are out of distribution for it (masking study, case study §4). Unfreezing vision with a lower LR (`vision_lr_scale=0.1`) was part of the v020 result.

## SmolVLA-UVA (`smolvla_uva`, v025-v029)

SmolVLA plus an auxiliary video-feature prediction loss (`model.aux_weight`), trained on datasets with baked SigLIP future features (`dataset/add_video_features.py`). The auxiliary loss dominated the action loss until `aux_weight=0.003`. Not evaluated on the robot. Needs the LeRobot fork (`github.com/33may/lerobot`, branch `vbti/main`).

## GR00T N1.6 (NVIDIA)

- Eagle-Block2A-2B (SigLIP-2) VLM + 32-layer DiT flow-matching action head, ~3B params.
- Finetune: VLM frozen, top 4 layers unfrozen; projector + action head trained.
- Cameras: `modality_keys` list order; asserts every view is present (no missing-camera fallback).
- Needs `meta/modality.json` in the dataset; `logic/train/backends/groot.py` generates it.
- Images: shortest edge 256, 95% center crop.
- Multi-frame via `delta_indices`. New robots use the `new_embodiment` tag (projector index 10).
- `forward()` returns `{"loss": tensor}`. Multi-GPU via DeepSpeed ZeRO-2 / ZeRO-3.

## Differences that bite

| | SmolVLA | GR00T |
|---|---|---|
| `forward()` | `(loss, dict)` | `{"loss": t}` |
| Missing camera | pad + mask | assertion |
| Image prep | 512 padded | 256 cropped |
| Frames | last only | `delta_indices` |
| Multi-GPU | Accelerate | DeepSpeed |

Image transforms are per backend; never share them.

## Training behaviour seen on duck/cup

- Useful checkpoints peak and then regress while loss keeps falling: v008 good at 20k-25k, collapsed after 30k; v019 `dual_cup_15` 10/15 at 20k and 25k, 6/15 at 30k. Pull and evaluate several checkpoints.
- v005 loss floor at 0.050 when real and sim data were mixed (conflicting gradients).
- Rest bias: v003 predictions sat on the rest pose (≈ -98 / 99 on `shoulder_lift` / `elbow_flex`) after rest frames were trimmed.
- Data efficiency (v021-v024, `checkpoint_sweep`, n=20): 48 / 96 / 192 / 383 / 765 episodes gave 2 / 4 / 7 / 14 / 19 of 20. Only the trend is meaningful at n=20.
