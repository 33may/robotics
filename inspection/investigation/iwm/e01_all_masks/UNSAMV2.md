# UnSAMv2 for e01 — inference setup

Paper: "Self-Supervised Learning Enables Segment Anything at Any Granularity"
(Yu, Darrell, Wang; arXiv 2511.13714). Code: https://github.com/yujunwei04/UnSAMv2,
Apache-2.0. Set up 2026-10-08 for inference only, verified on CPU only (GPU was taken).

## Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsam
cd ~/projects/robotics      # anywhere but ~/projects/UnSAMv2 (see gotcha 5)
CUDA_VISIBLE_DEVICES="" python inspection/investigation/iwm/e01_all_masks/unsamv2_masks.py \
    --image inspection/data/runs/2209-compute-ray-demo5/steps/000/rgb.png \
    --granularity 0.3 --out inspection/data/iwm/_setup/unsamv2/demo5_s000_plus_g0.3 \
    --variant plus --device cpu
```

`--variant base|plus` (default plus), `--device cpu|cuda` (default cpu; cpu hides the GPU
via `CUDA_VISIBLE_DEVICES=""` before torch is imported). Optional overrides of the notebook
settings: `--pred-iou-thresh`, `--stability-score-thresh`, `--points-per-side`,
`--points-per-batch` (batch size only, result-invariant, lower it if RAM is short).

Output in `--out`: `masks.npy` bool (N,H,W) largest-first, `masks.json` (per mask:
area, bbox xywh, predicted_iou, stability_score, prompt point; plus every setting, the
checkpoint sha256, repo commit, timings), `overlay.png`.

## What the script does (= the repo's whole_image_segmentation.ipynb)

```python
sam2 = build_sam2("configs/unsamv2_small.yaml", ckpt, device=device, apply_postprocessing=True)
gen = SAM2AutomaticMaskGenerator(model=sam2, points_per_side=64, points_per_batch=128,
    mask_threshold=-1, pred_iou_thresh=0.77, stability_score_thresh=0.9,
    stability_score_offset=0.7, crop_n_layers=0, box_nms_thresh=0.7,
    crop_n_points_downscale_factor=1, min_mask_region_area=0, use_m2m=True)
anns = gen.generate(image_rgb_uint8, gra=G)
```

On cuda the notebook also enters `torch.autocast("cuda", bfloat16)` and turns on TF32; the
script does the same for `--device cuda`. The notebook's optional CascadePSP refinement
(`segmentation-refinement` package, CUDA-only) is not installed and not applied; the
authors' HF demo does not apply it either.

## Granularity value

- Valid range **0.1 .. 1.0** (paper, HF demo slider `0.1..1.0` step 0.01, eval script
  sweeps 0.1..1.0). `generate()` defaults to 1.0. Outside the range the script only warns.
- **Low = fine parts, high = whole objects.** Paper: "At low granularity values, the model
  reveals fine-grained parts with precise boundaries; as the granularity increases, it
  gradually merges regions into larger and more semantically coherent entities."
- Definition in training (pseudo labels, `granularity_divide_and_conquer/compute_granularity.py`):
  within one image's mask hierarchy, `g = 0.1 + 0.9 * (sqrt(A) - sqrt(A_min)) / (sqrt(A_max) - sqrt(A_min))`
  with A the mask area, A_min/A_max the smallest/largest mask of that image. So g is a
  relative mask-size level per image (sqrt-area normalised), not an absolute pixel size
  and not a hierarchy depth. (The arXiv HTML renders the paper's formula linear in area,
  `g = 0.1 + 0.9 (A - A_min)/(A_max - A_min)`; the code uses sqrt. Not verified against
  the PDF.)
- Mechanically: g is Fourier-embedded (128 dims, temperature 100) -> 3-layer MLP ->
  one extra sparse prompt token next to the point prompt. The mask decoder has an extra
  "granularity" output token whose mask is what `generate` returns (one mask per grid
  point, see gotcha 9).
- Demo defaults: whole-image 0.15, interactive 0.2, video 0.33.

## Model

- Backbone: **SAM 2.1 hiera-small** (`configs/unsamv2_small.yaml`: embed_dim 96, stages
  [1,2,11,2], 1024 px input). 46.4 M parameters in the checkpoint.
- Checkpoints hold the full model; base vs plus differ in exactly 70 tensors: 56 LoRA
  tensors (rank 8, alpha 16, on q/k/v/out_proj of the mask decoder's two-way transformer)
  plus 14 granularity-module tensors (prompt-encoder Fourier MLP + embedding, decoder
  gra token + its hypernetwork MLP). Everything else is identical frozen SAM-2.
- UnSAMv2 = trained on CutLER pseudo masks only (unsupervised). UnSAMv2+ = divide stage
  additionally uses SA-1B ground-truth masks (lightly supervised). Paper AR1000 on
  whole-image: 68.3 vs 74.1. Notebook and HF demo default to plus.

## Weights

`~/projects/UnSAMv2/checkpoints/` (Google Drive, fetched with `gdown <file-id>`):

| file | drive id | bytes | sha256 |
| --- | --- | --- | --- |
| `unsamv2_ckpt.pt` (base) | `1iYmYFeJxY4aI1lG5F2ZFL8tzzyGkCgX0` | 188689844 | `d1f5b587b979918f0dd5e7e6623b3165997efc9b48bb22bdb9f3700c6b348e62` |
| `unsamv2_plus_ckpt.pt` (plus) | `1M56QN2CWfou7h5azxJvc_igmbQh9D2af` | 188689844 | `4067a6966b06df984828f537da0f02389ff28655a8985a1e1f4a3e1de4077195` |

Both are full training checkpoints (`model`, `optimizer`, `epoch`, ...); the code loads
`["model"]` with `weights_only=True`.

## Repo and env

- Clone: `~/projects/UnSAMv2` at commit `2c9db1fd7da3c17590358eeb8c110231bd6a0889`
  (2026-09-15 "Add Apache 2.0 license"). No repo file was modified. Editable install adds
  `__pycache__` dirs only.
- Env `unsam` (miniconda): Python 3.11.17, torch 2.11.0+cu128, torchvision 0.26.0+cu128
  (same build as `robo`; cu128 wheels carry sm_120 for the RTX 5090), hydra-core 1.3.7,
  omegaconf 2.3.1, iopath 0.1.10, numpy 2.4.6, pillow 12.3.0, tqdm 4.70.1,
  opencv-python-headless 5.0.0.93 (overlay only), gdown 6.4.1 (download only).
- Install that was done:

```bash
conda create -n unsam python=3.11
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install hydra-core iopath opencv-python-headless gdown
cd ~/projects/UnSAMv2/sam2 && SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e .
```

## CPU smoke test (2026-10-08)

Frame `inspection/data/runs/2209-compute-ray-demo5/steps/000/rgb.png` (1280x720),
variant plus, notebook settings, 24-core CPU shared with a running GPU job.
Outputs under `inspection/data/iwm/_setup/unsamv2/demo5_s000_plus_g{0.3,0.8}/`.

| run | thresholds (iou / stab) | masks | generate | wall incl. import | peak RSS | pixels covered |
| --- | --- | --- | --- | --- | --- | --- |
| `demo5_s000_plus_g0.3` | 0.77 / 0.9 (notebook) | 46 | 363 s | 6:06 | 7.9 GB | 33 % |
| `demo5_s000_plus_g0.8` | 0.77 / 0.9 (notebook) | 5 | 304 s | 5:06 | 7.9 GB | 26 % |
| `demo5_s000_plus_g0.3_evalthresh` | 0.65 / 0.7 (eval script) | 194 | 315 s | 5:20 | 8.3 GB | 91 % |

Model load is 0.3 s; the time is the 4096-point grid through the decoder twice (m2m)
plus full-resolution upsampling, all on CPU. What the overlays show:

- g = 0.3: 46 small masks (largest 20.8 k px = 2.3 % of the frame): shelf contents,
  wall patches, parts of the table edge. The heatsink (the object) has no mask of its
  own and no per-fin masks; the table top has none either.
- g = 0.8: 5 masks, the heatsink is one clean whole-object mask (predicted IoU 0.90+),
  the rest are wall / shelf regions. Table top still unmasked.
- g = 0.3 with the eval-script thresholds (extra run, `--pred-iou-thresh 0.65
  --stability-score-thresh 0.7`): 194 masks, 91 % of pixels covered, on average 3
  masks per pixel. The heatsink now gets several chunky masks (fin groups, the front
  strip, the left face) but still no per-fin masks; most of the extra masks are
  fragments of the plain table top and wall.
- So on this frame the notebook thresholds leave large plain surfaces unmasked, and the
  looser eval thresholds fill them with surface fragments. Threshold choice (gotcha 10)
  dominates the mask count far more than the granularity value does.

## Gotchas

1. INSTALL.md wants Python 3.10 + torch 2.5.1+cu121 and `requirements.txt` is the
   whole-repo list (gradio, mmcv 1.6.2, timm 0.6.11, detectron2, DINOv3, pinned
   numpy 2.1.2 / pillow 9.5 / setuptools 52). None of that is needed for inference;
   the `sam2/setup.py` deps (torch, torchvision, numpy, tqdm, hydra-core, iopath,
   pillow) are enough. Ignored the pins; nothing broke on torch 2.11 / numpy 2.4.
2. `sam2/pyproject.toml` lists torch as a build requirement, so a plain `pip install -e .`
   would pull a second torch into the isolated build env. Use `--no-build-isolation
   --no-deps` after installing torch. `SAM2_BUILD_CUDA=0` skips the CUDA extension.
3. The repo **commits a prebuilt binary** `sam2/sam2/_C.so` (py3.10, CUDA 12 build) and
   prebuilt `MultiScaleDeformableAttention` `.so`/eggs under
   `granularity_divide_and_conquer/`. Treat as untrusted. `_C` is only imported lazily by
   `sam2.utils.misc.get_connected_components`, which runs only when
   `min_mask_region_area > 0` (hole/sprinkle removal) or in the video predictor. The
   notebook and the script use `min_mask_region_area=0`, so it is never loaded. Do not
   raise `min_mask_region_area` without first deleting that `.so` and rebuilding it.
4. `sam2/build_sam.py` imports `training.utils.checkpoint_utils` (LoRA split/apply), so
   the `training` package from the repo is part of the inference install; it needs
   `iopath`.
5. Run Python from anywhere **except** `~/projects/UnSAMv2` itself: `build_sam.py` raises
   if the repo dir `sam2/` shadows the `sam2` package.
6. Hydra: the config name is `configs/unsamv2_small.yaml`, resolved inside the installed
   package by `initialize_config_module("sam2")` in `sam2/__init__.py`; it does not depend
   on cwd.
7. `build_sam2` logs a warning if the checkpoint lacks
   `sam_prompt_encoder.granularity_round_embedding.*`; those keys are optional and absent
   in both checkpoints. Harmless.
8. LoRA is wired after the base weights load (`setup_lora_after_loading`), printing
   "Applying LoRA to UnSAMv2..." to stdout. 56 tensors are loaded from the checkpoint.
   `model.eval()` turns off the LoRA dropout (0.1) and the granularity-MLP dropout.
9. `SAM2AutomaticMaskGenerator` passes `multimask_output=True`, but `MaskDecoderGra.forward`
   always returns the single granularity-token mask, so there is **one candidate per grid
   point** (4096 for points_per_side 64), not three. The reported `predicted_iou` is the
   max over the three standard multimask IoU-head outputs, not a head trained for the
   granularity mask. `stability_score` is computed on the returned mask with the
   notebook's `mask_threshold=-1`, offset 0.7.
10. The notebook's filter thresholds (0.77 / 0.9) differ from the authors' evaluation
    script (`evaluation/whole_image/coco_annotator.py`: g < 0.5 -> 0.65 / 0.7, else
    0.5 / 0.7; points_per_side 64, use_m2m) and the notebook itself says "worth to try
    different combinations". Hence the CLI overrides.
11. The mask-decoder attention in this fork is plain matmul + softmax (no SDPA / flash
    kernel selection), so CPU works without the transformer.py edit that SAM-2's
    INSTALL.md mentions. It also stores `last_attention_map` on every attention module.
12. CPU memory: ~6 GB RSS during generation at points_per_batch 128 on a 1280x720 image
    (mask logits are upsampled to full resolution per batch). Lower `--points-per-batch`
    if needed; results do not change.
13. `masks.npy` is bool at full resolution: ~0.92 MB per mask at 1280x720, so several
    hundred MB per run. Pack it if this becomes a problem.
14. gdown 6.x dropped `--fuzzy`; pass the Drive file id directly. It keeps the authors'
    file names (`unsamv2_ckpt.pt`, `unsamv2_plus_ckpt.pt`).
15. GPU use is **untested** here. The cu128 torch build supports sm_120; the script
    enables bf16 autocast + TF32 on cuda like the notebook. First GPU run should be checked
    against the CPU outputs (bf16 may change mask counts slightly).
