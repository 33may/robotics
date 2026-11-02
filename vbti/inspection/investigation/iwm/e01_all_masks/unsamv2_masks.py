"""UnSAMv2 whole-image segmentation of one image at one granularity.

Mirrors sam2/notebooks/whole_image_segmentation.ipynb of
https://github.com/yujunwei04/UnSAMv2 (see UNSAMV2.md for commit, env, gotchas):
build_sam2(configs/unsamv2_small.yaml, ckpt) + SAM2AutomaticMaskGenerator with the
notebook's settings, then generate(image, gra=G). The notebook's optional CascadePSP
refinement (CUDA-only, extra package) is NOT applied; the HF demo does not use it either.

Run (conda env `unsam`, from anywhere except ~/projects/UnSAMv2 itself):
  CUDA_VISIBLE_DEVICES="" python vbti/inspection/investigation/iwm/e01_all_masks/unsamv2_masks.py \
      --image PATH.png --granularity 0.3 --out OUTDIR [--variant base|plus] [--device cpu|cuda]

Writes into OUTDIR:
  masks.npy    bool (N, H, W), masks sorted by area, largest first
  masks.json   per-mask predicted_iou / stability_score / area / bbox + every setting used
  overlay.png  all masks over the image (random colour at 0.5 alpha, blue contours)

Granularity G: 0.1 = finest parts ... 1.0 = whole objects (paper, HF demo slider range).
"""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

UNSAMV2_ROOT = Path.home() / "projects" / "UnSAMv2"
CHECKPOINTS = {
    "base": UNSAMV2_ROOT / "checkpoints" / "unsamv2_ckpt.pt",
    "plus": UNSAMV2_ROOT / "checkpoints" / "unsamv2_plus_ckpt.pt",
}
# Hydra resolves this inside the installed `sam2` package (sam2/sam2/configs/).
MODEL_CFG = "configs/unsamv2_small.yaml"

# SAM2AutomaticMaskGenerator settings, copied verbatim from the notebook.
GENERATOR_KWARGS = dict(
    points_per_side=64,
    points_per_batch=128,
    mask_threshold=-1,
    pred_iou_thresh=0.77,
    stability_score_thresh=0.9,
    stability_score_offset=0.7,
    crop_n_layers=0,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=0,
    use_m2m=True,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--granularity", required=True, type=float,
                   help="0.1 (finest parts) .. 1.0 (whole objects)")
    p.add_argument("--out", required=True, type=Path, help="output directory (created)")
    p.add_argument("--variant", choices=sorted(CHECKPOINTS), default="plus",
                   help="base = UnSAMv2 (unsupervised), plus = UnSAMv2+ (default)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    # Optional deviations from the notebook; defaults keep the notebook values.
    p.add_argument("--pred-iou-thresh", type=float, default=None)
    p.add_argument("--stability-score-thresh", type=float, default=None)
    p.add_argument("--points-per-side", type=int, default=None)
    p.add_argument("--points-per-batch", type=int, default=None,
                   help="batch size only; does not change the result")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(UNSAMV2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def overlay(image, masks):
    """Notebook's show_anns, rendered without matplotlib: largest masks first,
    random colour at alpha 0.5, blue contour on top."""
    import cv2
    rng = np.random.default_rng(94704)
    out = image.astype(np.float32)
    for m in masks:
        color = rng.random(3) * 255
        out[m] = out[m] * 0.5 + color * 0.5
    out = out.astype(np.uint8)
    for m in masks:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        cv2.drawContours(out, contours, -1, (0, 0, 255), thickness=1)
    return Image.fromarray(out)


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU
    if not 0.1 <= args.granularity <= 1.0:
        print(f"warning: granularity {args.granularity} is outside the trained range [0.1, 1.0]")

    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = torch.device(args.device)
    if device.type == "cuda":
        # Notebook: bf16 autocast + TF32 on Ampere or newer.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()

    kwargs = dict(GENERATOR_KWARGS)
    for name in ("pred_iou_thresh", "stability_score_thresh", "points_per_side", "points_per_batch"):
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)

    ckpt = CHECKPOINTS[args.variant]
    image = np.array(Image.open(args.image).convert("RGB"))
    H, W = image.shape[:2]

    t0 = time.perf_counter()
    sam2 = build_sam2(MODEL_CFG, str(ckpt), device=device, apply_postprocessing=True)
    generator = SAM2AutomaticMaskGenerator(model=sam2, **kwargs)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    with autocast:
        anns = generator.generate(image, gra=args.granularity)
    t_gen = time.perf_counter() - t0

    anns.sort(key=lambda a: a["area"], reverse=True)
    masks = (np.stack([a["segmentation"] for a in anns]).astype(bool)
             if anns else np.zeros((0, H, W), dtype=bool))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "masks.npy", masks)
    records = [{
        "id": i,
        "area": int(a["area"]),
        "bbox_xywh": [float(v) for v in a["bbox"]],
        "predicted_iou": float(a["predicted_iou"]),
        "stability_score": float(a["stability_score"]),
        "point_xy": [float(v) for v in a["point_coords"][0]],
    } for i, a in enumerate(anns)]
    settings = {
        "image": str(args.image.resolve()),
        "image_hw": [H, W],
        "granularity": args.granularity,
        "variant": args.variant,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "model_cfg": MODEL_CFG,
        "build_sam2": {"apply_postprocessing": True},
        "generator": kwargs,
        "cascadepsp": False,
        "unsamv2_commit": repo_commit(),
        "device": args.device,
        "autocast_bf16": device.type == "cuda",
        "torch": torch.__version__,
        "n_masks": len(anns),
        "mask_order": "area descending",
        "seconds_model_load": round(t_load, 2),
        "seconds_generate": round(t_gen, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    json.dump({"settings": settings, "masks": records},
              open(args.out / "masks.json", "w"), indent=1)
    overlay(image, masks).save(args.out / "overlay.png")
    print(f"{len(anns)} masks  granularity {args.granularity}  variant {args.variant}  "
          f"device {args.device}  load {t_load:.1f}s  generate {t_gen:.1f}s  -> {args.out}")


if __name__ == "__main__":
    main()
