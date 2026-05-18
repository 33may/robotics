"""Bake teacher visual features into a LeRobot dataset for UVA aux-loss training.

Adds a new column `observation.video_features.{layer}_{S}x{S}` containing fp16
tensors of shape (S, S, feat_dim) per row, where S = --spatial-size and feat_dim
comes from the teacher's SigLIP last layer.

Usage:
  python -m vbti.logic.dataset.add_video_features \
      --dataset eternalmay33/06_black_cup_red_bg_depth \
      --teacher vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model \
      --layer siglip_output \
      --spatial-size 4 \
      --target-camera observation.images.wrist
"""
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import to trigger registry registration of siglip_output
import vbti.logic.dataset.target_extractors.siglip_output  # noqa: F401
from vbti.logic.dataset.target_extractors import get as get_extractor


def build_feature_key(layer: str, spatial_size: int) -> str:
    """Self-describing column name: 'observation.video_features.{layer}_{S}x{S}'."""
    return f"observation.video_features.{layer}_{spatial_size}x{spatial_size}"


def _preprocess_image_for_teacher(img: torch.Tensor) -> torch.Tensor:
    """SmolVLA's image preprocessing: tensor in [0,1] -> [-1, 1].

    Caller must already have resized to teacher's expected input size.
    """
    return img * 2.0 - 1.0


def main():
    p = argparse.ArgumentParser(description="Bake video features into a LeRobot dataset")
    p.add_argument("--dataset", required=True, help="Local path or HF hub repo_id")
    p.add_argument("--teacher", required=True, help="Path to teacher SmolVLAPolicy checkpoint")
    p.add_argument("--layer", default="siglip_output", help="Extractor registry name")
    p.add_argument("--spatial-size", type=int, default=4)
    p.add_argument("--target-camera", default="observation.images.wrist")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--root", default=None, help="LeRobot dataset cache root")
    p.add_argument("--force", action="store_true", help="Overwrite existing column")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("bake")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"loading teacher from {args.teacher}")
    teacher = SmolVLAPolicy.from_pretrained(args.teacher).to(device).eval()

    log.info(f"opening dataset {args.dataset}")
    dataset = LeRobotDataset(args.dataset, root=args.root)

    feature_key = build_feature_key(args.layer, args.spatial_size)
    log.info(f"feature column: {feature_key}")

    if feature_key in dataset.features and not args.force:
        raise RuntimeError(
            f"Column '{feature_key}' already exists. Pass --force to overwrite."
        )

    extractor = get_extractor(args.layer)

    n_frames = len(dataset)
    log.info(f"baking {n_frames} frames at spatial_size={args.spatial_size}")

    out_chunks = []
    for start in tqdm(range(0, n_frames, args.batch_size), unit="batch"):
        end = min(start + args.batch_size, n_frames)
        imgs = []
        for idx in range(start, end):
            sample = dataset[idx]
            img = sample[args.target_camera]  # (3, H, W) or (T, 3, H, W) float in [0, 1]
            if img.ndim == 4:  # (T, 3, H, W) — take t=0
                img = img[0]
            imgs.append(img)
        imgs_t = torch.stack(imgs, dim=0).to(device)
        imgs_t = _preprocess_image_for_teacher(imgs_t)

        feats = extractor(teacher, imgs_t, spatial_size=args.spatial_size)
        if args.dtype == "fp16":
            feats = feats.half()
        out_chunks.append(feats.cpu())

    out = torch.cat(out_chunks, dim=0)
    log.info(f"baked tensor shape={tuple(out.shape)} dtype={out.dtype}")

    _persist_column(dataset, feature_key, out)
    log.info("done")


def _persist_column(dataset: LeRobotDataset, key: str, tensor: torch.Tensor):
    """Write the baked tensor as a new column in the underlying HF Arrow dataset.

    The column is stored as nested lists (HF datasets / parquet native format);
    the LeRobot dataloader will wrap it back to tensor on access.

    Output dir: {dataset.root}/data/hf_dataset_with_uva — sibling cache, not
    a destructive overwrite of the original parquet chunks.
    """
    hf = dataset.hf_dataset
    new_col = [tensor[i].tolist() for i in range(len(tensor))]
    hf_new = hf.add_column(key, new_col)
    out_dir = Path(dataset.root) / "data" / "hf_dataset_with_uva"
    hf_new.save_to_disk(str(out_dir))


if __name__ == "__main__":
    main()
