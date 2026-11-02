"""Extract + embed + save images from an additional dataset (clean only).

Streams frames via DataLoader, embeds with SigLIP, saves JPEGs.
Appends to existing images/manifest.csv and results/records.npz.

Usage:
    python vbti/research/how_to_mask_right/add_dataset.py \
        --repo_id eternalmay33/09_merged --n_frames 1000 --tag 09_merged
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from masking_lib import get_embeddings_batch, load_siglip

CAMERAS = ["top", "left", "right", "gripper"]
IMG_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"


def log_ram(label: str):
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1e9
    avail = psutil.virtual_memory().available / 1e9
    print(f"  [RAM] {label}: process={rss:.1f}GB, available={avail:.1f}GB")


def save_tensor_as_jpeg(img: torch.Tensor, path: Path, quality: int = 90):
    from PIL import Image
    arr = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=quality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=3000)
    parser.add_argument("--tag", type=str, required=True,
                        help="Short name for this dataset (e.g. '09_merged')")
    parser.add_argument("--loader_batch", type=int, default=128)
    parser.add_argument("--embed_batch", type=int, default=256)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id)
    total = len(dataset)
    step = max(total // args.n_frames, 1)
    indices = list(range(0, total, step))[:args.n_frames]

    print(f"Dataset: {args.repo_id} ({total} total frames)")
    print(f"Sampling {len(indices)} frames, tag='{args.tag}'")

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.loader_batch, num_workers=0)

    # Load SigLIP
    model, processor, device = load_siglip()
    log_ram("after SigLIP load")

    # Prepare output dirs
    IMG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    for cam in CAMERAS:
        (IMG_DIR / cam).mkdir(exist_ok=True)

    # Open CSV in append mode
    csv_path = IMG_DIR / "manifest.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "filepath", "frame_idx", "camera", "condition",
        "mask_shape", "fill_mode", "cos_sim", "dataset_source",
    ])
    if not csv_exists:
        writer.writeheader()

    all_embeddings = []
    all_meta = []
    n_saved = 0
    frame_offset = 0

    for batch in tqdm(loader, desc=f"Processing {args.tag}"):
        bs = batch[f"observation.images.{CAMERAS[0]}"].shape[0]
        batch_indices = indices[frame_offset:frame_offset + bs]
        frame_offset += bs

        imgs_to_embed = []
        meta = []

        for j in range(bs):
            dataset_idx = batch_indices[j]
            for cam in CAMERAS:
                img = batch[f"observation.images.{cam}"][j].clone()
                imgs_to_embed.append(img)
                meta.append({"frame_idx": dataset_idx, "camera": cam})

                # Save JPEG
                fname = f"{args.tag}_{dataset_idx:06d}_clean.jpg"
                fpath = IMG_DIR / cam / fname
                save_tensor_as_jpeg(img, fpath)

                writer.writerow({
                    "filepath": str(fpath.resolve()),
                    "frame_idx": dataset_idx,
                    "camera": cam,
                    "condition": "clean",
                    "mask_shape": "none",
                    "fill_mode": "none",
                    "cos_sim": "1.0",
                    "dataset_source": args.tag,
                })
                n_saved += 1

        # Embed batch
        embs = get_embeddings_batch(imgs_to_embed, model, processor, device,
                                    batch_size=args.embed_batch)
        del imgs_to_embed

        for i, m in enumerate(meta):
            all_embeddings.append(embs[i])
            all_meta.append(m)

        del batch
        gc.collect()

    csv_file.close()

    # Free model
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    log_ram("after processing")

    # Save embeddings
    new_embeddings = np.array(all_embeddings)
    new_frame_idx = np.array([m["frame_idx"] for m in all_meta])
    new_cameras = np.array([m["camera"] for m in all_meta])
    new_conditions = np.array(["clean"] * len(all_meta))

    # Merge with existing records if present
    records_path = RESULTS_DIR / "records.npz"
    if records_path.exists():
        print("Merging with existing records.npz...")
        old = np.load(records_path, allow_pickle=False)
        merged_emb = np.concatenate([old["embeddings"], new_embeddings])
        merged_cos = np.concatenate([old["cos_sims"],
                                     np.ones(len(new_embeddings))])
        merged_fi = np.concatenate([old["frame_idx"], new_frame_idx])
        merged_cam = np.concatenate([old["cameras"], new_cameras])
        merged_cond = np.concatenate([old["conditions"], new_conditions])
        old.close()
    else:
        merged_emb = new_embeddings
        merged_cos = np.ones(len(new_embeddings))
        merged_fi = new_frame_idx
        merged_cam = new_cameras
        merged_cond = new_conditions

    np.savez_compressed(
        records_path,
        embeddings=merged_emb,
        cos_sims=merged_cos,
        frame_idx=merged_fi,
        cameras=merged_cam,
        conditions=merged_cond,
    )

    print(f"\nSaved {n_saved} images to {IMG_DIR}")
    print(f"Merged embeddings: {merged_emb.shape[0]} total in {records_path}")
    print("Done.")


if __name__ == "__main__":
    main()
