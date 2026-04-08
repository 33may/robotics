"""Save clean + masked images to disk for FiftyOne exploration.

Streams shards one at a time — peak RAM = 1 shard (~1GB).
Outputs JPEG images + a CSV manifest with all metadata.

Usage:
    python vbti/research/how_to_mask_right/save_images.py [--n_masked 500]
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from masking_lib import FILL_MODES, SHAPES

BLUR_K = 51
MASK_SEED = 42

CACHE_DIR = Path(__file__).parent / "cached_frames"
IMG_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"


def save_tensor_as_jpeg(img: torch.Tensor, path: Path, quality: int = 90):
    """Save (C, H, W) float [0,1] tensor as JPEG."""
    from PIL import Image
    arr = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=quality)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_masked", type=int, default=500)
    args = parser.parse_args()

    with open(CACHE_DIR / "manifest.json") as f:
        manifest = json.load(f)

    cameras = manifest["cameras"]
    all_indices = manifest["frame_indices"]
    n_shards = manifest["n_shards"]

    # Pick masked frames (same logic as run_masking_analysis.py)
    masked_step = max(len(all_indices) // args.n_masked, 1)
    masked_indices_set = set(all_indices[::masked_step][:args.n_masked])

    # Prepare output dirs
    IMG_DIR.mkdir(exist_ok=True)
    for cam in cameras:
        (IMG_DIR / cam).mkdir(exist_ok=True)

    # Load precomputed embeddings if available (to include cos_sim in manifest)
    cos_sim_lookup = {}
    records_path = RESULTS_DIR / "records.npz"
    if records_path.exists():
        data = np.load(records_path, allow_pickle=False)
        frame_idx = data["frame_idx"].copy()
        cameras_arr = data["cameras"].copy()
        conditions_arr = data["conditions"].copy()
        cos_sims = data["cos_sims"].copy()
        data.close()
        for i in range(len(conditions_arr)):
            key = (int(frame_idx[i]), str(cameras_arr[i]), str(conditions_arr[i]))
            cos_sim_lookup[key] = float(cos_sims[i])
        del frame_idx, cameras_arr, conditions_arr, cos_sims
        print(f"Loaded {len(cos_sim_lookup)} cos_sim values from records.npz")

    # CSV manifest for FiftyOne
    csv_path = IMG_DIR / "manifest.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "filepath", "frame_idx", "camera", "condition",
        "mask_shape", "fill_mode", "cos_sim", "dataset_source",
    ])
    writer.writeheader()

    n_saved = 0
    for shard_i in tqdm(range(n_shards), desc="Saving images"):
        shard = torch.load(CACHE_DIR / f"shard_{shard_i:04d}.pt",
                           weights_only=False)

        for dataset_idx, cam_dict in shard.items():
            should_mask = dataset_idx in masked_indices_set

            for cam_idx, cam in enumerate(cameras):
                img = cam_dict[cam]

                # Save clean image
                clean_name = f"{dataset_idx:06d}_clean.jpg"
                clean_path = IMG_DIR / cam / clean_name
                save_tensor_as_jpeg(img, clean_path)
                cos = cos_sim_lookup.get((dataset_idx, cam, "clean"), 1.0)
                writer.writerow({
                    "filepath": str(clean_path.resolve()),
                    "frame_idx": dataset_idx,
                    "camera": cam,
                    "condition": "clean",
                    "mask_shape": "none",
                    "fill_mode": "none",
                    "cos_sim": f"{cos:.4f}",
                    "dataset_source": "01_02_black",
                })
                n_saved += 1

                # Save masked variants
                if should_mask:
                    for shape_name, shape_fn in SHAPES.items():
                        for fill in FILL_MODES:
                            torch.manual_seed(MASK_SEED + dataset_idx + cam_idx)
                            masked_img = shape_fn(img, fill, BLUR_K)

                            fname = f"{dataset_idx:06d}_{shape_name}_{fill}.jpg"
                            fpath = IMG_DIR / cam / fname
                            save_tensor_as_jpeg(masked_img, fpath)

                            cond = f"{shape_name}_{fill}"
                            cos = cos_sim_lookup.get(
                                (dataset_idx, cam, cond), -1.0)
                            writer.writerow({
                                "filepath": str(fpath.resolve()),
                                "frame_idx": dataset_idx,
                                "camera": cam,
                                "condition": cond,
                                "mask_shape": shape_name,
                                "fill_mode": fill,
                                "cos_sim": f"{cos:.4f}",
                            "dataset_source": "01_02_black",
                            })
                            n_saved += 1
                            del masked_img

        del shard
        gc.collect()

    csv_file.close()
    print(f"\nSaved {n_saved} images to {IMG_DIR}")
    print(f"Manifest: {csv_path}")


if __name__ == "__main__":
    main()
