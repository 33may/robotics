"""Extract frames from LeRobot dataset into sharded .pt files.

Each shard holds one DataLoader batch (~256 frames × 4 cameras).
Never more than one batch in RAM at a time.

Usage:
    python vbti/research/how_to_mask_right/extract_frames.py \
        --n_frames 3000 --batch_size 256
"""
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "eternalmay33/01_02_black_merged"
CAMERAS = ["top", "left", "right", "gripper"]
OUT_DIR = Path(__file__).parent / "cached_frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = LeRobotDataset(REPO_ID)
    step = max(len(dataset) // args.n_frames, 1)
    indices = list(range(0, len(dataset), step))[:args.n_frames]

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=0)

    OUT_DIR.mkdir(exist_ok=True)

    # Clean old shards
    for old in OUT_DIR.glob("shard_*.pt"):
        old.unlink()

    print(f"Extracting {len(indices)} frames from {REPO_ID} "
          f"(batch_size={args.batch_size}, 4 workers)...")

    shard_idx = 0
    frame_offset = 0
    for batch in tqdm(loader, desc="Extracting"):
        bs = batch[f"observation.images.{CAMERAS[0]}"].shape[0]
        batch_indices = indices[frame_offset:frame_offset + bs]
        frame_offset += bs

        shard = {}
        for j in range(bs):
            cam_dict = {}
            for cam in CAMERAS:
                # .clone() detaches from batch tensor so GC can free it
                cam_dict[cam] = batch[f"observation.images.{cam}"][j].clone()
            shard[batch_indices[j]] = cam_dict

        shard_path = OUT_DIR / f"shard_{shard_idx:04d}.pt"
        torch.save(shard, shard_path)
        del shard, batch
        shard_idx += 1

    # Save manifest
    manifest = {
        "repo_id": REPO_ID,
        "cameras": CAMERAS,
        "n_frames": len(indices),
        "n_shards": shard_idx,
        "frame_indices": indices,
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("shard_*.pt")) / 1e6
    print(f"Saved {shard_idx} shards ({total_mb:.0f} MB total) + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
