#!/usr/bin/env python3
"""Subsample a nerfstudio/COLMAP dataset by taking every Nth frame."""
import argparse
import json
import shutil
from pathlib import Path


def subsample_dataset(src_dir: Path, dst_dir: Path, step: int = 4, copy_images: bool = False):
    """Create a reduced dataset with every Nth frame.

    Args:
        src_dir: Source dataset directory (must contain transforms.json)
        dst_dir: Destination directory for reduced dataset
        step: Take every Nth frame (default: 4)
        copy_images: If True, copy images. If False, create symlinks.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # Read original transforms
    with open(src_dir / "transforms.json") as f:
        data = json.load(f)

    original_count = len(data['frames'])
    print(f"Original frames: {original_count}")

    # Filter to every Nth frame
    subset_frames = data['frames'][::step]
    print(f"Subset frames: {len(subset_frames)} (every {step}th)")

    # Create output directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Create new transforms with subset
    subset_data = {k: v for k, v in data.items() if k != 'frames'}
    subset_data['frames'] = subset_frames

    # Write new transforms.json
    with open(dst_dir / "transforms.json", 'w') as f:
        json.dump(subset_data, f, indent=4)
    print(f"Created: {dst_dir / 'transforms.json'}")

    # Create images directory and link/copy images
    images_dir = dst_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for frame in subset_frames:
        src_img = src_dir / frame['file_path']
        dst_img = dst_dir / frame['file_path']

        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()

        if copy_images:
            shutil.copy(src_img, dst_img)
        else:
            dst_img.symlink_to(src_img.resolve())

    action = "Copied" if copy_images else "Symlinked"
    print(f"{action} {len(subset_frames)} images to {images_dir}")

    # Copy sparse_pc.ply if exists
    sparse_pc = src_dir / "sparse_pc.ply"
    if sparse_pc.exists():
        shutil.copy(sparse_pc, dst_dir / "sparse_pc.ply")
        print("Copied sparse_pc.ply")

    print(f"\nDone! Dataset ready at: {dst_dir}")
    return dst_dir


def main():
    parser = argparse.ArgumentParser(description="Subsample a nerfstudio dataset")
    parser.add_argument("src", type=Path, help="Source dataset directory")
    parser.add_argument("dst", type=Path, help="Destination directory")
    parser.add_argument("-n", "--step", type=int, default=4, help="Take every Nth frame (default: 4)")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinks")

    args = parser.parse_args()
    subsample_dataset(args.src, args.dst, args.step, args.copy)


if __name__ == "__main__":
    main()
