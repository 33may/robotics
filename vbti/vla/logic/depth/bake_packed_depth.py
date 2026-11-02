"""Bake turbo-RGB depth into a LeRobot v3.0 dataset, overwriting packed-PNG.

Use case: lerobot-train ships unmodified to remote and gets fed a regular RGB
image for ``observation.images.gripper_depth`` instead of packed uint16 bytes.
The frozen SigLIP encoder sees a smooth turbo image whose color encodes
distance.

What it does, per data parquet:
  1. Read the parquet with HF datasets (preserves Image-typed PNG bytes).
  2. For each row:
       a. Decode the PNG (packed: R=high_byte, G=low_byte of uint16 depth).
       b. Reconstruct uint16 metric depth (× depth_scale → meters).
       c. Clip to [clip_min_m, clip_max_m] → linear normalize → turbo colormap.
       d. Re-encode the resulting RGB as a fresh PNG.
  3. Write the parquet back with HF Image typing intact.

Stats are NOT recomputed. Training config uses ImageNet stats for vision
inputs (``use_imagenet_stats: true``) so the per-feature dataset stats for
``gripper_depth`` are unused. The leftover stale stats remain harmless.

Usage:
    conda run -n lerobot python -m vbti.logic.depth.bake_packed_depth \\
        --repo_id=eternalmay33/04_05_06_07_merged_may-sim_depth \\
        --depth-key=observation.images.gripper_depth \\
        --clip-min-m=0.05 --clip-max-m=0.20 --overwrite
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.utils import (
    get_hf_features_from_features,
    to_parquet_with_hf_images,
)
from vbti.logic.depth.colorize import colorize_fixed_clip, unpack_rgb_to_uint16


LEROBOT_CACHE = Path("~/.cache/huggingface/lerobot").expanduser()


def _resolve_root(repo_id: str, root: str | None = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    return LEROBOT_CACHE / repo_id


def _bytes_to_uint8_rgb(b: bytes) -> np.ndarray:
    """PNG bytes → (H, W, 3) uint8."""
    return np.array(Image.open(io.BytesIO(b)))


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """(H, W, 3) uint8 → PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _bake_one_parquet(
    src_path: Path,
    dst_path: Path,
    depth_key: str,
    clip_min_m: float,
    clip_max_m: float,
    depth_scale_m: float,
    hf_features: datasets.Features,
) -> int:
    """Bake one parquet; returns row count."""
    src_ds = datasets.Dataset.from_parquet(str(src_path))
    df = src_ds.to_pandas()

    if depth_key not in df.columns:
        sys.exit(f"depth_key '{depth_key}' not found in {src_path}")

    # Re-encode each row's depth column.
    new_col = []
    for row in df[depth_key]:
        b = row["bytes"] if isinstance(row, dict) else row
        packed = _bytes_to_uint8_rgb(b)              # (H, W, 3) uint8 — packed
        d_u16 = unpack_rgb_to_uint16(packed)         # (H, W) uint16
        d_m = d_u16.astype(np.float32) * depth_scale_m
        baked = colorize_fixed_clip(d_m, clip_min_m, clip_max_m)  # (H, W, 3) uint8 — turbo RGB
        new_bytes = _rgb_to_png_bytes(baked)
        new_col.append({"bytes": new_bytes, "path": None})

    df[depth_key] = new_col

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    to_parquet_with_hf_images(df, dst_path, features=hf_features)
    return len(df)


def bake(
    repo_id: str,
    depth_key: str = "observation.images.gripper_depth",
    root: str | None = None,
    out_repo_id: str | None = None,
    out_root: str | None = None,
    clip_min_m: float = 0.05,
    clip_max_m: float = 0.20,
    depth_scale_m: float = 1e-4,
    overwrite: bool = False,
    workers: int = 1,
) -> Path:
    """Bake turbo-RGB depth into the dataset's data parquets."""
    src = _resolve_root(repo_id, root)
    if not src.exists():
        sys.exit(f"src does not exist: {src}")

    in_place = out_repo_id is None
    if in_place:
        # Stage to a temp dir alongside the source, then atomic move.
        tmp = src.parent / f"{src.name}.bake_tmp"
        if tmp.exists():
            print(f"[wipe stale tmp] {tmp}")
            shutil.rmtree(tmp)
        dst = tmp
    else:
        dst = _resolve_root(out_repo_id, out_root)
        if dst.exists():
            if overwrite:
                print(f"[wipe] {dst}")
                shutil.rmtree(dst)
            else:
                sys.exit(f"dst exists: {dst}  (pass --overwrite to wipe)")

    print(f"[bake] {repo_id}")
    print(f"   depth_key={depth_key}  clip=[{clip_min_m:.3f}, {clip_max_m:.3f}] m  scale={depth_scale_m}")
    print(f"   src={src}")
    print(f"   dst={dst}{'  (in-place via temp + move)' if in_place else ''}")

    # ── 1. Load src metadata via LeRobotDatasetMetadata so shapes are normalized
    #       to tuples (raw info.json has shapes as lists which trips
    #       ``get_hf_features_from_features``'s ``shape == (1,)`` check). ──
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo_id, root=str(src))
    if depth_key not in meta.features:
        sys.exit(f"depth_key '{depth_key}' not in features:\n  " + "\n  ".join(meta.features))
    if meta.features[depth_key]["dtype"] != "image":
        sys.exit(f"depth_key dtype must be 'image', got {meta.features[depth_key]['dtype']}")

    hf_features = get_hf_features_from_features(meta.features)

    # ── 2. Copy non-data tree (videos, meta, tasks) verbatim ──
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("videos", "meta"):
        s = src / sub
        if s.exists():
            print(f"[copy] {sub}/")
            shutil.copytree(s, dst / sub, dirs_exist_ok=True)

    # ── 3. Bake each data parquet ──
    src_data = src / "data"
    dst_data = dst / "data"
    parquets = sorted(src_data.rglob("file-*.parquet"))
    print(f"[bake] {len(parquets)} data parquets, workers={workers}")

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for p in parquets:
                rel = p.relative_to(src_data)
                jobs.append(pool.submit(
                    _bake_one_parquet,
                    p, dst_data / rel, depth_key,
                    clip_min_m, clip_max_m, depth_scale_m, hf_features,
                ))
            total = 0
            for fut in tqdm(as_completed(jobs), total=len(jobs), desc="parquets"):
                total += fut.result()
    else:
        total = 0
        for p in tqdm(parquets, desc="parquets"):
            rel = p.relative_to(src_data)
            n = _bake_one_parquet(
                p, dst_data / rel, depth_key, clip_min_m, clip_max_m, depth_scale_m, hf_features
            )
            total += n
    print(f"[bake] {total} frames re-encoded")

    # ── 4. In-place: swap dirs ──
    if in_place:
        backup = src.parent / f"{src.name}.old"
        if backup.exists():
            shutil.rmtree(backup)
        print(f"[swap] {src}  →  {backup}")
        src.rename(backup)
        print(f"[swap] {dst}  →  {src}")
        dst.rename(src)
        print(f"[done] in-place; old version preserved at {backup}")
        print(f"       remove with: rm -rf {backup}")
        return src

    print(f"[done] {dst}")
    return dst


def _verify(dst: Path, depth_key: str) -> None:
    """Decode one frame from the baked dataset and check it's turbo RGB, not packed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata.create  # noqa: just to ensure import works
    repo_id = "/".join(dst.parts[-2:])
    ds = LeRobotDataset(repo_id, root=str(dst))
    item = ds[0]
    t = item[depth_key]
    arr = (t.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)  # (3, H, W)
    # Quick heuristic: turbo RGB has well-mixed R/G/B; packed has dominant G with cycles.
    means = arr.mean(axis=(1, 2))  # (3,)
    print(f"[verify] depth tensor shape={tuple(t.shape)}  channel-means(R,G,B)={means}")
    print(f"[verify] frame 0 keys: {sorted(item.keys())}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-repo-id", default=None,
                    help="If omitted, overwrite repo_id in place via temp+move.")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--depth-key", default="observation.images.gripper_depth")
    ap.add_argument("--clip-min-m", type=float, default=0.05)
    ap.add_argument("--clip-max-m", type=float, default=0.20)
    ap.add_argument("--depth-scale-m", type=float, default=1e-4)
    ap.add_argument("--overwrite", action="store_true",
                    help="If --out-repo-id is given and points to existing dir, wipe it.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel processes for per-parquet baking.")
    args = ap.parse_args(argv)

    dst = bake(
        repo_id=args.repo_id,
        depth_key=args.depth_key,
        root=args.root,
        out_repo_id=args.out_repo_id,
        out_root=args.out_root,
        clip_min_m=args.clip_min_m,
        clip_max_m=args.clip_max_m,
        depth_scale_m=args.depth_scale_m,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    _verify(dst, args.depth_key)


if __name__ == "__main__":
    main()
