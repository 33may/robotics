"""Add a constant zero-RGB ``observation.images.gripper_depth`` to a no-depth dataset.

Use case: the depth corpus is turbo-baked RGB. To merge no-depth episodes with the
turbo-baked depth dataset, the no-depth dataset must gain an image-dtype
``gripper_depth`` feature with the same dtype/shape. Every frame gets the same
all-zero PNG (RGB 0,0,0) — visually distinct from any turbo color, so the model
learns a clean "no depth signal" pattern.

What it writes (mirror of strip_feature.py in reverse):
  - data/chunk-*/file-*.parquet     → add the column with HF Image-typed PNG bytes
  - meta/info.json                  → add features[<key>] entry
  - meta/stats.json                 → add identity stats for <key>
  - meta/episodes/chunk-*/file-*.parquet
                                    → add stats/<key>/* columns (zero / identity)

Stats: min=0, max=1, mean=0, std=1 across the 3 channels — identity normalization.
v018-style config sets ``use_imagenet_stats: true`` for vision so these are unused;
they only need to be valid shape.

Usage:
    conda run -n lerobot python -m vbti.logic.dataset.add_zero_depth_turbo \\
        --src=eternalmay33/01_02_03_merged_may-sim \\
        --dst=eternalmay33/01_02_03_merged_may-sim_padded \\
        --reference=eternalmay33/04_red_cup_depth \\
        --feature=observation.images.gripper_depth \\
        --overwrite
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


LEROBOT_CACHE = Path("~/.cache/huggingface/lerobot").expanduser()


def _resolve_root(repo_id: str, root: str | None = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    return LEROBOT_CACHE / repo_id


def _zero_png_bytes(height: int, width: int) -> bytes:
    """Return PNG bytes for a black (H, W, 3) uint8 image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _add_data_column(
    src_path: Path,
    dst_path: Path,
    feature_key: str,
    png_bytes: bytes,
) -> int:
    """Add an HF Image-typed column ``feature_key`` filled with the same PNG.

    Uses the HF datasets library to write a parquet that LeRobot's dataset
    loader (which expects the {bytes, path} struct) can read back natively.
    Returns row count.
    """
    import datasets
    src_ds: datasets.Dataset = datasets.Dataset.from_parquet(str(src_path))  # type: ignore[assignment]
    n = src_ds.num_rows
    # Build the column as a list of dicts (HF Image-compatible)
    col = [{"bytes": png_bytes, "path": None}] * n
    new_ds = src_ds.add_column(feature_key, col)  # type: ignore[arg-type]
    # Cast the new column to Image type so HF writes the proper struct schema
    feats = dict(new_ds.features)
    feats[feature_key] = datasets.Image()
    new_ds = new_ds.cast(datasets.Features(feats))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    new_ds.to_parquet(str(dst_path))
    return n


def _add_episodes_stats_columns(
    src_path: Path,
    dst_path: Path,
    feature_key: str,
    reference_eps_path: Path,
) -> None:
    """Add stats/<feature_key>/* columns to the episodes parquet using pyarrow
    to preserve the nested ``list<list<list<double>>>`` schema.

    Reference schema (from real _depth datasets):
      stats/<key>/min..q99: list<list<list<double>>>  → [[[c0]], [[c1]], [[c2]]]
      stats/<key>/count:    list<int64>               → [N]
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_table = pq.read_table(src_path)
    n_rows = src_table.num_rows

    # Read reference schema for the depth stats columns to copy exactly.
    ref_schema = pq.read_schema(reference_eps_path)
    add_fields = {}
    for f in ref_schema:
        if f.name.startswith(f"stats/{feature_key}/"):
            add_fields[f.name] = f.type

    if not add_fields:
        sys.exit(f"reference has no stats/{feature_key}/* cols")

    # Build per-row values as plain Python lists (pyarrow infers list types fine).
    triple_zero = [[[0.0]], [[0.0]], [[0.0]]]
    triple_one = [[[1.0]], [[1.0]], [[1.0]]]
    count_zero = [0]

    for col_name, col_type in add_fields.items():
        suffix = col_name[len(f"stats/{feature_key}/"):]
        if suffix == "count":
            values = [count_zero] * n_rows
        elif suffix in ("max", "std"):
            values = [triple_one] * n_rows
        else:
            values = [triple_zero] * n_rows
        new_col = pa.array(values, type=col_type)
        src_table = src_table.append_column(col_name, new_col)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(src_table, dst_path)


def _patch_info_json(
    src_path: Path,
    dst_path: Path,
    feature_key: str,
    height: int,
    width: int,
) -> None:
    info = json.loads(src_path.read_text())
    info.setdefault("features", {})[feature_key] = {
        "dtype": "image",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(info, indent=2))


def _patch_stats_json(
    src_path: Path | None,
    dst_path: Path,
    feature_key: str,
) -> None:
    stats = {}
    if src_path is not None and src_path.exists():
        stats = json.loads(src_path.read_text())
    # Identity stats: min=0, max=1, mean=0, std=1 per channel.
    stats[feature_key] = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.0]], [[0.0]], [[0.0]]],
        "std": [[[1.0]], [[1.0]], [[1.0]]],
        "count": [0],
        "q01": [[[0.0]], [[0.0]], [[0.0]]],
        "q10": [[[0.0]], [[0.0]], [[0.0]]],
        "q50": [[[0.0]], [[0.0]], [[0.0]]],
        "q90": [[[0.0]], [[0.0]], [[0.0]]],
        "q99": [[[0.0]], [[0.0]], [[0.0]]],
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(stats, indent=2))


def add_zero_depth_turbo(
    src_repo_id: str,
    dst_repo_id: str,
    feature_key: str,
    reference_repo_id: str,
    src_root: str | None = None,
    dst_root: str | None = None,
    reference_root: str | None = None,
    overwrite: bool = False,
    height: int = 480,
    width: int = 640,
) -> Path:
    src = _resolve_root(src_repo_id, src_root)
    dst = _resolve_root(dst_repo_id, dst_root)
    ref = _resolve_root(reference_repo_id, reference_root)

    if not src.exists():
        sys.exit(f"src does not exist: {src}")
    if not ref.exists():
        sys.exit(f"reference does not exist: {ref}")
    if dst.exists():
        if overwrite:
            print(f"[wipe] {dst}")
            shutil.rmtree(dst)
        else:
            sys.exit(f"dst exists: {dst}  (pass --overwrite to wipe)")

    # Pre-flight: feature must NOT already be in src, must be in reference
    src_info = json.loads((src / "meta" / "info.json").read_text())
    if feature_key in src_info.get("features", {}):
        sys.exit(f"feature '{feature_key}' already in src — use a no-depth source")
    ref_info = json.loads((ref / "meta" / "info.json").read_text())
    if feature_key not in ref_info.get("features", {}):
        sys.exit(f"feature '{feature_key}' not in reference {reference_repo_id}")

    # Extract H, W from reference if available
    ref_feat = ref_info["features"][feature_key]
    if "shape" in ref_feat and len(ref_feat["shape"]) >= 2:
        height, width = int(ref_feat["shape"][0]), int(ref_feat["shape"][1])
    print(f"[add] {feature_key} dtype=image shape=({height},{width},3) → {dst_repo_id}")

    # 1. tasks.parquet — copy verbatim
    src_tasks = src / "meta" / "tasks.parquet"
    dst_tasks = dst / "meta" / "tasks.parquet"
    dst_tasks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_tasks, dst_tasks)

    # 2. videos/ — copy verbatim
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    if src_videos.exists():
        dst_videos.mkdir(parents=True, exist_ok=True)
        for sub in sorted(src_videos.iterdir()):
            print(f"[copy] videos/{sub.name}")
            shutil.copytree(sub, dst_videos / sub.name)

    # 3. data parquets — add the new column
    png_bytes = _zero_png_bytes(height, width)
    src_data = src / "data"
    dst_data = dst / "data"
    parquets = sorted(src_data.rglob("file-*.parquet"))
    print(f"[add] {len(parquets)} data parquets")
    total = 0
    for p in tqdm(parquets, desc="data parquets"):
        rel = p.relative_to(src_data)
        n = _add_data_column(p, dst_data / rel, feature_key, png_bytes)
        total += n
    print(f"[add] {total} rows total")

    # 4. meta/episodes parquets — add stats columns from a reference shape
    src_eps = src / "meta" / "episodes"
    dst_eps = dst / "meta" / "episodes"
    ref_eps_files = sorted((ref / "meta" / "episodes").rglob("file-*.parquet"))
    if not ref_eps_files:
        sys.exit("reference has no meta/episodes parquets")
    ref_eps_path = ref_eps_files[0]
    eps = sorted(src_eps.rglob("file-*.parquet"))
    print(f"[add] {len(eps)} meta/episodes parquets")
    for p in tqdm(eps, desc="episodes parquets"):
        rel = p.relative_to(src_eps)
        _add_episodes_stats_columns(p, dst_eps / rel, feature_key, ref_eps_path)

    # 5. info.json + stats.json
    _patch_info_json(
        src / "meta" / "info.json",
        dst / "meta" / "info.json",
        feature_key,
        height,
        width,
    )
    src_stats_json = src / "meta" / "stats.json"
    _patch_stats_json(
        src_stats_json if src_stats_json.exists() else None,
        dst / "meta" / "stats.json",
        feature_key,
    )

    print(f"\n[done] {dst}")
    return dst


def _verify(dst_repo_id: str, dst_root: str | None, feature_key: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = _resolve_root(dst_repo_id, dst_root)
    meta = LeRobotDatasetMetadata(dst_repo_id, root=str(root))
    assert feature_key in meta.features, f"feature missing in features dict: {feature_key}"
    print(f"[verify] features keys ({len(meta.features)}): {sorted(meta.features)}")
    print(f"[verify] total_episodes={meta.total_episodes}  total_frames={meta.total_frames}")

    ds = LeRobotDataset(dst_repo_id, root=str(root))
    item = ds[0]
    matched = [k for k in item.keys() if feature_key in k]
    if not matched:
        sys.exit(f"feature did not surface in __getitem__: {sorted(item.keys())}")
    print(f"[verify] frame 0 loaded ok, {feature_key} present, dtype={item[feature_key].dtype} shape={item[feature_key].shape}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dst-root", default=None)
    ap.add_argument("--reference", required=True, help="repo_id of a dataset that already has the feature, used to copy stats schema")
    ap.add_argument("--reference-root", default=None)
    ap.add_argument("--feature", default="observation.images.gripper_depth")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    add_zero_depth_turbo(
        src_repo_id=args.src,
        dst_repo_id=args.dst,
        feature_key=args.feature,
        reference_repo_id=args.reference,
        src_root=args.src_root,
        dst_root=args.dst_root,
        reference_root=args.reference_root,
        overwrite=args.overwrite,
        height=args.height,
        width=args.width,
    )
    if not args.no_verify:
        _verify(args.dst, args.dst_root, args.feature)


if __name__ == "__main__":
    main()
