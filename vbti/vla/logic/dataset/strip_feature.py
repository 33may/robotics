"""Strip a feature from a LeRobot v3.0 dataset, producing a new dataset.

Use case: starting from a merged dataset that has both RGB and depth, drop the
depth feature for a control twin used by an A/B training run. The two datasets
end up with byte-identical RGB videos, identical episode/frame indexing, and
identical RGB stats — only the dropped feature column is gone.

What gets stripped:
  - data/chunk-*/file-*.parquet     → drop the column
  - meta/info.json                  → drop the features[<key>] entry
  - meta/stats.json                 → drop the <key> entry
  - meta/episodes/chunk-*/file-*.parquet
                                    → drop columns matching ``stats/<key>/*``
                                      and ``videos/<key>/*`` if present
  - videos/<key>/                   → removed (only present when the dropped
                                      feature is video-dtype; image-dtype keys
                                      have no video tree)

What is preserved byte-identical:
  - tasks.parquet
  - all videos/<other_key>/ trees
  - data parquet rows other than the dropped column
  - RGB stats in info.json/stats.json/meta/episodes parquet

Usage:
    conda run -n lerobot python -m vbti.logic.dataset.strip_feature \\
        --src=eternalmay33/04_05_06_07_merged_may-sim_depth \\
        --dst=eternalmay33/04_05_06_07_merged_may-sim \\
        --feature=observation.images.gripper_depth
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


LEROBOT_CACHE = Path("~/.cache/huggingface/lerobot").expanduser()


def _resolve_root(repo_id: str, root: str | None = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    return LEROBOT_CACHE / repo_id


def _strip_data_parquet(src_path: Path, dst_path: Path, feature_key: str) -> None:
    """Drop ``feature_key`` column from a single data parquet file.

    For image-dtype features the column carries HF Image-typed PNG bytes — once
    dropped, there are no Image columns left, so a vanilla pandas write is fine.
    """
    df = pd.read_parquet(src_path)
    if feature_key in df.columns:
        df = df.drop(columns=[feature_key])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path)


def _strip_episodes_parquet(src_path: Path, dst_path: Path, feature_key: str) -> None:
    """Drop ``stats/<feature_key>/*`` and ``videos/<feature_key>/*`` columns."""
    df = pd.read_parquet(src_path)
    drop_prefixes = [f"stats/{feature_key}/", f"videos/{feature_key}/"]
    drop = [c for c in df.columns if any(c.startswith(p) for p in drop_prefixes)]
    if drop:
        df = df.drop(columns=drop)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path)


def _strip_info_json(src_path: Path, dst_path: Path, feature_key: str) -> dict:
    info = json.loads(src_path.read_text())
    if feature_key in info.get("features", {}):
        del info["features"][feature_key]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(info, indent=2))
    return info


def _strip_stats_json(src_path: Path, dst_path: Path, feature_key: str) -> None:
    stats = json.loads(src_path.read_text())
    if feature_key in stats:
        del stats[feature_key]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(stats, indent=2))


def strip_feature(
    src_repo_id: str,
    dst_repo_id: str,
    feature_key: str,
    src_root: str | None = None,
    dst_root: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Strip ``feature_key`` from ``src_repo_id`` → write a new dataset at ``dst_repo_id``."""
    src = _resolve_root(src_repo_id, src_root)
    dst = _resolve_root(dst_repo_id, dst_root)

    if not src.exists():
        sys.exit(f"src does not exist: {src}")
    if dst.exists():
        if overwrite:
            print(f"[wipe] {dst}")
            shutil.rmtree(dst)
        else:
            sys.exit(f"dst exists: {dst}  (pass --overwrite to wipe)")

    # ── pre-flight: feature must exist in src ──
    src_info = json.loads((src / "meta" / "info.json").read_text())
    if feature_key not in src_info.get("features", {}):
        sys.exit(
            f"feature '{feature_key}' not in src features:\n  "
            + "\n  ".join(src_info.get("features", {}).keys())
        )
    feature_dtype = src_info["features"][feature_key]["dtype"]
    print(f"[strip] {feature_key} (dtype={feature_dtype}) from {src_repo_id}  →  {dst_repo_id}")

    # ── 1. tasks.parquet — copy verbatim ──
    src_tasks = src / "meta" / "tasks.parquet"
    dst_tasks = dst / "meta" / "tasks.parquet"
    dst_tasks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_tasks, dst_tasks)

    # ── 2. videos/ — copy all subdirs except the dropped feature's, if video ──
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    if src_videos.exists():
        dst_videos.mkdir(parents=True, exist_ok=True)
        for sub in sorted(src_videos.iterdir()):
            if sub.name == feature_key:
                # only video-dtype features have a tree here; skip it
                print(f"[skip] videos/{sub.name}")
                continue
            print(f"[copy] videos/{sub.name}")
            shutil.copytree(sub, dst_videos / sub.name)

    # ── 3. data/ — strip column from every parquet ──
    src_data = src / "data"
    dst_data = dst / "data"
    parquets = sorted(src_data.rglob("file-*.parquet"))
    print(f"[strip] {len(parquets)} data parquets")
    for p in parquets:
        rel = p.relative_to(src_data)
        _strip_data_parquet(p, dst_data / rel, feature_key)

    # ── 4. meta/episodes/ — strip stats/<key>/* and videos/<key>/* columns ──
    src_eps = src / "meta" / "episodes"
    dst_eps = dst / "meta" / "episodes"
    eps = sorted(src_eps.rglob("file-*.parquet"))
    print(f"[strip] {len(eps)} meta/episodes parquets")
    for p in eps:
        rel = p.relative_to(src_eps)
        _strip_episodes_parquet(p, dst_eps / rel, feature_key)

    # ── 5. meta/info.json + meta/stats.json — drop key ──
    _strip_info_json(src / "meta" / "info.json", dst / "meta" / "info.json", feature_key)
    src_stats_json = src / "meta" / "stats.json"
    if src_stats_json.exists():
        _strip_stats_json(src_stats_json, dst / "meta" / "stats.json", feature_key)

    print(f"\n[done] {dst}")
    return dst


def _verify(dst_repo_id: str, dst_root: str | None, feature_key: str) -> None:
    """Load the stripped dataset and assert the feature is gone."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

    root = _resolve_root(dst_repo_id, dst_root)
    meta = LeRobotDatasetMetadata(dst_repo_id, root=str(root))
    assert feature_key not in meta.features, f"feature still in features dict: {feature_key}"
    assert feature_key not in (meta.stats or {}), f"feature still in stats: {feature_key}"
    print(f"[verify] features keys ({len(meta.features)}): {sorted(meta.features)}")
    print(f"[verify] total_episodes={meta.total_episodes}  total_frames={meta.total_frames}")

    # Also sanity-check by loading one frame end-to-end
    ds = LeRobotDataset(dst_repo_id, root=str(root))
    item = ds[0]
    item_keys_with_feature = [k for k in item.keys() if feature_key in k]
    assert not item_keys_with_feature, f"feature surfaced in __getitem__: {item_keys_with_feature}"
    print(f"[verify] frame 0 loads ok, keys: {sorted(item.keys())}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--src", required=True, help="Source repo_id")
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--dst", required=True, help="Destination repo_id")
    ap.add_argument("--dst-root", default=None)
    ap.add_argument("--feature", required=True, help="Feature key to strip")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    strip_feature(
        src_repo_id=args.src,
        dst_repo_id=args.dst,
        feature_key=args.feature,
        src_root=args.src_root,
        dst_root=args.dst_root,
        overwrite=args.overwrite,
    )
    if not args.no_verify:
        _verify(args.dst, args.dst_root, args.feature)


if __name__ == "__main__":
    main()
