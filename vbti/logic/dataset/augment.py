"""Modular augmentation pipeline for LeRobot v3.0 datasets.

Each augmentation is a self-contained class that knows how to:
  1. prepare() — ensure its lookup data exists (idempotent)
  2. state_names() — declare what columns it adds to the state vector
  3. build_lookup() — load data into a (episode, frame) -> ndarray map

The orchestrator (augment_dataset) stacks augmentations, reuses existing data,
and only recomputes what's missing.

Usage:
    # Detection only (auto-finds/merges source parquets for merged datasets):
    python -m vbti.logic.dataset.augment eternalmay33/01_02_03_merged_may-sim \\
        --augmentations detection --drop top_duck

    # Detection + phase:
    python -m vbti.logic.dataset.augment eternalmay33/02_black_full_center \\
        --augmentations detection phase

    # Dry run:
    python -m vbti.logic.dataset.augment eternalmay33/02_black_full_center --dry-run

    # Add phase to an already-augmented dataset that has detection:
    python -m vbti.logic.dataset.augment eternalmay33/02_aug_v1 \\
        --augmentations phase -o eternalmay33/02_aug_v2
"""

from __future__ import annotations

import abc
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vbti.logic.dataset import LEROBOT_CACHE, resolve_dataset_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECTS = ["duck", "cup"]
PHASE_NAMES = ["reach", "pregrasp", "grasp", "transport", "release"]
DEFAULT_CAMERAS = ["left", "right", "top", "gripper"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Augmentation(abc.ABC):
    """Base class for dataset augmentation modules."""

    name: str  # unique identifier

    @abc.abstractmethod
    def prepare(self, dataset_path: Path) -> Path:
        """Ensure auxiliary data file exists. Returns path to it.

        This is idempotent — if the data already exists, return immediately.
        If it doesn't, compute or derive it (e.g. run detection, merge from
        source datasets, compute phase labels).
        """
        ...

    @abc.abstractmethod
    def state_names(self) -> list[str]:
        """Return ordered list of state column names this augmentation adds."""
        ...

    @abc.abstractmethod
    def build_lookup(self, data_path: Path) -> dict[tuple[int, int], np.ndarray]:
        """Load prepared data into a (episode_index, frame_index) -> float32 array map."""
        ...

    def zeros(self) -> np.ndarray:
        """Default value for frames without data."""
        return np.zeros(len(self.state_names()), dtype=np.float32)


# ---------------------------------------------------------------------------
# Detection augmentation
# ---------------------------------------------------------------------------


def _load_episode_meta(dataset_path: Path) -> pd.DataFrame:
    """Load episode metadata from a LeRobot dataset."""
    ep_dir = dataset_path / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata in {ep_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def _find_detection_sources(dataset_path: Path) -> list[tuple[Path, int]] | None:
    """Try to find source detection parquets for a merged dataset.

    Looks at the data/chunk-NNN directories to find where episodes came from,
    then scans sibling datasets in the cache for detection_results.parquet files
    whose episode counts match the chunk boundaries.

    Returns list of (parquet_path, episode_offset) or None if not found.
    """
    ep_df = _load_episode_meta(dataset_path)
    total_eps = len(ep_df)

    # Look for source datasets in the same author directory
    author_dir = dataset_path.parent
    if not author_dir.exists():
        return None

    # Find all sibling datasets that have detection_results.parquet
    candidates: list[tuple[str, Path, int]] = []
    for ds_dir in sorted(author_dir.iterdir()):
        if ds_dir == dataset_path or not ds_dir.is_dir():
            continue
        det_path = ds_dir / "detection_results.parquet"
        if not det_path.exists():
            continue
        try:
            src_ep_df = _load_episode_meta(ds_dir)
            candidates.append((ds_dir.name, det_path, len(src_ep_df)))
        except (FileNotFoundError, Exception):
            continue

    if not candidates:
        return None

    # Try to find a combination that covers all episodes in order.
    # The merged dataset is a sequential concatenation, so we look for
    # candidates whose episode counts sum to total_eps in the right order.
    #
    # Strategy: greedily match from the beginning. For each candidate,
    # check if its episode count matches the next chunk of the merged dataset
    # by comparing frame lengths.
    merged_lengths = ep_df["length"].values
    offset = 0
    sources: list[tuple[Path, int]] = []
    used = set()

    while offset < total_eps:
        found = False
        for name, det_path, n_eps in candidates:
            if name in used:
                continue
            # Check if this candidate's frame lengths match the merged dataset
            # at the current offset
            if offset + n_eps > total_eps:
                continue
            try:
                src_ds = det_path.parent
                src_ep_df = _load_episode_meta(src_ds)
                src_lengths = src_ep_df["length"].values
                merged_chunk = merged_lengths[offset : offset + n_eps]
                if len(src_lengths) == len(merged_chunk) and np.array_equal(src_lengths, merged_chunk):
                    sources.append((det_path, offset))
                    used.add(name)
                    offset += n_eps
                    found = True
                    break
            except Exception:
                continue
        if not found:
            return None  # can't find matching sources

    return sources if offset == total_eps else None


def _merge_detection_parquets(
    sources: list[tuple[Path, int]],
    output_path: Path,
) -> Path:
    """Merge detection parquets from source datasets with episode remapping."""
    dfs = []
    for det_path, ep_offset in sources:
        df = pd.read_parquet(det_path)
        if ep_offset > 0:
            df = df.copy()
            df["episode_index"] = df["episode_index"] + ep_offset
        dfs.append(df)
        src_name = det_path.parent.name
        print(f"  {src_name}: {len(df)} rows, offset +{ep_offset}")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Ensure correct types
    merged["frame_index"] = merged["frame_index"].astype(np.int64)
    merged["episode_index"] = merged["episode_index"].astype(np.int64)
    for col in merged.columns:
        if col not in ("frame_index", "episode_index"):
            merged[col] = merged[col].astype(np.float32)

    merged.to_parquet(output_path, index=False)
    print(f"  Merged: {len(merged)} rows -> {output_path.name}")
    return output_path


class DetectionAugmentation(Augmentation):
    """Add object detection coordinates (cx, cy) to state vector."""

    name = "detection"

    def __init__(
        self,
        cameras: list[str] | None = None,
        include_confidence: bool = False,
        drop: set[str] | None = None,
        parquet_path: str | Path | None = None,
    ):
        self.cameras = cameras or list(DEFAULT_CAMERAS)
        self.include_confidence = include_confidence
        self.drop = drop or set()
        self.parquet_path = Path(parquet_path) if parquet_path else None

    def prepare(self, dataset_path: Path) -> Path:
        # 0. Custom parquet path — use directly
        if self.parquet_path is not None:
            if not self.parquet_path.is_absolute():
                self.parquet_path = dataset_path / self.parquet_path
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Custom detection parquet not found: {self.parquet_path}")
            print(f"[detection] Using custom parquet: {self.parquet_path.name}")
            return self.parquet_path

        det_path = dataset_path / "detection_results.parquet"

        # 1. Already exists — reuse
        if det_path.exists():
            print(f"[detection] Reusing existing: {det_path.name}")
            return det_path

        # 2. Try to merge from source datasets (merged dataset case)
        print(f"[detection] No detection parquet found, searching for sources...")
        sources = _find_detection_sources(dataset_path)
        if sources:
            print(f"[detection] Found {len(sources)} source datasets, merging:")
            return _merge_detection_parquets(sources, det_path)

        # 3. Run OWLv2 detection from scratch
        print(f"[detection] No sources found, running OWLv2 detection...")
        from vbti.logic.detection.process_dataset import process_dataset
        result_path = process_dataset(
            dataset_path=str(dataset_path),
            cameras=self.cameras,
        )
        assert result_path is not None
        return Path(result_path)

    def state_names(self) -> list[str]:
        names = []
        for cam in self.cameras:
            for obj in OBJECTS:
                if f"{cam}_{obj}" in self.drop:
                    continue
                names.append(f"{cam}_{obj}_cx")
                names.append(f"{cam}_{obj}_cy")
                if self.include_confidence:
                    names.append(f"{cam}_{obj}_conf")
        return names

    def build_lookup(self, data_path: Path) -> dict[tuple[int, int], np.ndarray]:
        df = pd.read_parquet(data_path)
        cols = self.state_names()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Detection parquet missing columns: {missing}\n"
                f"Available: {sorted(df.columns.tolist())}"
            )

        lookup: dict[tuple[int, int], np.ndarray] = {}
        ep = df["episode_index"].values
        fr = df["frame_index"].values
        vals = df[cols].values.astype(np.float32)
        np.nan_to_num(vals, nan=0.0, copy=False)
        for i in range(len(df)):
            lookup[(int(ep[i]), int(fr[i]))] = vals[i]
        return lookup


# ---------------------------------------------------------------------------
# Phase augmentation
# ---------------------------------------------------------------------------


def _find_phase_sources(dataset_path: Path) -> list[tuple[Path, int]] | None:
    """Same logic as detection but for phase_labels.parquet."""
    ep_df = _load_episode_meta(dataset_path)
    total_eps = len(ep_df)
    author_dir = dataset_path.parent
    if not author_dir.exists():
        return None

    candidates: list[tuple[str, Path, int]] = []
    for ds_dir in sorted(author_dir.iterdir()):
        if ds_dir == dataset_path or not ds_dir.is_dir():
            continue
        phase_path = ds_dir / "phase_labels.parquet"
        if not phase_path.exists():
            continue
        try:
            src_ep_df = _load_episode_meta(ds_dir)
            candidates.append((ds_dir.name, phase_path, len(src_ep_df)))
        except Exception:
            continue

    if not candidates:
        return None

    merged_lengths = ep_df["length"].values
    offset = 0
    sources: list[tuple[Path, int]] = []
    used = set()

    while offset < total_eps:
        found = False
        for name, phase_path, n_eps in candidates:
            if name in used:
                continue
            if offset + n_eps > total_eps:
                continue
            try:
                src_ds = phase_path.parent
                src_ep_df = _load_episode_meta(src_ds)
                src_lengths = src_ep_df["length"].values
                merged_chunk = merged_lengths[offset : offset + n_eps]
                if len(src_lengths) == len(merged_chunk) and np.array_equal(src_lengths, merged_chunk):
                    sources.append((phase_path, offset))
                    used.add(name)
                    offset += n_eps
                    found = True
                    break
            except Exception:
                continue
        if not found:
            return None

    return sources if offset == total_eps else None


class PhaseAugmentation(Augmentation):
    """Add one-hot task phase encoding to state vector."""

    name = "phase"

    def prepare(self, dataset_path: Path) -> Path:
        phase_path = dataset_path / "phase_labels.parquet"

        if phase_path.exists():
            print(f"[phase] Reusing existing: {phase_path.name}")
            return phase_path

        print(f"[phase] No phase labels found, searching for sources...")
        sources = _find_phase_sources(dataset_path)
        if sources:
            print(f"[phase] Found {len(sources)} source datasets, merging:")
            return _merge_detection_parquets(sources, phase_path)

        # Run phase detection from scratch
        print(f"[phase] No sources found, computing phase labels...")
        from vbti.logic.detection.phases import process_phases_dataset
        result_path = process_phases_dataset(str(dataset_path))
        return Path(result_path)

    def state_names(self) -> list[str]:
        return [f"phase_{p}" for p in PHASE_NAMES]

    def build_lookup(self, data_path: Path) -> dict[tuple[int, int], np.ndarray]:
        df = pd.read_parquet(data_path)
        num_phases = len(PHASE_NAMES)
        lookup: dict[tuple[int, int], np.ndarray] = {}
        ep = df["episode_index"].values
        fr = df["frame_index"].values
        ph = df["phase"].values
        for i in range(len(df)):
            onehot = np.zeros(num_phases, dtype=np.float32)
            phase_idx = int(ph[i])
            if 0 <= phase_idx < num_phases:
                onehot[phase_idx] = 1.0
            lookup[(int(ep[i]), int(fr[i]))] = onehot
        return lookup


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY: dict[str, type[Augmentation]] = {
    "detection": DetectionAugmentation,
    "phase": PhaseAugmentation,
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _compute_stats(values: np.ndarray) -> dict[str, list]:
    """Compute LeRobot-style stats for an (N, D) array."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def augment_dataset(
    source_dataset: str,
    output_name: str | None = None,
    augmentations: list[Augmentation] | None = None,
    root: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Create augmented dataset by stacking augmentation modules onto the state vector.

    Each augmentation's prepare() step runs first (idempotent — skips if data exists),
    then all lookups are built, and finally the output dataset is assembled.

    Args:
        source_dataset: repo_id or path
        output_name: output repo_id (default: source + "_aug")
        augmentations: list of Augmentation instances to apply
        root: override dataset root path
        dry_run: print plan without writing

    Returns:
        Path to the created augmented dataset directory.
    """
    if augmentations is None:
        augmentations = []

    src_dir = resolve_dataset_path(source_dataset, root=root)
    if output_name is None:
        output_name = source_dataset + "_aug"
    dst_dir = LEROBOT_CACHE / output_name

    print(f"Source:  {src_dir}")
    print(f"Output:  {dst_dir}")

    # Load source info
    with open(src_dir / "meta" / "info.json") as f:
        info: dict[str, Any] = json.load(f)

    original_names: list[str] = info["features"]["observation.state"]["names"]
    original_dim = info["features"]["observation.state"]["shape"][0]

    # Build new state layout
    new_names = list(original_names)
    for aug in augmentations:
        new_names.extend(aug.state_names())
    new_dim = len(new_names)

    print(f"State:   {original_dim}d -> {new_dim}d")
    print(f"Augmentations: {[a.name for a in augmentations]}")

    # Print layout
    idx = original_dim
    layout_parts = [f"[0:{original_dim}] original"]
    for aug in augmentations:
        n = len(aug.state_names())
        layout_parts.append(f"[{idx}:{idx + n}] {aug.name}")
        idx += n
    print(f"Layout:  {' | '.join(layout_parts)}")

    if dry_run:
        print("\n[DRY RUN] Would create:")
        print(f"  {dst_dir}/meta/info.json  (state {original_dim} -> {new_dim})")
        print(f"  {dst_dir}/meta/stats.json (recomputed)")
        data_files = sorted((src_dir / "data").rglob("*.parquet"))
        print(f"  {dst_dir}/data/ ({len(data_files)} parquet files)")
        vid_dir = src_dir / "videos"
        if vid_dir.exists():
            vid_count = sum(1 for _ in vid_dir.rglob("*.mp4"))
            print(f"  {dst_dir}/videos/ ({vid_count} video files — symlinked)")
        print(f"\nState names:")
        for i, name in enumerate(new_names):
            marker = " *" if i >= original_dim else ""
            print(f"  [{i:2d}] {name}{marker}")
        return dst_dir

    # -------------------------------------------------------------------
    # Phase 1: Prepare all augmentation data (idempotent)
    # -------------------------------------------------------------------
    print("\n--- Preparing augmentation data ---")
    data_paths: list[Path] = []
    for aug in augmentations:
        path = aug.prepare(src_dir)
        data_paths.append(path)

    # -------------------------------------------------------------------
    # Phase 2: Build lookups
    # -------------------------------------------------------------------
    print("\n--- Building lookups ---")
    lookups: list[dict[tuple[int, int], np.ndarray]] = []
    zero_vecs: list[np.ndarray] = []
    for aug, path in zip(augmentations, data_paths):
        lookup = aug.build_lookup(path)
        lookups.append(lookup)
        zero_vecs.append(aug.zeros())
        print(f"  {aug.name}: {len(lookup)} frames loaded")

    # -------------------------------------------------------------------
    # Phase 3: Create output dataset
    # -------------------------------------------------------------------
    print("\n--- Writing augmented dataset ---")

    if dst_dir.exists():
        print(f"Removing existing: {dst_dir}")
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True)
    (dst_dir / "meta" / "episodes").mkdir(parents=True)
    (dst_dir / "data").mkdir(parents=True)

    # Symlink videos
    src_videos = src_dir / "videos"
    dst_videos = dst_dir / "videos"
    if src_videos.exists():
        print("Symlinking videos...")
        for cam_dir in sorted(src_videos.iterdir()):
            if not cam_dir.is_dir():
                continue
            for chunk_dir in sorted(cam_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                dst_chunk = dst_videos / cam_dir.name / chunk_dir.name
                dst_chunk.mkdir(parents=True, exist_ok=True)
                for vid_file in sorted(chunk_dir.iterdir()):
                    if vid_file.is_file():
                        rel = os.path.relpath(vid_file, dst_chunk)
                        os.symlink(rel, dst_chunk / vid_file.name)
        vid_count = sum(1 for _ in dst_videos.rglob("*.mp4"))
        print(f"  {vid_count} video files symlinked")

    # Copy tasks
    src_tasks = src_dir / "meta" / "tasks.parquet"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst_dir / "meta" / "tasks.parquet")

    # Process data parquets
    src_data = src_dir / "data"
    data_files = sorted(src_data.rglob("*.parquet"))
    all_new_states: list[np.ndarray] = []

    print(f"Processing {len(data_files)} data files...")
    for file_idx, src_file in enumerate(data_files):
        rel = src_file.relative_to(src_data)
        dst_file = dst_dir / "data" / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_file)
        n = table.num_rows

        # Original state
        orig_state = np.array(
            [table.column("observation.state")[i].as_py() for i in range(n)],
            dtype=np.float32,
        )

        # Read indices once
        ep_col = table.column("episode_index").to_pylist()
        fr_col = table.column("frame_index").to_pylist()

        # Stack augmentation columns
        extras = []
        for lookup, zeros in zip(lookups, zero_vecs):
            arr = np.array(
                [lookup.get((ep_col[i], fr_col[i]), zeros) for i in range(n)],
                dtype=np.float32,
            )
            extras.append(arr)

        new_state = np.concatenate([orig_state] + extras, axis=1) if extras else orig_state
        all_new_states.append(new_state)

        # Replace observation.state column
        new_state_col = pa.FixedSizeListArray.from_arrays(
            pa.array(new_state.ravel(), type=pa.float32()),
            list_size=new_dim,
        )
        col_idx = table.schema.get_field_index("observation.state")
        table = table.set_column(
            col_idx,
            pa.field("observation.state", pa.list_(pa.float32(), new_dim)),
            new_state_col,
        )
        pq.write_table(table, dst_file)
        print(f"  [{file_idx + 1}/{len(data_files)}] {rel} ({n} rows)")

    # -------------------------------------------------------------------
    # Phase 4: Stats
    # -------------------------------------------------------------------
    print("Computing stats...")
    all_states = np.concatenate(all_new_states, axis=0)
    new_state_stats = _compute_stats(all_states)

    with open(src_dir / "meta" / "stats.json") as f:
        stats = json.load(f)
    stats["observation.state"] = new_state_stats
    with open(dst_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # -------------------------------------------------------------------
    # Phase 5: Episode metadata (per-episode stats)
    # -------------------------------------------------------------------
    print("Updating episode metadata...")

    # Build per-episode state arrays
    ep_state_map: dict[int, list[np.ndarray]] = {}
    state_offset = 0
    for src_file in data_files:
        table = pq.read_table(src_file, columns=["episode_index"])
        ep_list = table.column("episode_index").to_pylist()
        n = len(ep_list)
        file_states = all_states[state_offset : state_offset + n]
        for i in range(n):
            ep = ep_list[i]
            if ep not in ep_state_map:
                ep_state_map[ep] = []
            ep_state_map[ep].append(file_states[i])
        state_offset += n

    ep_stats: dict[int, dict[str, list]] = {}
    for ep, state_list in ep_state_map.items():
        ep_stats[ep] = _compute_stats(np.array(state_list, dtype=np.float64))

    # Rewrite episode meta parquets
    src_ep_dir = src_dir / "meta" / "episodes"
    dst_ep_dir = dst_dir / "meta" / "episodes"
    stat_keys = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

    ep_meta_files = sorted(src_ep_dir.rglob("*.parquet"))
    for src_ep_file in ep_meta_files:
        rel = src_ep_file.relative_to(src_ep_dir)
        dst_ep_file = dst_ep_dir / rel
        dst_ep_file.parent.mkdir(parents=True, exist_ok=True)

        ep_table = pq.read_table(src_ep_file)
        ep_indices = ep_table.column("episode_index").to_pylist()

        for stat_name in stat_keys:
            col_name = f"stats/observation.state/{stat_name}"
            if col_name not in ep_table.column_names:
                continue
            new_values = []
            for ep in ep_indices:
                if ep in ep_stats:
                    new_values.append(ep_stats[ep][stat_name])
                else:
                    orig = ep_table.column(col_name)[ep_indices.index(ep)].as_py()
                    new_values.append(list(orig) + [0.0] * (new_dim - original_dim))

            col_idx = ep_table.schema.get_field_index(col_name)
            inner_type = pa.int64() if stat_name == "count" else pa.float64()
            new_col = pa.array(new_values, type=pa.list_(inner_type))
            ep_table = ep_table.set_column(col_idx, ep_table.schema.field(col_name), new_col)

        pq.write_table(ep_table, dst_ep_file)

    print(f"  {len(ep_meta_files)} episode files updated")

    # -------------------------------------------------------------------
    # Phase 6: Write info.json
    # -------------------------------------------------------------------
    info["features"]["observation.state"]["shape"] = [new_dim]
    info["features"]["observation.state"]["names"] = new_names
    with open(dst_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone! {dst_dir}")
    print(f"  State: {original_dim}d -> {new_dim}d "
          f"({info['total_frames']} frames, {info['total_episodes']} episodes)")
    return dst_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_augmentations(args: argparse.Namespace) -> list[Augmentation]:
    """Build augmentation list from CLI args."""
    augs: list[Augmentation] = []
    names = args.augmentations or []

    for name in names:
        if name == "detection":
            augs.append(DetectionAugmentation(
                cameras=args.cameras,
                include_confidence=args.include_confidence,
                drop=set(args.drop) if args.drop else None,
                parquet_path=args.detection_parquet,
            ))
        elif name == "phase":
            augs.append(PhaseAugmentation())
        else:
            raise ValueError(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY)}")
    return augs


def main():
    parser = argparse.ArgumentParser(
        description="Augment LeRobot dataset with modular state-vector features.",
    )
    parser.add_argument("source", help="Source dataset repo_id or path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output dataset repo_id (default: source_aug)")
    parser.add_argument("--augmentations", nargs="+", default=["detection"],
                        choices=list(AUGMENTATION_REGISTRY),
                        help="Augmentations to apply (default: detection)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera names for detection (default: left right top)")
    parser.add_argument("--drop", nargs="+", default=None,
                        help="Drop specific cam_obj pairs (e.g. --drop top_duck)")
    parser.add_argument("--include-confidence", action="store_true",
                        help="Include detection confidence values")
    parser.add_argument("--detection-parquet", default=None,
                        help="Custom detection parquet file (e.g. detection_results_hold.parquet)")
    parser.add_argument("--root", default=None, help="Override dataset root path")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    args = parser.parse_args()

    augmentations = _build_augmentations(args)
    augment_dataset(
        source_dataset=args.source,
        output_name=args.output,
        augmentations=augmentations,
        root=args.root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
