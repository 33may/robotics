"""Stride-based episode subsampler for LeRobot datasets.

Produces a new LeRobotDataset that keeps every Nth episode of a source dataset.
Uses lerobot.datasets.dataset_tools internal helpers (_copy_and_reindex_*) so
video files are copied at chunk-level when possible, and only re-encoded for
mixed chunks. Episode indices are reindexed 0..M-1 in the output.

Usage
-----
    python -m vbti.logic.dataset.subsample \\
        --src=eternalmay33/duck_cup_v020_all \\
        --stride=4 \\
        --dst=eternalmay33/duck_cup_v020_every4

After build, rsync to remote:
    rsync -aP ~/.cache/huggingface/lerobot/eternalmay33/duck_cup_v020_every4/ \\
              vbti@<remote>:/home/vbti/anton/data/eternalmay33/duck_cup_v020_every4/
"""

from pathlib import Path

from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
    _copy_and_reindex_videos,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME


def subsample(
    src: str,
    stride: int,
    dst: str,
    output_dir: str | None = None,
    start: int = 0,
) -> str:
    """Build a new dataset containing every Nth episode of `src`.

    Args:
        src: source repo_id (e.g. "eternalmay33/duck_cup_v020_all").
        stride: keep every Nth episode (stride=4 → 25%, stride=8 → 12.5%).
        dst: destination repo_id.
        output_dir: where to write `dst` on disk. Default: HF_LEROBOT_HOME/<dst>.
        start: first episode index to keep (default 0).

    Returns:
        Path to the new dataset root.
    """
    if stride < 2:
        raise ValueError(f"stride must be >= 2, got {stride}")

    src_ds = LeRobotDataset(src)
    total = src_ds.meta.total_episodes
    kept = list(range(start, total, stride))
    print(f"Source: {src} — {total} episodes, {src_ds.meta.total_frames} frames")
    print(f"Keep:   every {stride}th from {start} → {len(kept)} episodes "
          f"(first={kept[:3]}, last={kept[-3:]})")

    dst_root = Path(output_dir).expanduser().resolve() if output_dir else HF_LEROBOT_HOME / dst

    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    episode_mapping = {old: new for new, old in enumerate(kept)}

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=dst,
        fps=src_ds.meta.fps,
        features=src_ds.meta.features,
        robot_type=src_ds.meta.robot_type,
        root=dst_root,
        use_videos=len(src_ds.meta.video_keys) > 0,
        chunks_size=src_ds.meta.chunks_size,
        data_files_size_in_mb=src_ds.meta.data_files_size_in_mb,
        video_files_size_in_mb=src_ds.meta.video_files_size_in_mb,
    )

    video_metadata = None
    if src_ds.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(src_ds, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(src_ds, new_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(
        src_ds, new_meta, episode_mapping, data_metadata, video_metadata
    )

    out = LeRobotDataset(repo_id=dst, root=dst_root)
    print(f"\nDone: {dst_root}")
    print(f"  episodes:     {out.meta.total_episodes}")
    print(f"  total_frames: {out.meta.total_frames}")
    print(f"  fps:          {out.meta.fps}")
    return str(dst_root)


if __name__ == "__main__":
    import fire
    fire.Fire(subsample)
