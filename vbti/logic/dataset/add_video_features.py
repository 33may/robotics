"""Bake teacher visual features into a LeRobot dataset for UVA aux-loss training.

Creates a NEW dataset copy at --output containing all original data PLUS a new
column ``observation.video_features.{layer}_{S}x{S}`` with fp16/fp32 tensors of
shape (t_future, S, S, feat_dim) per frame, where S = --spatial-size,
feat_dim comes from the teacher's SigLIP last layer, and t_future = --t-future.

Each row stores the **pre-assembled future window**: frame t holds the features
of frames t+1, t+2, ..., t+t_future, clamped at episode boundaries (the last
frame of the episode is repeated if t+k would fall outside). This means the
policy can read the full future window directly via observation_delta_indices=[0]
without touching any other observation.* key.

Dataset copy layout:
  {output}/meta/         — full copy of source meta/ (info.json patched in-place)
  {output}/videos/       — symlink to source videos/ dir (mp4s are unchanged)
  {output}/data/chunk-*/file-*.parquet — each source parquet rewritten with the
                           new column added (sliced by the file's `index` values).

Usage:
  python -m vbti.logic.dataset.add_video_features \\
      --dataset eternalmay33/06_black_cup_red_bg_depth \\
      --teacher vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model \\
      --layer siglip_output \\
      --spatial-size 4 \\
      --t-future 4 \\
      --target-camera observation.images.gripper \\
      --output /path/to/new_dataset_with_uva
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad

DEFAULT_REWRITE_CHUNK_ROWS = 256

# Import to trigger registry registration of siglip_output
import vbti.logic.dataset.target_extractors.siglip_output  # noqa: F401
from vbti.logic.dataset.target_extractors import get as get_extractor, list_available


def build_feature_key(layer: str, spatial_size: int) -> str:
    """Self-describing column name: 'observation.video_features.{layer}_{S}x{S}'."""
    return f"observation.video_features.{layer}_{spatial_size}x{spatial_size}"


def _preprocess_image_for_teacher(
    img: torch.Tensor,
    teacher: SmolVLAPolicy,
) -> torch.Tensor:
    """Apply SmolVLA's full image preprocessing to a single (3,H,W) tensor.

    1. resize_with_pad to teacher.config.resize_imgs_with_padding (expects (B,C,H,W))
    2. normalize [0,1] -> [-1,1]
    """
    # resize_with_pad expects (B, C, H, W)
    img4d = img.unsqueeze(0)  # (1, 3, H, W)
    if teacher.config.resize_imgs_with_padding is not None:
        w, h = teacher.config.resize_imgs_with_padding
        img4d = resize_with_pad(img4d, w, h, pad_value=0)
    img4d = img4d * 2.0 - 1.0  # [0,1] -> [-1,1]
    return img4d.squeeze(0)    # back to (3, H', W')


def _bake_features(
    dataset: LeRobotDataset,
    teacher: SmolVLAPolicy,
    extractor,
    target_camera: str,
    spatial_size: int,
    batch_size: int,
    dtype: str,
    device: str,
    log: logging.Logger,
) -> torch.Tensor:
    """Walk every frame, run teacher, return (N, S, S, D) tensor (fp16 or fp32).

    Keeps the entire output in RAM as a torch tensor — fp16 at 4x4x1152 is ~780 MB
    for 21k frames, which fits comfortably.
    """
    n_frames = len(dataset)
    log.info(f"baking {n_frames} frames | camera={target_camera} | spatial_size={spatial_size}")

    out_chunks = []
    for start in tqdm(range(0, n_frames, batch_size), unit="batch"):
        end = min(start + batch_size, n_frames)
        imgs = []
        for idx in range(start, end):
            sample = dataset[idx]
            img = sample[target_camera]  # (3, H, W) or (T, 3, H, W) float in [0,1]
            if img.ndim == 4:            # (T, 3, H, W) — take t=0
                img = img[0]
            img = _preprocess_image_for_teacher(img, teacher)
            imgs.append(img)
        # Stack -> (B, 3, H', W')
        imgs_t = torch.stack(imgs, dim=0).to(device)
        feats = extractor(teacher, imgs_t, spatial_size=spatial_size)
        if dtype == "fp16":
            feats = feats.half()
        out_chunks.append(feats.cpu())

    out = torch.cat(out_chunks, dim=0)   # (N, S, S, D)
    log.info(f"baked tensor shape={tuple(out.shape)} dtype={out.dtype}")
    return out


def _episode_last_index(episode_index: list[int]) -> dict[int, int]:
    """Map episode id -> its last global frame index.

    Episodes are assumed contiguous in global index (true for LeRobot v3 datasets).
    """
    ep_last: dict[int, int] = {}
    for global_idx, ep in enumerate(episode_index):
        if ep not in ep_last or global_idx > ep_last[ep]:
            ep_last[ep] = global_idx
    return ep_last


def _assemble_windows_for_indices(
    baked: torch.Tensor,          # (N, S, S, D) per-frame features, global frame index
    episode_index: list[int],     # (N,) episode id of each global frame index
    ep_last: dict[int, int],      # episode id -> last global frame index
    indices: list[int],           # global frame indices of ONE parquet file's rows
    t_future: int,
) -> torch.Tensor:
    """Build future windows for one parquet file's rows: g -> [feat[g+1]..feat[g+t_future]].

    Frames near an episode end are clamped: if g+k exceeds the episode's last
    frame, that last frame's feature is repeated.

    Assembling per-file (rather than one global (N, t_future, S, S, D) tensor)
    keeps peak RAM at ~the per-frame tensor size. The full windowed tensor is
    t_future x larger — 41 GB for a 337k-frame dataset at 4x4x960 fp16, which
    plus the 10 GB per-frame tensor would not fit the remote's 60 GB.

    Returns (len(indices), t_future, S, S, D), same dtype as `baked`.
    """
    _, S1, S2, D = baked.shape
    out = torch.empty((len(indices), t_future, S1, S2, D), dtype=baked.dtype)
    for i, g in enumerate(indices):
        last = ep_last[episode_index[g]]
        for k in range(1, t_future + 1):
            out[i, k - 1] = baked[min(g + k, last)]
    return out


def _feature_arrow_type(dtype_str: str) -> pa.DataType:
    pa_dtype = pa.float16() if dtype_str == "fp16" else pa.float32()
    innermost_type = pa.list_(pa.field("item", pa_dtype))
    inner_type = pa.list_(pa.field("item", innermost_type))
    mid_type = pa.list_(pa.field("item", inner_type))
    return pa.list_(pa.field("item", mid_type))


def _read_episode_index_from_parquets(src_root: Path, log: logging.Logger) -> list[int]:
    """Read `episode_index` and `index` columns from all data parquets.

    Returns a list of length N (total frames) where result[global_frame_index]
    = episode_index for that frame.
    """
    src_data = src_root / "data"
    parquet_files = sorted(src_data.glob("*/*.parquet"))
    log.info(f"reading episode_index from {len(parquet_files)} parquet files")

    # Collect (global_index, episode_id) pairs
    pairs: list[tuple[int, int]] = []
    for pq_path in parquet_files:
        table = pq.read_table(str(pq_path), columns=["index", "episode_index"])
        indices = table.column("index").to_pylist()
        ep_ids = table.column("episode_index").to_pylist()
        pairs.extend(zip(indices, ep_ids))

    # Sort by global index and build flat list
    pairs.sort(key=lambda x: x[0])
    n = pairs[-1][0] + 1  # global indices are 0-based
    episode_index = [0] * n
    for global_idx, ep_id in pairs:
        episode_index[global_idx] = ep_id

    log.info(f"episode_index array length={n}")
    return episode_index


def _build_dataset_copy(
    src_root: Path,
    output: Path,
    feature_key: str,
    baked: torch.Tensor,          # (N, S, S, D) — per-frame features, fp16 or fp32
    episode_index: list[int],     # (N,) episode id per global frame index
    dtype_str: str,
    spatial_size: int,
    t_future: int,
    log: logging.Logger,
    rewrite_chunk_rows: int = DEFAULT_REWRITE_CHUNK_ROWS,
) -> None:
    """Build the new dataset copy at `output`.

    Steps:
      1. Copy meta/ from src_root -> output/meta/
      2. Patch output/meta/info.json to add the new feature entry
      3. Symlink src_root/videos -> output/videos
      4. For each data/chunk-*/file-*.parquet: read with pyarrow, slice baked tensor
         by the file's `index` column values, add new column, write parquet.
    """
    feat_dim = baked.shape[-1]
    log.info(f"building dataset copy at {output}")

    # --- 1. Copy meta/ -------------------------------------------------------
    src_meta = src_root / "meta"
    dst_meta = output / "meta"
    log.info(f"copying {src_meta} -> {dst_meta}")
    shutil.copytree(str(src_meta), str(dst_meta))

    # --- 2. Patch meta/info.json ---------------------------------------------
    info_path = dst_meta / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["features"][feature_key] = {
        "dtype": "float16" if dtype_str == "fp16" else "float32",
        "shape": [t_future, spatial_size, spatial_size, feat_dim],
        "names": None,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    log.info(f"patched info.json: added feature '{feature_key}' shape=[{t_future},{spatial_size},{spatial_size},{feat_dim}]")

    # --- 3. Symlink videos/ --------------------------------------------------
    src_videos = (src_root / "videos").resolve()
    dst_videos = output / "videos"
    if src_videos.exists():
        os.symlink(str(src_videos), str(dst_videos))
        log.info(f"symlinked videos: {dst_videos} -> {src_videos}")
    else:
        log.warning(f"source videos dir not found at {src_videos}, skipping symlink")

    # --- 4. Rewrite data parquets --------------------------------------------
    src_data = src_root / "data"
    dst_data = output / "data"

    parquet_files = sorted(src_data.glob("*/*.parquet"))
    log.info(f"found {len(parquet_files)} parquet files to rewrite")

    ep_last = _episode_last_index(episode_index)
    feature_type = _feature_arrow_type(dtype_str)
    feature_field = pa.field(feature_key, feature_type)

    for src_pq in tqdm(parquet_files, desc="rewriting parquets", unit="file"):
        # Preserve chunk-XXX/file-YYY.parquet sub-path
        rel = src_pq.relative_to(src_data)
        dst_pq = dst_data / rel
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(src_pq))
        indices = table.column("index").to_pylist()  # global frame indices

        writer = None
        try:
            for row_start in range(0, table.num_rows, rewrite_chunk_rows):
                row_end = min(row_start + rewrite_chunk_rows, table.num_rows)
                chunk_indices = indices[row_start:row_end]
                chunk_table = table.slice(row_start, row_end - row_start)

                window = _assemble_windows_for_indices(
                    baked,
                    episode_index,
                    ep_last,
                    chunk_indices,
                    t_future,
                )
                new_col = pa.array(window.tolist(), type=feature_type)
                chunk_table = chunk_table.append_column(feature_field, new_col)

                if writer is None:
                    writer = pq.ParquetWriter(str(dst_pq), chunk_table.schema, compression="snappy")
                writer.write_table(chunk_table)
        finally:
            if writer is not None:
                writer.close()

    log.info(f"dataset copy complete at {output}")


def _verify_copy(
    output: Path,
    feature_key: str,
    spatial_size: int,
    feat_dim: int,
    t_future: int,
    log: logging.Logger,
) -> None:
    """Re-open the output dataset with LeRobotDataset and verify the new feature."""
    # Use a synthetic repo_id — only `root` matters for local-only datasets.
    # LeRobotDataset.__init__ tries to load from disk first (via load_metadata),
    # and only hits HF Hub if local files are missing.
    fake_repo_id = "local/verify_copy"
    log.info(f"verifying output dataset at {output}")
    check = LeRobotDataset(fake_repo_id, root=str(output))
    assert feature_key in check.features, (
        f"VERIFICATION FAILED: '{feature_key}' not in reloaded dataset.features. "
        f"Available: {list(check.features.keys())}"
    )
    sample = check[0]
    actual_shape = tuple(sample[feature_key].shape)
    expected_suffix = (t_future, spatial_size, spatial_size, feat_dim)
    assert actual_shape[-4:] == expected_suffix, (
        f"VERIFICATION FAILED: expected shape ending {expected_suffix}, got {actual_shape}"
    )
    log.info(f"verification OK: {feature_key} shape={actual_shape}")


def main():
    p = argparse.ArgumentParser(description="Bake video features into a NEW LeRobot dataset copy")
    p.add_argument("--dataset", required=True, help="HF hub repo_id (e.g. user/my_dataset)")
    p.add_argument("--teacher", required=True, help="Path to teacher SmolVLAPolicy checkpoint")
    p.add_argument("--layer", default="siglip_output", help="Extractor registry name")
    p.add_argument("--spatial-size", type=int, default=4)
    p.add_argument("--t-future", type=int, default=4,
                   help="Future window depth: row t stores features of frames t+1..t+t_future "
                        "(clamped at episode boundaries). Default: 4")
    p.add_argument("--target-camera", default="observation.images.gripper",
                   help="Camera key to extract features from")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument(
        "--rewrite-chunk-rows",
        type=int,
        default=DEFAULT_REWRITE_CHUNK_ROWS,
        help=(
            "Rows per in-memory Arrow conversion while rewriting parquets. "
            "Lower values use less RAM; default: 256"
        ),
    )
    p.add_argument("--root", default=None, help="LeRobot dataset cache root (source)")
    p.add_argument("--output", required=True, help="Absolute path for the new dataset copy (must not exist)")
    p.add_argument("--force", action="store_true", help="Allow --output to already exist (will delete it first)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bake")

    # -------------------------------------------------------------------------
    # Validate args BEFORE loading anything expensive
    # -------------------------------------------------------------------------
    available = list_available()
    if args.layer not in available:
        raise SystemExit(
            f"ERROR: layer '{args.layer}' not in extractor registry. "
            f"Available: {available}"
        )

    if args.t_future < 1:
        raise SystemExit(f"ERROR: --t-future must be >= 1, got {args.t_future}")
    if args.rewrite_chunk_rows < 1:
        raise SystemExit(
            f"ERROR: --rewrite-chunk-rows must be >= 1, got {args.rewrite_chunk_rows}"
        )

    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        raise SystemExit(f"ERROR: --teacher path does not exist: {teacher_path}")

    output = Path(args.output)
    if output.exists():
        if args.force:
            log.warning(f"--force set: removing existing output dir {output}")
            shutil.rmtree(str(output))
        else:
            raise SystemExit(
                f"ERROR: --output already exists: {output}\n"
                f"Pass --force to overwrite."
            )
    output.mkdir(parents=True, exist_ok=False)

    # -------------------------------------------------------------------------
    # Load teacher
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"loading teacher from {args.teacher}")
    teacher = SmolVLAPolicy.from_pretrained(str(teacher_path)).to(device).eval()

    # -------------------------------------------------------------------------
    # Load source dataset
    # -------------------------------------------------------------------------
    log.info(f"opening dataset {args.dataset} (root={args.root})")
    dataset = LeRobotDataset(args.dataset, root=args.root)

    if args.target_camera not in dataset.meta.camera_keys:
        log.warning(
            f"target-camera '{args.target_camera}' not in dataset camera_keys: "
            f"{dataset.meta.camera_keys}. Proceeding anyway."
        )

    feature_key = build_feature_key(args.layer, args.spatial_size)
    log.info(f"feature column: {feature_key}")

    if feature_key in dataset.features and not args.force:
        raise SystemExit(
            f"ERROR: Column '{feature_key}' already exists in source dataset. "
            f"Pass --force if you still want to bake."
        )

    extractor = get_extractor(args.layer)

    # -------------------------------------------------------------------------
    # Bake features — keep in memory as a single torch tensor (N, S, S, D)
    # -------------------------------------------------------------------------
    baked = _bake_features(
        dataset=dataset,
        teacher=teacher,
        extractor=extractor,
        target_camera=args.target_camera,
        spatial_size=args.spatial_size,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=device,
        log=log,
    )

    feat_dim = baked.shape[-1]
    log.info(f"feat_dim={feat_dim}")

    # -------------------------------------------------------------------------
    # Read episode_index per global frame index from parquets
    # -------------------------------------------------------------------------
    src_root = Path(dataset.root)
    episode_index = _read_episode_index_from_parquets(src_root, log)

    # -------------------------------------------------------------------------
    # Build the new dataset copy. Future windows ((t_future, S, S, D) per row)
    # are assembled per-parquet-file inside _build_dataset_copy from the
    # per-frame `baked` tensor — the full (N, t_future, S, S, D) windowed tensor
    # is never materialized (t_future x larger; 41 GB for this dataset).
    # -------------------------------------------------------------------------
    _build_dataset_copy(
        src_root=src_root,
        output=output,
        feature_key=feature_key,
        baked=baked,
        episode_index=episode_index,
        dtype_str=args.dtype,
        spatial_size=args.spatial_size,
        t_future=args.t_future,
        log=log,
        rewrite_chunk_rows=args.rewrite_chunk_rows,
    )

    # -------------------------------------------------------------------------
    # Verify the output is readable by LeRobotDataset
    # -------------------------------------------------------------------------
    _verify_copy(
        output=output,
        feature_key=feature_key,
        spatial_size=args.spatial_size,
        feat_dim=feat_dim,
        t_future=args.t_future,
        log=log,
    )

    log.info("bake complete")


if __name__ == "__main__":
    main()
