"""Distillation CLI: cache frames, train per-camera detectors, evaluate.

Usage:
    # Cache all frames (one-time, ~80GB on disk):
    python -m vbti.logic.detection.distill cache

    # Train a single camera:
    python -m vbti.logic.detection.distill train --cam top

    # Train all cameras in sequence:
    python -m vbti.logic.detection.distill train --all

    # Evaluate:
    python -m vbti.logic.detection.distill eval --cam top --checkpoint PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vbti.logic.detection.distill.distill_model import DistilledDetector, count_params
from vbti.logic.detection.process_dataset import (
    VideoReader,
    get_video_path,
    load_dataset_meta,
)

# -------- constants --------
DATASET_PATH = Path(
    "/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim"
)
DATASET_PATH_NO_OBJECTS = Path(
    "/home/may33/.cache/huggingface/lerobot/eternalmay33/distill_no_objects"
)
CLEAN_PARQUET_PATH = Path("/home/may33/.cache/vbti/detection_labels_final.parquet")
CLEAN_PARQUET_PATH_NO_OBJECTS = Path(
    "/home/may33/.cache/vbti/detection_labels_final_no_objects.parquet"
)
CACHE_ROOT = Path("/home/may33/.cache/vbti/distill_frames_224")
CACHE_ROOT_NO_OBJECTS = Path("/home/may33/.cache/vbti/distill_frames_no_objects_224")
NO_OBJECTS_EP_OFFSET = 1000  # no_objects episodes get +1000 to avoid collision with main
IMG_SIZE = 224
CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]

NATIVE_W, NATIVE_H = 640, 480
Y2_THR_NORM_TD = 0.85  # v9: top_duck arm-lock geometric filter (bottom 15%)

TRAIN_EPISODES = range(0, 200)
VAL_EPISODES = range(200, 244)

TRAINING_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/training"
)
# Legacy alias (some callers may still reference this)
OUTPUT_ROOT = TRAINING_ROOT


def load_all_labels(seed: int = 42, val_frac: float = 0.15) -> pd.DataFrame:
    """Load v11-final labels from both main and no_objects datasets, add `source`
    and per-frame stratified `split` columns, return one combined DataFrame.

    Episode-index namespace:
      - main: original 0..243
      - no_obj: original 0..5 remapped to NO_OBJECTS_EP_OFFSET..+5  (= 1000..1005)

    Stratified split: per (source, episode), randomly assign `val_frac` of rows
    to 'val' (fixed seed). This guarantees every episode contributes to both
    train and val while keeping the split deterministic.
    """
    main = pd.read_parquet(CLEAN_PARQUET_PATH)
    main["source"] = "main"

    no_obj = pd.read_parquet(CLEAN_PARQUET_PATH_NO_OBJECTS)
    no_obj["source"] = "no_obj"
    # Shift no_obj episode indices to avoid collision with main.
    # The no_obj frame cache was written with the same +1000 offset (ep_offset arg).
    no_obj["episode_index"] = no_obj["episode_index"].astype(np.int64) + NO_OBJECTS_EP_OFFSET

    combined = pd.concat([main, no_obj], ignore_index=True)

    rng = np.random.RandomState(seed)
    split_col = np.array(["train"] * len(combined), dtype=object)
    for (src, ep), grp in combined.groupby(["source", "episode_index"], sort=False):
        n = len(grp)
        n_val = max(1, int(round(n * val_frac)))
        idx = grp.index.to_numpy()
        val_idx = rng.choice(idx, size=min(n_val, n), replace=False)
        split_col[val_idx] = "val"
    combined["split"] = split_col

    return combined


# -------- CACHE --------

def _cache_paths(cam: str, cache_root: Path | None = None):
    """Return (memmap_path, index_path_shared)."""
    root = cache_root if cache_root is not None else CACHE_ROOT
    cam_dir = root / cam
    cam_dir.mkdir(parents=True, exist_ok=True)
    return cam_dir / "frames.uint8", root / "index.parquet"


def _count_total_frames(episodes: pd.DataFrame) -> int:
    return int(episodes["length"].sum())


def cache_frames(force: bool = False, dataset_path: Path | None = None,
                 cache_root: Path | None = None, ep_offset: int = 0):
    """Decode all 4 cameras to per-cam memmap arrays of (N, 224, 224, 3) uint8.

    Also write a shared index.parquet mapping (cam, episode, frame) -> flat_idx.
    The flat_idx is the same across cameras (since all cams have the same
    episodes and frame counts), so we only store it once but apply it per cam.

    Args:
        dataset_path: LeRobot dataset path (defaults to DATASET_PATH = main)
        cache_root:   Output root for memmap + index.parquet
                      (defaults to CACHE_ROOT = main cache dir)
        ep_offset:    Offset added to episode_index in the written index.parquet
                      (used to shift no_objects eps by +1000).
    """
    import cv2

    ds = dataset_path if dataset_path is not None else DATASET_PATH
    root = cache_root if cache_root is not None else CACHE_ROOT
    info, episodes = load_dataset_meta(ds)
    fps = info["fps"]
    total_frames = _count_total_frames(episodes)
    print(f"[cache] dataset={ds.name}  {len(episodes)} episodes, {total_frames} total frames (ep_offset={ep_offset})")

    # Build flat index (same for all cams).
    # We record the ORIGINAL ep_idx from the source dataset, then add ep_offset
    # on write so downstream (DistillDataset) sees a non-colliding namespace.
    rows = []
    flat = 0
    ep_flat_start: dict[int, int] = {}
    for _, ep in episodes.iterrows():
        ep_idx = int(ep["episode_index"])
        n = int(ep["length"])
        ep_flat_start[ep_idx] = flat
        for fi in range(n):
            rows.append((ep_idx + ep_offset, fi, flat))
            flat += 1
    assert flat == total_frames

    # Write index once (identical across cams, so "cam" column expands per-cam on the fly)
    index_df = pd.DataFrame(rows, columns=["episode_index", "frame_index", "flat_idx"])
    index_path = root / "index.parquet"
    root.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(index_path, index=False)
    print(f"[cache] Wrote {index_path} ({len(index_df)} rows)")

    # Estimate size
    bytes_per_cam = total_frames * IMG_SIZE * IMG_SIZE * 3
    gb_total = 4 * bytes_per_cam / 1e9
    print(f"[cache] Expected total on disk: {gb_total:.1f} GB")

    for cam in CAMERAS:
        memmap_path, _ = _cache_paths(cam, cache_root=root)
        if memmap_path.exists() and not force:
            size = memmap_path.stat().st_size
            if size == bytes_per_cam:
                print(f"[cache] {cam}: already cached ({size/1e9:.1f} GB), skipping")
                continue
            else:
                print(f"[cache] {cam}: stale ({size} vs expected {bytes_per_cam}), redoing")

        print(f"[cache] {cam}: allocating memmap ({bytes_per_cam/1e9:.1f} GB)")
        arr = np.memmap(
            memmap_path, dtype=np.uint8, mode="w+",
            shape=(total_frames, IMG_SIZE, IMG_SIZE, 3),
        )

        cam_key = f"observation.images.{cam}"
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"

        video_groups = episodes.groupby([chunk_col, file_col])
        pbar = tqdm(total=total_frames, desc=f"  {cam}", unit="frame")

        for (chunk_idx, file_idx), group_eps in video_groups:
            video_path = get_video_path(ds, cam_key, int(chunk_idx), int(file_idx))
            if not video_path.exists():
                print(f"  [cache] Missing: {video_path}")
                continue

            group_eps = group_eps.sort_values(from_ts_col)
            with VideoReader(video_path, NATIVE_W, NATIVE_H) as reader:
                for _, ep_row in group_eps.iterrows():
                    ep_idx = int(ep_row["episode_index"])
                    n_ep = int(ep_row["length"])
                    from_ts = float(ep_row[from_ts_col])
                    start_frame = round(from_ts * fps)

                    if reader._pos < start_frame:
                        reader.skip(start_frame - reader._pos)

                    base = ep_flat_start[ep_idx]
                    for local_idx in range(n_ep):
                        frame = reader.read_one()
                        if frame is None:
                            break
                        # cv2.resize with INTER_AREA for downscale quality
                        small = cv2.resize(
                            frame, (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        arr[base + local_idx] = small
                        pbar.update(1)

        pbar.close()
        arr.flush()
        del arr
        print(f"[cache] {cam}: done -> {memmap_path}")

    print("[cache] All cameras cached")


# -------- DATASET --------

class DistillDataset(Dataset):
    """Loads cached frames + v11-final teacher labels for one camera.
    Combines main (eternalmay33/01_02_03_merged_may-sim) + no_objects datasets.

    Expects `labels_df` pre-processed with columns:
      episode_index, frame_index, source ∈ {'main','no_obj'}, split ∈ {'train','val'},
      {cam}_{obj}_{cx,cy,conf,trust} for all (cam,obj).

    All filter logic (v8/v9/v10 + trust-gated interpolation) is already baked
    into the labels — nothing done in-memory here.

    Per-source conf_target rule (for BCE):
      - source='no_obj' OR (source='main' AND cam='gripper'): conf_target = trust
      - source='main' AND cam ∈ {left,right,top}:          conf_target = (teacher_conf >= 0.08)
    """

    def __init__(
        self,
        cam: str,
        labels_df: pd.DataFrame,
        split: str,                     # "train" or "val"
        memmap_main_path: Path,
        memmap_no_obj_path: Path,
        total_frames_main: int,
        total_frames_no_obj: int,
    ):
        self.cam = cam
        self.memmap_main_path = memmap_main_path
        self.memmap_no_obj_path = memmap_no_obj_path
        self.total_frames_main = total_frames_main
        self.total_frames_no_obj = total_frames_no_obj

        # Filter to split
        sub = labels_df.loc[labels_df["split"] == split].reset_index(drop=True)

        # Merge with per-source flat_idx caches
        main_idx = pd.read_parquet(CACHE_ROOT / "index.parquet")
        main_idx["source"] = "main"
        no_obj_idx = pd.read_parquet(CACHE_ROOT_NO_OBJECTS / "index.parquet")
        no_obj_idx["source"] = "no_obj"
        idx_all = pd.concat([main_idx, no_obj_idx], ignore_index=True)
        merged = sub.merge(idx_all, on=["episode_index", "frame_index", "source"], how="inner")
        if len(merged) != len(sub):
            raise RuntimeError(
                f"[{cam}/{split}] Label/cache mismatch: {len(sub)} labels vs "
                f"{len(merged)} matched"
            )

        # -- duck --
        duck_cx = merged[f"{cam}_duck_cx"].fillna(0.5).to_numpy(dtype=np.float32)
        duck_cy = merged[f"{cam}_duck_cy"].fillna(0.5).to_numpy(dtype=np.float32)
        duck_conf_raw = merged[f"{cam}_duck_conf"].fillna(0.0).to_numpy(dtype=np.float32)
        duck_trust = merged[f"{cam}_duck_trust"].fillna(0).to_numpy(dtype=np.float32)

        # -- cup --
        cup_cx = merged[f"{cam}_cup_cx"].fillna(0.5).to_numpy(dtype=np.float32)
        cup_cy = merged[f"{cam}_cup_cy"].fillna(0.5).to_numpy(dtype=np.float32)
        cup_conf_raw = merged[f"{cam}_cup_conf"].fillna(0.0).to_numpy(dtype=np.float32)
        cup_trust = merged[f"{cam}_cup_trust"].fillna(0).to_numpy(dtype=np.float32)

        # Conf targets = trust. Universally. Trust is the filter's binary
        # verdict "is there a real duck/cup in this frame, with a clean label?".
        # Raw teacher conf was already consumed during filtering (to decide
        # trust) — it's not a training signal. Second-guessing trust with a
        # conf_raw threshold leaks interpolated-on-trust=0 values into the
        # conf target, which taught the student to claim conf=1 on ~90% of
        # rejected side-cam rows (v1_baseline arm-base overconfidence bug,
        # 2026-04-21).
        source_arr = merged["source"].to_numpy()
        duck_conf_target = duck_trust.astype(np.float32)
        cup_conf_target = cup_trust.astype(np.float32)

        # Clip coords
        duck_cx = np.clip(duck_cx, 0.0, 1.0)
        duck_cy = np.clip(duck_cy, 0.0, 1.0)
        cup_cx = np.clip(cup_cx, 0.0, 1.0)
        cup_cy = np.clip(cup_cy, 0.0, 1.0)

        # Drop only if BOTH obj trust=0 on main source (rare).
        # For no_obj source, keep all rows — they ARE the negative examples we need.
        both_zero_trust = (duck_trust < 0.5) & (cup_trust < 0.5)
        is_no_obj = source_arr == "no_obj"
        drop_mask = both_zero_trust & (~is_no_obj)
        n_before = len(duck_trust)
        if drop_mask.any():
            keep = ~drop_mask
            merged = merged.loc[keep].reset_index(drop=True)
            duck_cx = duck_cx[keep]; duck_cy = duck_cy[keep]
            duck_conf_raw = duck_conf_raw[keep]; duck_trust = duck_trust[keep]
            duck_conf_target = duck_conf_target[keep]
            cup_cx = cup_cx[keep]; cup_cy = cup_cy[keep]
            cup_conf_raw = cup_conf_raw[keep]; cup_trust = cup_trust[keep]
            cup_conf_target = cup_conf_target[keep]
            source_arr = source_arr[keep]
        n_after = len(duck_trust)

        self.flat_idx = merged["flat_idx"].to_numpy(dtype=np.int64)
        self.sources = source_arr
        self.teacher_duck_conf = duck_conf_raw
        self.teacher_cup_conf = cup_conf_raw
        self.duck_trust = duck_trust
        self.cup_trust = cup_trust

        # Target layout: [duck_cx, duck_cy, duck_conf_target, cup_cx, cup_cy, cup_conf_target]
        # Coord loss uses trust as mask (stored separately on self); BCE uses conf_target.
        self.targets = np.stack(
            [duck_cx, duck_cy, duck_conf_target, cup_cx, cup_cy, cup_conf_target], axis=1
        ).astype(np.float32)
        self.coord_masks = np.stack([duck_trust, cup_trust], axis=1).astype(np.float32)

        n_no = int((source_arr == "no_obj").sum())
        n_main = int((source_arr == "main").sum())
        neg_duck = int((duck_conf_target < 0.5).sum())
        neg_cup = int((cup_conf_target < 0.5).sum())
        print(
            f"  [ds {cam}/{split}] N_in={n_before:,} dropped={n_before - n_after:,} "
            f"-> N={n_after:,}  (main={n_main:,}, no_obj={n_no:,})  "
            f"neg_duck_conf={neg_duck} neg_cup_conf={neg_cup}"
        )

        # Opened lazily per worker
        self._mm_main = None
        self._mm_no_obj = None

    def _get_mm(self, source: str):
        if source == "main":
            if self._mm_main is None:
                self._mm_main = np.memmap(
                    self.memmap_main_path, dtype=np.uint8, mode="r",
                    shape=(self.total_frames_main, IMG_SIZE, IMG_SIZE, 3),
                )
            return self._mm_main
        else:
            if self._mm_no_obj is None:
                self._mm_no_obj = np.memmap(
                    self.memmap_no_obj_path, dtype=np.uint8, mode="r",
                    shape=(self.total_frames_no_obj, IMG_SIZE, IMG_SIZE, 3),
                )
            return self._mm_no_obj

    def __len__(self):
        return len(self.flat_idx)

    def __getitem__(self, i):
        source = str(self.sources[i])
        fi = int(self.flat_idx[i])
        mm = self._get_mm(source)
        img = mm[fi]  # (224,224,3) uint8
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C,H,W)
        # We return (img, target, coord_mask, source_flag) — loader collates to tensors.
        return (
            torch.from_numpy(img),
            torch.from_numpy(self.targets[i]),
            torch.from_numpy(self.coord_masks[i]),
            0 if source == "main" else 1,  # int source flag for per-source val metrics
        )


# -------- LOSS --------

def distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> dict:
    """MSE on coords masked by coord_mask (=trust), BCE (or focal) on conf.

    pred/target shape: (B, 6) = [duck_cx, duck_cy, duck_conf, cup_cx, cup_cy, cup_conf].
    target[:, 2] / [:, 5] are binarized conf targets (for BCE).
    coord_mask shape: (B, 2) = [duck_trust, cup_trust]. Coord MSE weighted by these.
    If coord_mask is None (legacy behaviour), mask equals the conf targets.
    focal_gamma > 0 uses focal loss on the conf head (helps class imbalance).
    """
    # BCE on conf head uses conf target columns
    duck_conf_t = target[:, 2]
    cup_conf_t = target[:, 5]

    # Coord MSE masked by trust (or by conf target if mask not given)
    if coord_mask is None:
        duck_mask = duck_conf_t
        cup_mask = cup_conf_t
    else:
        duck_mask = coord_mask[:, 0]
        cup_mask = coord_mask[:, 1]

    duck_mse = F.mse_loss(pred[:, 0:2], target[:, 0:2], reduction="none").sum(dim=1)
    cup_mse = F.mse_loss(pred[:, 3:5], target[:, 3:5], reduction="none").sum(dim=1)

    duck_n = duck_mask.sum().clamp_min(1.0)
    cup_n = cup_mask.sum().clamp_min(1.0)
    coord_loss = (duck_mse * duck_mask).sum() / duck_n \
                 + (cup_mse * cup_mask).sum() / cup_n

    # Conf head loss: BCE or focal BCE
    eps = 1e-6
    dc_p = pred[:, 2].clamp(eps, 1 - eps)
    cc_p = pred[:, 5].clamp(eps, 1 - eps)
    if focal_gamma > 0.0:
        # Focal BCE per sample, then mean
        def _focal(p, t, gamma):
            p_t = torch.where(t > 0.5, p, 1 - p)
            ce = -torch.log(p_t)
            return (ce * (1.0 - p_t).pow(gamma)).mean()
        conf_loss = _focal(dc_p, duck_conf_t, focal_gamma) + _focal(cc_p, cup_conf_t, focal_gamma)
    else:
        conf_loss = F.binary_cross_entropy(dc_p, duck_conf_t) \
                    + F.binary_cross_entropy(cc_p, cup_conf_t)

    total = coord_loss + conf_loss
    return {"total": total, "coord": coord_loss.detach(), "conf": conf_loss.detach()}


# -------- METRICS --------

@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_coord = 0.0
    total_conf = 0.0
    n_batches = 0

    all_pred = []
    all_tgt = []
    all_src = []  # 0 = main, 1 = no_obj

    pbar = tqdm(loader, desc="  val", unit="batch", leave=False)
    for img, tgt, coord_mask, src in pbar:
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        cm = coord_mask.to(device, non_blocking=True)
        out = model(img)
        losses = distill_loss(out, tgt, coord_mask=cm)
        total_loss += losses["total"].item()
        total_coord += losses["coord"].item()
        total_conf += losses["conf"].item()
        n_batches += 1

        all_pred.append(out.cpu().numpy())
        all_tgt.append(tgt.cpu().numpy())
        all_src.append(src.numpy() if hasattr(src, "numpy") else np.asarray(src))

        pbar.set_postfix(
            total=f"{losses['total'].item():.4f}",
            coord=f"{losses['coord'].item():.4f}",
            conf=f"{losses['conf'].item():.4f}",
        )

    pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 6))
    tgt = np.concatenate(all_tgt, axis=0) if all_tgt else np.zeros((0, 6))
    src_arr = np.concatenate(all_src, axis=0) if all_src else np.zeros((0,), dtype=np.int64)

    metrics = {
        "loss_total": total_loss / max(1, n_batches),
        "loss_coord": total_coord / max(1, n_batches),
        "loss_conf": total_conf / max(1, n_batches),
    }

    def _per_obj_metrics(pred_sub, tgt_sub, prefix: str):
        for obj_idx, obj in zip([(0, 1, 2), (3, 4, 5)], OBJECTS):
            cx_p = pred_sub[:, obj_idx[0]]
            cy_p = pred_sub[:, obj_idx[1]]
            cf_p = pred_sub[:, obj_idx[2]]
            cx_t = tgt_sub[:, obj_idx[0]]
            cy_t = tgt_sub[:, obj_idx[1]]
            cf_t = tgt_sub[:, obj_idx[2]]

            pos_mask = cf_t > 0.5
            if pos_mask.sum() > 0:
                err_x = np.abs(cx_p[pos_mask] - cx_t[pos_mask]) * NATIVE_W
                err_y = np.abs(cy_p[pos_mask] - cy_t[pos_mask]) * NATIVE_H
                err = np.sqrt(err_x ** 2 + err_y ** 2)
                metrics[f"{prefix}{obj}_err_median_px"] = float(np.median(err))
                metrics[f"{prefix}{obj}_err_p95_px"] = float(np.percentile(err, 95))
            else:
                metrics[f"{prefix}{obj}_err_median_px"] = float("nan")
                metrics[f"{prefix}{obj}_err_p95_px"] = float("nan")

            metrics[f"{prefix}{obj}_detection_rate"] = float(
                (cf_p[pos_mask] > 0.5).mean() if pos_mask.sum() > 0 else float("nan")
            )
            neg_mask = ~pos_mask
            metrics[f"{prefix}{obj}_false_positive_rate"] = float(
                (cf_p[neg_mask] > 0.5).mean() if neg_mask.sum() > 0 else float("nan")
            )

    # Overall (combined)
    _per_obj_metrics(pred, tgt, prefix="")
    # Per-source
    main_mask = src_arr == 0
    no_obj_mask = src_arr == 1
    if main_mask.sum() > 0:
        _per_obj_metrics(pred[main_mask], tgt[main_mask], prefix="main_")
    if no_obj_mask.sum() > 0:
        _per_obj_metrics(pred[no_obj_mask], tgt[no_obj_mask], prefix="no_obj_")

    return metrics


# -------- TRAIN --------

@dataclass
class TrainConfig:
    cam: str
    run_name: str = "v11_baseline"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 40
    min_epochs: int = 15
    patience: int = 10
    num_workers: int = 4
    seed: int = 42
    val_frac: float = 0.15
    model_backbone: str = "mobilenet_v3_small"  # or mobilenet_v3_large
    focal_gamma: float = 0.0                    # 0.0 = standard BCE
    augment: bool = False                       # ColorJitter 0.2
    resume_from: str | None = None              # path to existing .pt to extend from


def _make_loaders(cam: str, cfg: TrainConfig):
    """Build train/val DataLoaders using both main and no_obj datasets."""
    labels_df = load_all_labels(seed=cfg.seed, val_frac=cfg.val_frac)

    _, main_eps = load_dataset_meta(DATASET_PATH)
    total_main = _count_total_frames(main_eps)
    _, no_obj_eps = load_dataset_meta(DATASET_PATH_NO_OBJECTS)
    total_no_obj = _count_total_frames(no_obj_eps)

    mm_main_path, _ = _cache_paths(cam, cache_root=CACHE_ROOT)
    mm_no_obj_path, _ = _cache_paths(cam, cache_root=CACHE_ROOT_NO_OBJECTS)

    train_ds = DistillDataset(
        cam, labels_df, "train",
        mm_main_path, mm_no_obj_path,
        total_main, total_no_obj,
    )
    val_ds = DistillDataset(
        cam, labels_df, "val",
        mm_main_path, mm_no_obj_path,
        total_main, total_no_obj,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_ds, val_ds, train_loader, val_loader


def _apply_augment(img_batch: torch.Tensor) -> torch.Tensor:
    """Conservative color jitter on a (B,3,H,W) [0,1] batch. No spatial transform
    (would require recomputing target coords)."""
    # In-place-ish: sample per-batch jitter factors, apply uniformly per-channel.
    device = img_batch.device
    b = img_batch.shape[0]
    brightness = (1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.4)  # ±0.2
    contrast   = (1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.4)
    hue_shift  = (torch.rand(b, 3, 1, 1, device=device) - 0.5) * 0.1         # per-channel tint ±0.05
    out = img_batch * brightness
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out + hue_shift
    return out.clamp(0.0, 1.0)


def train_one(cam: str, cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = TRAINING_ROOT / cfg.run_name / cam
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\n[train] ==== camera: {cam}  run: {cfg.run_name} ====")
    print(f"[train] out_dir: {out_dir}")
    print(f"[train] config: {cfg}")

    train_ds, val_ds, train_loader, val_loader = _make_loaders(cam, cfg)
    print(f"[train] train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    # Model
    model = DistilledDetector(backbone=cfg.model_backbone).to(device)
    print(f"[train] model params: {count_params(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.max_epochs)

    # -- RESUME from existing checkpoint (extension mode) --
    # Loads model weights; if the checkpoint also has optim/sched state, restores
    # them. Fresh cosine annealing starts over the NEW --max-epochs window.
    # Seeds best_score/best_epoch from existing best.pt so we only overwrite
    # best.pt when extension actually beats it.
    csv_start_offset = 0
    csv_header_written = False
    if cfg.resume_from:
        rp = Path(cfg.resume_from)
        if not rp.exists():
            raise FileNotFoundError(f"--resume-from path not found: {rp}")
        ck = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[train] RESUMED model weights from {rp}")
        if "optim" in ck:
            try:
                optim.load_state_dict(ck["optim"])
                print(f"[train] resumed optimizer state")
            except Exception as e:
                print(f"[train] optim state load failed ({e}); keeping fresh AdamW")
        else:
            print(f"[train] checkpoint has no optimizer state; fresh AdamW at lr={cfg.lr}")
        if "sched" in ck:
            try:
                sched.load_state_dict(ck["sched"])
                print(f"[train] resumed scheduler state")
            except Exception:
                pass

    # Early stop on main pixel error (sum of duck + cup medians). Smaller is better.
    best_score = math.inf
    best_epoch = -1
    patience_left = cfg.patience
    history = []

    # Metrics CSV: fresh for cold start, append-and-continue for resume.
    metrics_csv = out_dir / "metrics.csv"
    if cfg.resume_from and metrics_csv.exists():
        prev_df = pd.read_csv(metrics_csv)
        if len(prev_df) > 0:
            csv_start_offset = int(prev_df["epoch"].max())
            # Seed history so the training.png plot shows the full curve.
            history = prev_df.to_dict("records")
        csv_fp = open(metrics_csv, "a", newline="")
        csv_header_written = True  # already present on disk
        # Seed best_score from existing best.pt so extension only overwrites on real improvement.
        best_pt = out_dir / "best.pt"
        if best_pt.exists():
            try:
                prev_best = torch.load(best_pt, map_location="cpu", weights_only=False)
                pv = prev_best.get("val", {}) or {}
                pd_med = pv.get("main_duck_err_median_px", float("nan"))
                pc_med = pv.get("main_cup_err_median_px", float("nan"))
                if np.isfinite(pd_med) and np.isfinite(pc_med):
                    best_score = float(pd_med) + float(pc_med)
                    best_epoch = int(prev_best.get("epoch", -1))
                    print(f"[train] existing best.pt: score={best_score:.2f} @ ep{best_epoch}")
            except Exception as e:
                print(f"[train] could not read prev best.pt ({e}); starting best_score=inf")
        print(f"[train] resume: csv_start_offset={csv_start_offset} (continuing epoch numbering)")
    else:
        csv_fp = open(metrics_csv, "w", newline="")
    csv_writer = None

    t_start = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{cfg.max_epochs}", unit="batch")
        sum_total = sum_coord = sum_conf = 0.0
        n_batches = 0
        for img, tgt, cm, _src in pbar:
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            cm = cm.to(device, non_blocking=True)
            if cfg.augment:
                img = _apply_augment(img)
            out = model(img)
            losses = distill_loss(out, tgt, coord_mask=cm, focal_gamma=cfg.focal_gamma)
            loss = losses["total"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            sum_total += loss.item()
            sum_coord += losses["coord"].item()
            sum_conf += losses["conf"].item()
            n_batches += 1
            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                coord=f"{losses['coord'].item():.4f}",
                conf=f"{losses['conf'].item():.4f}",
            )

        sched.step()

        train_total = sum_total / max(1, n_batches)
        train_coord = sum_coord / max(1, n_batches)
        train_conf = sum_conf / max(1, n_batches)

        val_metrics = eval_epoch(model, val_loader, device)

        # Monitor metric: main-source pixel error sum (excludes no_obj which has no coords)
        duck_med_main = val_metrics.get("main_duck_err_median_px", val_metrics.get("duck_err_median_px", float("nan")))
        cup_med_main = val_metrics.get("main_cup_err_median_px", val_metrics.get("cup_err_median_px", float("nan")))
        def _num(x):
            return x if isinstance(x, (int, float)) and np.isfinite(x) else 9999.0
        score = _num(duck_med_main) + _num(cup_med_main)

        row = {
            "epoch": csv_start_offset + epoch + 1,
            "lr": optim.param_groups[0]["lr"],
            "train_total": train_total,
            "train_coord": train_coord,
            "train_conf": train_conf,
            "val_total": val_metrics["loss_total"],
            "val_coord": val_metrics["loss_coord"],
            "val_conf": val_metrics["loss_conf"],
            "score_pixel": score,
            "duck_err_median_px": val_metrics.get("duck_err_median_px", float("nan")),
            "duck_err_p95_px": val_metrics.get("duck_err_p95_px", float("nan")),
            "cup_err_median_px": val_metrics.get("cup_err_median_px", float("nan")),
            "cup_err_p95_px": val_metrics.get("cup_err_p95_px", float("nan")),
            "main_duck_err_median_px": val_metrics.get("main_duck_err_median_px", float("nan")),
            "main_cup_err_median_px": val_metrics.get("main_cup_err_median_px", float("nan")),
            "duck_detection_rate": val_metrics.get("duck_detection_rate", float("nan")),
            "cup_detection_rate": val_metrics.get("cup_detection_rate", float("nan")),
            "duck_false_positive_rate": val_metrics.get("duck_false_positive_rate", float("nan")),
            "cup_false_positive_rate": val_metrics.get("cup_false_positive_rate", float("nan")),
            "no_obj_duck_false_positive_rate": val_metrics.get("no_obj_duck_false_positive_rate", float("nan")),
            "no_obj_cup_false_positive_rate": val_metrics.get("no_obj_cup_false_positive_rate", float("nan")),
        }
        history.append(row)

        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_fp, fieldnames=list(row.keys()))
            if not csv_header_written:
                csv_writer.writeheader()
                csv_header_written = True
        csv_writer.writerow(row)
        csv_fp.flush()

        abs_epoch = csv_start_offset + epoch + 1
        print(
            f"[train] {cfg.run_name}/{cam} ep {abs_epoch}: "
            f"train {train_total:.4f}  "
            f"val pixel-score {score:.1f}  "
            f"main d/c med {duck_med_main:.1f}/{cup_med_main:.1f}  "
            f"no_obj fp d/c {val_metrics.get('no_obj_duck_false_positive_rate', float('nan')):.3f}/"
            f"{val_metrics.get('no_obj_cup_false_positive_rate', float('nan')):.3f}"
        )

        torch.save({"model": model.state_dict(), "epoch": abs_epoch,
                    "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "val": val_metrics, "config": asdict(cfg)},
                   out_dir / "last.pt")

        _plot_history(history, out_dir / "training.png", cam)

        improved = score < best_score
        can_stop = (epoch + 1) >= cfg.min_epochs

        if improved:
            best_score = score
            best_epoch = abs_epoch
            torch.save({"model": model.state_dict(), "epoch": abs_epoch,
                        "optim": optim.state_dict(), "sched": sched.state_dict(),
                        "val": val_metrics, "config": asdict(cfg)},
                       out_dir / "best.pt")
            patience_left = cfg.patience
            print(f"[train] new best pixel-score={best_score:.2f}, saved best.pt")
        else:
            if can_stop:
                patience_left -= 1
                print(f"[train] no improvement, patience={patience_left}")
                if patience_left <= 0:
                    print(f"[train] early stop at epoch {abs_epoch} (best @ ep{best_epoch})")
                    break
            else:
                print(f"[train] below min_epochs ({cfg.min_epochs}), continuing")

    csv_fp.close()
    elapsed = time.perf_counter() - t_start
    print(f"[train] {cfg.run_name}/{cam} done in {elapsed:.1f}s, best pixel-score={best_score:.2f} @ ep {best_epoch}")

    return {
        "cam": cam,
        "run_name": cfg.run_name,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "elapsed": elapsed,
        "history": history,
        "final_metrics": history[best_epoch - 1] if best_epoch > 0 else history[-1],
    }


def _plot_history(history: list[dict], path: Path, cam: str):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_total"], label="train total", color="C0")
    ax.plot(df["epoch"], df["val_total"], label="val total", color="C1")
    ax.plot(df["epoch"], df["train_coord"], label="train coord", color="C0", ls="--", alpha=0.5)
    ax.plot(df["epoch"], df["val_coord"], label="val coord", color="C1", ls="--", alpha=0.5)
    ax.plot(df["epoch"], df["train_conf"], label="train conf", color="C0", ls=":", alpha=0.5)
    ax.plot(df["epoch"], df["val_conf"], label="val conf", color="C1", ls=":", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{cam}: loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Pixel error
    ax = axes[1]
    ax.plot(df["epoch"], df["duck_err_median_px"], label="duck median", color="C2")
    ax.plot(df["epoch"], df["duck_err_p95_px"], label="duck p95", color="C2", ls="--", alpha=0.5)
    ax.plot(df["epoch"], df["cup_err_median_px"], label="cup median", color="C3")
    ax.plot(df["epoch"], df["cup_err_p95_px"], label="cup p95", color="C3", ls="--", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("pixel error")
    ax.set_title(f"{cam}: val pixel error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Detection rate
    ax = axes[2]
    ax.plot(df["epoch"], df["duck_detection_rate"], label="duck", color="C2")
    ax.plot(df["epoch"], df["cup_detection_rate"], label="cup", color="C3")
    ax.set_xlabel("epoch"); ax.set_ylabel("detection rate")
    ax.set_title(f"{cam}: val detection rate"); ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # False positive rate
    ax = axes[3]
    ax.plot(df["epoch"], df["duck_false_positive_rate"], label="duck", color="C2")
    ax.plot(df["epoch"], df["cup_false_positive_rate"], label="cup", color="C3")
    ax.set_xlabel("epoch"); ax.set_ylabel("false positive rate")
    ax.set_title(f"{cam}: val false positive rate"); ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Distilled detector — {cam} (baseline raw labels)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(cam: str, checkpoint: Path, run_name: str = "eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_cfg = TrainConfig(cam=cam, run_name=run_name)
    _, val_ds, _, val_loader = _make_loaders(cam, dummy_cfg)

    model = DistilledDetector().to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    metrics = eval_epoch(model, val_loader, device)
    print(json.dumps(metrics, indent=2))
    return metrics


# -------- CLI --------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_cache = sub.add_parser("cache", help="Decode all frames to disk cache")
    sub_cache.add_argument("--force", action="store_true")
    sub_cache.add_argument("--no-objects", action="store_true",
                           help="Cache the distill_no_objects dataset instead of main")

    sub_train = sub.add_parser("train", help="Train per-cam distilled detector")
    sub_train.add_argument("--cam", choices=CAMERAS, default=None)
    sub_train.add_argument("--all", action="store_true")
    sub_train.add_argument("--run", type=str, default="m1_baseline",
                           help="Run name — output dir: training/<run>/<cam>/")
    sub_train.add_argument("--max-epochs", type=int, default=40)
    sub_train.add_argument("--min-epochs", type=int, default=15)
    sub_train.add_argument("--patience", type=int, default=10)
    sub_train.add_argument("--batch-size", type=int, default=128)
    sub_train.add_argument("--lr", type=float, default=1e-3)
    sub_train.add_argument("--num-workers", type=int, default=4)
    sub_train.add_argument("--model", type=str, default="mobilenet_v3_small",
                           choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"])
    sub_train.add_argument("--focal-gamma", type=float, default=0.0,
                           help="Focal loss gamma for conf head; 0 = standard BCE")
    sub_train.add_argument("--augment", action="store_true",
                           help="Enable conservative color jitter augmentation")
    sub_train.add_argument("--resume-from", type=str, default=None,
                           help="Path to .pt to resume from (extension mode). "
                                "Loads model weights; also restores optim/sched state "
                                "if present in the checkpoint. Metrics.csv is appended "
                                "with continued epoch numbering. best.pt is only "
                                "overwritten if extension beats existing best.")

    sub_eval = sub.add_parser("eval", help="Evaluate a checkpoint on val")
    sub_eval.add_argument("--cam", choices=CAMERAS, required=True)
    sub_eval.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    if args.cmd == "cache":
        if getattr(args, "no_objects", False):
            cache_frames(
                force=args.force,
                dataset_path=DATASET_PATH_NO_OBJECTS,
                cache_root=CACHE_ROOT_NO_OBJECTS,
                ep_offset=NO_OBJECTS_EP_OFFSET,
            )
        else:
            cache_frames(force=args.force)
    elif args.cmd == "train":
        if args.all:
            cams = CAMERAS
        else:
            if not args.cam:
                raise SystemExit("pass --cam or --all")
            cams = [args.cam]

        results = []
        for c in cams:
            cfg = TrainConfig(
                cam=c,
                run_name=args.run,
                lr=args.lr,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                model_backbone=args.model,
                focal_gamma=args.focal_gamma,
                augment=args.augment,
                resume_from=args.resume_from,
            )
            r = train_one(c, cfg)
            results.append(r)

        # Summary table
        print(f"\n=== FINAL RESULTS ({args.run}) ===")
        for r in results:
            fm = r["final_metrics"]
            print(
                f"{r['cam']:8s}  pixel-score={r['best_score']:.1f} @ ep{r['best_epoch']:2d}  "
                f"duck med={fm.get('duck_err_median_px', float('nan')):.1f}px  "
                f"cup med={fm.get('cup_err_median_px', float('nan')):.1f}px  "
                f"fp d={fm.get('duck_false_positive_rate', float('nan')):.2f}/"
                f"c={fm.get('cup_false_positive_rate', float('nan')):.2f}  "
                f"time={r['elapsed']:.0f}s"
            )
    elif args.cmd == "eval":
        evaluate(args.cam, Path(args.checkpoint))


if __name__ == "__main__":
    main()
