"""Post-process the distill_no_objects dataset: null absent-object detections,
render verification videos, and prepare training-ready label parquet.

Dataset: eternalmay33/distill_no_objects (6 episodes, 10767 frames)
Episode object convention:
  ep 0, 1  -> no objects on table
  ep 2, 3  -> cup only (duck absent)
  ep 4, 5  -> duck only (cup absent)

Flow:
  1. Read detection_results.parquet (produced by `process_dataset`).
  2. For each episode, null out the absent object(s) — set cx/cy/conf/bbox to NaN.
  3. Save cleaned parquet as detection_results_clean.parquet.
  4. Render verification videos per episode (2x2 cam grid, H.264, teacher bbox
     overlay for the present object only).

Usage:
    python -u scripts/distill_no_objects_prepare.py
    python -u scripts/distill_no_objects_prepare.py --skip-videos  # parquet only
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.process_dataset import (
    VideoReader,
    get_video_path,
    load_dataset_meta,
)


DATASET_REPO = "eternalmay33/distill_no_objects"
CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]
NATIVE_W, NATIVE_H = 640, 480
FPS = 30

# Episode -> which object is PRESENT (others nulled)
EP_PRESENT: dict[int, set[str]] = {
    0: set(),           # no objects
    1: set(),           # no objects
    2: {"cup"},         # cup only
    3: {"cup"},         # cup only
    4: {"duck"},        # duck only
    5: {"duck"},        # duck only
}

# Bbox/label columns produced by process_dataset
FIELDS = ["cx", "cy", "conf", "x1", "y1", "x2", "y2"]

OUTPUT_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/no_objects"
)


def null_absent_objects(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, null (set NaN) columns belonging to the object that is
    supposed to be absent for that episode."""
    df = df.copy()
    for ep_idx, present_set in EP_PRESENT.items():
        absent_objs = [o for o in OBJECTS if o not in present_set]
        if not absent_objs:
            continue
        mask = df["episode_index"] == ep_idx
        cols_to_null = []
        for cam in CAMERAS:
            for obj in absent_objs:
                for f in FIELDS:
                    col = f"{cam}_{obj}_{f}"
                    if col in df.columns:
                        cols_to_null.append(col)
        df.loc[mask, cols_to_null] = np.nan
    return df


def interpolate_present_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate cx/cy/conf/bbox cols for PRESENT objects per episode.

    Fills stride-skipped frames so training labels are dense across every
    frame. Does NOT touch absent (nulled) objects — those stay NaN.
    """
    df = df.copy()
    for ep_idx, present_set in EP_PRESENT.items():
        if not present_set:
            continue
        ep_mask = df["episode_index"] == ep_idx
        if not ep_mask.any():
            continue
        cols_to_interp = []
        for cam in CAMERAS:
            for obj in present_set:
                for f in FIELDS:
                    col = f"{cam}_{obj}_{f}"
                    if col in df.columns:
                        cols_to_interp.append(col)
        sub = df.loc[ep_mask, cols_to_interp].interpolate(
            method="linear", limit_direction="both",
        )
        df.loc[ep_mask, cols_to_interp] = sub.values
    return df


def null_failed_dino_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Null cx/cy/conf/bbox on rows that look like DINO failures.

    A DINO "failure" stride row is detected by simple heuristics on the raw
    stride output: tiny bbox area, bbox at image edge (y2<0.05 or y2>0.97),
    or very low conf. These are the failed detections that pollute linear
    interpolation (bbox morphing toward (0,0) corner).

    Applied ONLY to present objects per episode.
    """
    df = df.copy()
    min_area = 0.002      # bbox smaller than 0.2% of frame area -> garbage
    y2_lo = 0.05          # bbox bottom at top edge -> failed
    y2_hi = 0.97          # bbox bottom at bottom edge -> failed (arm lock)
    nulled = {}
    for ep_idx, present_set in EP_PRESENT.items():
        if not present_set:
            continue
        ep_mask = df["episode_index"] == ep_idx
        for cam in CAMERAS:
            for obj in present_set:
                x1c = f"{cam}_{obj}_x1"
                y1c = f"{cam}_{obj}_y1"
                x2c = f"{cam}_{obj}_x2"
                y2c = f"{cam}_{obj}_y2"
                if not all(c in df.columns for c in [x1c, y1c, x2c, y2c]):
                    continue
                x1 = df.loc[ep_mask, x1c]
                y1 = df.loc[ep_mask, y1c]
                x2 = df.loc[ep_mask, x2c]
                y2 = df.loc[ep_mask, y2c]
                area = (x2 - x1).clip(lower=0) * (y2 - y1).clip(lower=0)
                fail_mask = (
                    (area < min_area) | (y2 < y2_lo) | (y2 > y2_hi)
                ) & x1.notna()
                row_idx = df.index[ep_mask][fail_mask.values]
                if len(row_idx) == 0:
                    continue
                cols_to_null = [
                    f"{cam}_{obj}_{f}" for f in FIELDS if f"{cam}_{obj}_{f}" in df.columns
                ]
                df.loc[row_idx, cols_to_null] = np.nan
                key = f"ep{ep_idx}/{cam}/{obj}"
                nulled[key] = int(len(row_idx))
    if nulled:
        print("[null-failed] DINO-failure stride rows nulled:")
        for k, n in nulled.items():
            print(f"    {k}: {n}")
    return df


def draw_bbox(img: np.ndarray, bbox: tuple[int, int, int, int],
              color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    ty = max(12, y1 - 4)
    cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_episode_video(
    ep_idx: int,
    det_ep: pd.DataFrame,
    ds_path: Path,
    episodes: pd.DataFrame,
    out_dir: Path,
    trust_ep: pd.DataFrame | None = None,
) -> Path:
    """Render 2x2 cam grid with teacher bbox overlays (present-obj only) for this episode."""
    # Decode all frames per cam
    cam_frames: dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        cam_key = f"observation.images.{cam}"
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"
        row = episodes[episodes["episode_index"] == ep_idx].iloc[0]
        chunk = int(row[chunk_col])
        file_idx = int(row[file_col])
        from_ts = float(row[from_ts_col])
        length = int(row["length"])
        video_path = get_video_path(ds_path, cam_key, chunk, file_idx)
        start_frame = round(from_ts * FPS)
        arr = np.empty((length, NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
        with VideoReader(video_path, NATIVE_W, NATIVE_H) as r:
            r.skip(start_frame)
            for i in range(length):
                fr = r.read_one()
                if fr is None:
                    arr = arr[:i]
                    break
                arr[i] = fr
        cam_frames[cam] = arr
    T = min(len(cam_frames[c]) for c in CAMERAS)

    # Build per-frame teacher lookup
    det_by_frame = det_ep.set_index("frame_index")
    trust_by_frame = trust_ep.set_index("frame_index") if trust_ep is not None else None

    grid_w = 2 * NATIVE_W
    grid_h = 2 * NATIVE_H
    tmp_path = out_dir / f"ep{ep_idx}.tmp.mp4"
    final_path = out_dir / f"ep{ep_idx}.mp4"

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open {tmp_path}")

    layout = [("left", 0, 0), ("right", 0, 1), ("top", 1, 0), ("gripper", 1, 1)]
    present = EP_PRESENT[ep_idx]
    present_str = ",".join(sorted(present)) if present else "none"

    for t in range(T):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        trow = det_by_frame.loc[t] if t in det_by_frame.index else None
        srow = trust_by_frame.loc[t] if (trust_by_frame is not None and t in trust_by_frame.index) else None
        for cam, rr, cc in layout:
            img = cam_frames[cam][t].copy()
            if trow is not None:
                for obj, color_ok, color_bad in [
                    ("duck", (0, 255, 120), (60, 60, 255)),   # green accept, red reject
                    ("cup",  (0, 255, 120), (60, 60, 255)),
                ]:
                    if obj not in present:
                        continue
                    x1 = trow.get(f"{cam}_{obj}_x1", np.nan)
                    y1 = trow.get(f"{cam}_{obj}_y1", np.nan)
                    x2 = trow.get(f"{cam}_{obj}_x2", np.nan)
                    y2 = trow.get(f"{cam}_{obj}_y2", np.nan)
                    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                        continue
                    # Trust from v10 filter. Apply v9 top_duck y2 rule inline.
                    trust = 1
                    if srow is not None:
                        t_val = srow.get(f"{cam}_{obj}_trust", 1)
                        try:
                            trust = int(t_val) if pd.notna(t_val) else 1
                        except (TypeError, ValueError):
                            trust = 1
                    if cam == "top" and obj == "duck" and float(y2) > 0.85:
                        trust = 0  # v9 reject
                    bx1 = int(float(x1) * NATIVE_W)
                    by1 = int(float(y1) * NATIVE_H)
                    bx2 = int(float(x2) * NATIVE_W)
                    by2 = int(float(y2) * NATIVE_H)
                    conf = trow.get(f"{cam}_{obj}_conf", np.nan)
                    color = color_ok if trust == 1 else color_bad
                    label = f"{obj} {float(conf):.2f}" if np.isfinite(conf) else obj
                    if trust == 0:
                        label = "MASKED " + label
                    draw_bbox(img, (bx1, by1, bx2, by2), color, label)
            # cam label + present flag
            txt = f"{cam}  ep{ep_idx} (present: {present_str})"
            cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            canvas[rr * NATIVE_H:(rr + 1) * NATIVE_H, cc * NATIVE_W:(cc + 1) * NATIVE_W] = img
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()

    # ffmpeg -> H.264 yuv420p
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "22", "-preset", "fast",
        str(final_path),
    ], check=True)
    try:
        tmp_path.unlink()
    except Exception:
        pass
    print(f"[video] ep{ep_idx}: {T} frames -> {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    ds_path = resolve_dataset_path(DATASET_REPO)
    raw_path = ds_path / "detection_results.parquet"
    clean_path = ds_path / "detection_results_clean.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing: {raw_path}. Run process_dataset first.")

    print(f"Reading {raw_path}...")
    det = pd.read_parquet(raw_path)
    print(f"  {len(det)} rows, episodes: {sorted(det['episode_index'].unique())}")

    print("Nulling absent-object columns per episode...")
    det_clean = null_absent_objects(det)
    print("Nulling failed DINO stride detections (bbox garbage, edge/tiny)...")
    det_clean = null_failed_dino_detections(det_clean)
    print("Interpolating present-object bboxes across stride gaps...")
    det_clean = interpolate_present_objects(det_clean)
    det_clean.to_parquet(clean_path, index=False)
    print(f"  saved {clean_path}")

    # Sanity: show per-episode non-null count for present-obj cx
    for ep_idx in sorted(det_clean["episode_index"].unique()):
        sub = det_clean[det_clean["episode_index"] == ep_idx]
        counts = {}
        for obj in OBJECTS:
            cx_col = f"left_{obj}_cx"
            if cx_col in sub.columns:
                counts[obj] = int(sub[cx_col].notna().sum())
        print(f"  ep{ep_idx}: {counts}  present={EP_PRESENT[int(ep_idx)]}")

    if not args.skip_videos:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        _, episodes = load_dataset_meta(ds_path)
        trust_path = Path("/home/may33/.cache/vbti/detection_labels_final_no_objects.parquet")
        trust_df = pd.read_parquet(trust_path) if trust_path.exists() else None
        if trust_df is not None:
            print(f"  trust parquet loaded: {trust_path}")
        for ep_idx in sorted(det_clean["episode_index"].unique()):
            ep_idx = int(ep_idx)
            det_ep = det_clean[det_clean["episode_index"] == ep_idx]
            trust_ep = trust_df[trust_df["episode_index"] == ep_idx] if trust_df is not None else None
            render_episode_video(ep_idx, det_ep, ds_path, episodes, OUTPUT_ROOT, trust_ep=trust_ep)
        print(f"\n[done] videos in: {OUTPUT_ROOT}")

    print(f"[done] clean parquet: {clean_path}")


if __name__ == "__main__":
    main()
