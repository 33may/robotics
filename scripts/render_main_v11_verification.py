"""Render 10 random main-dataset episodes (01_02_03_merged_may-sim) with the
v11 final filter applied, so user can eyeball data quality before training.

Format matches the no_objects verification videos:
  - 2x2 cam grid (left, right, top, gripper)
  - Accepted bbox = green
  - Rejected bbox = red with "MASKED" label
  - H.264 encoded, Obsidian-playable

Output: distillation/data_analysis/main_v11_verification/ep{N}.mp4
"""
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vbti.logic.detection.process_dataset import (
    VideoReader,
    get_video_path,
    load_dataset_meta,
)

DATASET = Path(
    "/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim"
)
FINAL_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_final.parquet")
OUT_DIR = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/data_analysis/main_v11_verification"
)

CAMERAS = ["left", "right", "top", "gripper"]
NATIVE_W, NATIVE_H = 640, 480
FPS = 30

COLOR_ACCEPTED = (0, 255, 120)  # green
COLOR_REJECTED = (60, 60, 255)  # red


def draw_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    ty = max(14, y1 - 4)
    cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_episode(
    ep_idx: int,
    det_ep: pd.DataFrame,
    episodes: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Decode frames, overlay bboxes colored by trust, write H.264 mp4."""
    # Decode all frames per cam
    cam_frames: dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        cam_key = f"observation.images.{cam}"
        row = episodes[episodes["episode_index"] == ep_idx].iloc[0]
        chunk = int(row[f"videos/{cam_key}/chunk_index"])
        file_idx = int(row[f"videos/{cam_key}/file_index"])
        from_ts = float(row[f"videos/{cam_key}/from_timestamp"])
        length = int(row["length"])
        video_path = get_video_path(DATASET, cam_key, chunk, file_idx)
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

    det_by_frame = det_ep.set_index("frame_index")

    grid_w = 2 * NATIVE_W
    grid_h = 2 * NATIVE_H
    tmp_path = out_dir / f"ep{ep_idx:03d}.tmp.mp4"
    final_path = out_dir / f"ep{ep_idx:03d}.mp4"

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open {tmp_path}")

    layout = [("left", 0, 0), ("right", 0, 1), ("top", 1, 0), ("gripper", 1, 1)]

    for t in range(T):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        if t in det_by_frame.index:
            trow = det_by_frame.loc[t]
        else:
            trow = None
        for cam, rr, cc in layout:
            img = cam_frames[cam][t].copy()
            if trow is not None:
                for obj in ("duck", "cup"):
                    x1 = trow.get(f"{cam}_{obj}_x1", np.nan)
                    y1 = trow.get(f"{cam}_{obj}_y1", np.nan)
                    x2 = trow.get(f"{cam}_{obj}_x2", np.nan)
                    y2 = trow.get(f"{cam}_{obj}_y2", np.nan)
                    if not (pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2)):
                        continue
                    conf = trow.get(f"{cam}_{obj}_conf", np.nan)
                    trust_val = trow.get(f"{cam}_{obj}_trust", 1)
                    try:
                        trust = int(trust_val) if pd.notna(trust_val) else 1
                    except (TypeError, ValueError):
                        trust = 1

                    bx1 = int(float(x1) * NATIVE_W)
                    by1 = int(float(y1) * NATIVE_H)
                    bx2 = int(float(x2) * NATIVE_W)
                    by2 = int(float(y2) * NATIVE_H)
                    conf_f = float(conf) if pd.notna(conf) else 0.0
                    color = COLOR_ACCEPTED if trust == 1 else COLOR_REJECTED
                    prefix = "MASKED " if trust == 0 else ""
                    label = f"{prefix}{obj} {conf_f:.2f}"
                    draw_bbox(img, (bx1, by1, bx2, by2), color, label)
            # cam label
            txt = f"{cam}  ep{ep_idx}"
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
    parser.add_argument("--n", type=int, default=10, help="Number of episodes to render")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Explicit episode list (overrides --n/--seed)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {FINAL_PARQUET}")
    det = pd.read_parquet(FINAL_PARQUET)
    print(f"  {len(det)} rows, {det['episode_index'].nunique()} episodes")

    all_eps = sorted(det["episode_index"].unique().tolist())
    if args.episodes is not None:
        picks = list(args.episodes)
    else:
        rng = random.Random(args.seed)
        picks = sorted(rng.sample(all_eps, args.n))
    print(f"  episodes to render: {picks}")

    _, episodes = load_dataset_meta(DATASET)

    for ep in picks:
        ep_idx = int(ep)
        det_ep = det[det["episode_index"] == ep_idx]
        if len(det_ep) == 0:
            print(f"[skip] ep{ep_idx}: no rows in parquet")
            continue
        render_episode(ep_idx, det_ep, episodes, OUT_DIR)

    print(f"\n[done] {len(picks)} videos at: {OUT_DIR}")


if __name__ == "__main__":
    main()
