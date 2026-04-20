"""Render 3 images from each rejected gallery, showing BOTH teacher bboxes
(duck + cup) so we can judge whether the "other class" on those frames is
usable for training when one class is filter-rejected.

Input gallery images (reference only):
  data_analysis/gallery/rejected_gripper_cup_jaw/
  data_analysis/gallery/rejected_phase_A_no_blue/

Output:
  data_analysis/gallery/rejected_both_obj_review/
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vbti.logic.detection.process_dataset import VideoReader, get_video_path, load_dataset_meta

DATASET = Path("/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim")
CLEAN_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_clean.parquet")
GALLERY_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/data_analysis/gallery"
)
OUT = GALLERY_ROOT / "rejected_both_obj_review"
OUT.mkdir(parents=True, exist_ok=True)

CAMERAS = ["left", "right", "top", "gripper"]
NATIVE_W, NATIVE_H = 640, 480

# Picks: (gallery_name, ep, fr)
PICKS = [
    ("cup_jaw", 1, 162),
    ("cup_jaw", 2, 201),
    ("cup_jaw", 9, 0),
    ("phase_A_no_blue", 0, 405),
    ("phase_A_no_blue", 4, 314),
    ("phase_A_no_blue", 56, 615),
]


def decode_frame(ds_path: Path, cam: str, ep_idx: int, frame_idx: int, episodes: pd.DataFrame) -> np.ndarray:
    cam_key = f"observation.images.{cam}"
    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col = f"videos/{cam_key}/file_index"
    from_ts_col = f"videos/{cam_key}/from_timestamp"
    row = episodes[episodes["episode_index"] == ep_idx].iloc[0]
    chunk = int(row[chunk_col])
    file_idx = int(row[file_col])
    from_ts = float(row[from_ts_col])
    video_path = get_video_path(ds_path, cam_key, chunk, file_idx)
    start_frame = round(from_ts * 30)
    with VideoReader(video_path, NATIVE_W, NATIVE_H) as reader:
        reader.skip(start_frame + frame_idx)
        fr = reader.read_one()
    if fr is None:
        raise RuntimeError(f"frame read failed: ep{ep_idx} fr{frame_idx} cam={cam}")
    return fr


def draw_bbox(img: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              color: tuple[int, int, int], label: str) -> None:
    px1 = int(x1 * NATIVE_W); py1 = int(y1 * NATIVE_H)
    px2 = int(x2 * NATIVE_W); py2 = int(y2 * NATIVE_H)
    cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
    ty = max(16, py1 - 4)
    cv2.putText(img, label, (px1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (px1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    _, episodes = load_dataset_meta(DATASET)
    clean = pd.read_parquet(CLEAN_PARQUET)
    det = pd.read_parquet(DATASET / "detection_results.parquet")
    merged = clean.merge(
        det[[c for c in det.columns if c not in clean.columns or c in ("episode_index", "frame_index")]],
        on=["episode_index", "frame_index"], how="left",
    )

    for gallery, ep, fr in PICKS:
        cam = "gripper"
        try:
            img = decode_frame(DATASET, cam, ep, fr, episodes)
        except Exception as e:
            print(f"[skip] ep{ep} fr{fr}: {e}")
            continue
        row = merged[(merged["episode_index"] == ep) & (merged["frame_index"] == fr)]
        if len(row) == 0:
            print(f"[skip] ep{ep} fr{fr}: no label row")
            continue
        r = row.iloc[0]

        # Pull both obj bboxes + trust + conf
        info_lines = [f"ep{ep:03d} fr{fr:05d}  gallery: {gallery}"]
        for obj, color in [("duck", (0, 255, 255)), ("cup", (255, 80, 200))]:
            x1 = r.get(f"{cam}_{obj}_x1", np.nan)
            y1 = r.get(f"{cam}_{obj}_y1", np.nan)
            x2 = r.get(f"{cam}_{obj}_x2", np.nan)
            y2 = r.get(f"{cam}_{obj}_y2", np.nan)
            conf = r.get(f"{cam}_{obj}_conf", np.nan)
            trust = r.get(f"{cam}_{obj}_trust", np.nan)
            trust_i = int(trust) if pd.notna(trust) else -1
            conf_f = float(conf) if pd.notna(conf) else float("nan")
            info_lines.append(f"  {obj}: conf={conf_f:.3f} trust={trust_i}")
            if all(pd.notna(v) for v in [x1, y1, x2, y2]):
                label = f"{obj} c={conf_f:.2f} trust={trust_i}"
                draw_bbox(img, float(x1), float(y1), float(x2), float(y2), color, label)

        # Header strip
        header_h = 54
        H, W = img.shape[:2]
        out = np.zeros((H + header_h, W, 3), dtype=np.uint8)
        out[header_h:] = img
        for i, line in enumerate(info_lines[:3]):
            cv2.putText(out, line, (8, 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = OUT / f"{gallery}_ep{ep:03d}_fr{fr:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"[save] {out_path.name}: {info_lines[1:]}")


if __name__ == "__main__":
    main()
