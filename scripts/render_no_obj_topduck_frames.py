"""Render top-cam frames of ep4 between sec 26-33 (frames 780-990) to inspect
whether the gradual duck-bbox motion is (a) interpolation artifact between
sparse DINO samples, or (b) DINO actually detecting something moving.

Also render a few intermediate (non-stride) frames to expose the interpolation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vbti.logic.detection.process_dataset import VideoReader, get_video_path, load_dataset_meta

DS_PATH = Path("/home/may33/.cache/huggingface/lerobot/eternalmay33/distill_no_objects")
CLEAN_PARQUET = Path(DS_PATH / "detection_results_clean.parquet")
RAW_PARQUET = Path(DS_PATH / "detection_results.parquet")  # restored original (sparse)
TRUST_PARQUET = Path("/home/may33/.cache/vbti/detection_labels_clean_no_objects.parquet")
OUT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/no_objects/ep4_topduck_26s_33s"
)

EP = 4
CAM = "top"
OBJ = "duck"
FRAMES_TO_RENDER = list(range(780, 991, 15))  # every 0.5 sec
NATIVE_W, NATIVE_H = 640, 480
FPS = 30


def render():
    OUT.mkdir(parents=True, exist_ok=True)

    _, episodes = load_dataset_meta(DS_PATH)
    clean = pd.read_parquet(CLEAN_PARQUET)
    raw = pd.read_parquet(RAW_PARQUET)
    trust = pd.read_parquet(TRUST_PARQUET) if TRUST_PARQUET.exists() else None

    # subset ep4
    clean_ep = clean[clean["episode_index"] == EP].set_index("frame_index")
    raw_ep = raw[raw["episode_index"] == EP].set_index("frame_index")
    trust_ep = trust[trust["episode_index"] == EP].set_index("frame_index") if trust is not None else None

    # Decode just the frames we want
    cam_key = f"observation.images.{CAM}"
    row = episodes[episodes["episode_index"] == EP].iloc[0]
    chunk = int(row[f"videos/{cam_key}/chunk_index"])
    file_idx = int(row[f"videos/{cam_key}/file_index"])
    from_ts = float(row[f"videos/{cam_key}/from_timestamp"])
    video_path = get_video_path(DS_PATH, cam_key, chunk, file_idx)
    start_frame = round(from_ts * FPS)

    frames = {}
    with VideoReader(video_path, NATIVE_W, NATIVE_H) as r:
        r.skip(start_frame)
        pos = 0
        targets = set(FRAMES_TO_RENDER)
        max_fr = max(FRAMES_TO_RENDER)
        while pos <= max_fr:
            fr = r.read_one()
            if fr is None:
                break
            if pos in targets:
                frames[pos] = fr
            pos += 1

    for fr_idx in FRAMES_TO_RENDER:
        if fr_idx not in frames:
            continue
        img = frames[fr_idx].copy()

        # Get interp bbox (dense)
        if fr_idx in clean_ep.index:
            c = clean_ep.loc[fr_idx]
            x1 = c.get(f"{CAM}_{OBJ}_x1", np.nan)
            y1 = c.get(f"{CAM}_{OBJ}_y1", np.nan)
            x2 = c.get(f"{CAM}_{OBJ}_x2", np.nan)
            y2 = c.get(f"{CAM}_{OBJ}_y2", np.nan)
            conf = c.get(f"{CAM}_{OBJ}_conf", np.nan)
        else:
            x1 = y1 = x2 = y2 = conf = np.nan

        # Is this a stride-sampled frame (i.e., raw DINO has bbox) or interpolated?
        is_stride = False
        if fr_idx in raw_ep.index:
            rawrow = raw_ep.loc[fr_idx]
            rx1 = rawrow.get(f"{CAM}_{OBJ}_x1", np.nan)
            is_stride = bool(pd.notna(rx1))

        # Trust
        trust_val = 1
        if trust_ep is not None and fr_idx in trust_ep.index:
            t = trust_ep.loc[fr_idx].get(f"{CAM}_{OBJ}_trust", 1)
            try:
                trust_val = int(t) if pd.notna(t) else 1
            except (TypeError, ValueError):
                trust_val = 1

        # Apply v9 top_duck y2 rule
        y2_reject = bool(pd.notna(y2) and float(y2) > 0.85)
        effective_trust = 0 if y2_reject else trust_val

        # Draw
        if all(pd.notna(v) for v in [x1, y1, x2, y2]):
            bx1 = int(float(x1) * NATIVE_W); by1 = int(float(y1) * NATIVE_H)
            bx2 = int(float(x2) * NATIVE_W); by2 = int(float(y2) * NATIVE_H)
            color = (0, 255, 120) if effective_trust == 1 else (60, 60, 255)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            cf = float(conf) if pd.notna(conf) else 0.0
            prefix = "MASKED " if effective_trust == 0 else ""
            label = f"{prefix}duck {cf:.2f}"
            cv2.putText(img, label, (bx1, max(16, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (bx1, max(16, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Header
        stride_tag = "STRIDE (real DINO)" if is_stride else "INTERP"
        header = f"ep{EP} fr{fr_idx}  t={fr_idx/FPS:.2f}s  [{stride_tag}]  y2={float(y2) if pd.notna(y2) else 0:.2f}  trust={effective_trust}"
        out = np.zeros((img.shape[0] + 32, img.shape[1], 3), dtype=np.uint8)
        out[32:] = img
        cv2.putText(out, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = OUT / f"fr{fr_idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"[save] {out_path.name}  stride={is_stride}  trust={effective_trust}  y2={float(y2) if pd.notna(y2) else -1:.2f}")


if __name__ == "__main__":
    render()
