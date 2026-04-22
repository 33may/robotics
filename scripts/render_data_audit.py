"""Render 8 diagnostic frames (2 per camera) from detection_labels_final.parquet
for the distillation data audit. Also renders 2 no_objects frames.

Saves individual PNGs + 2×4 grid to:
  /home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/
  detection/distillation/training/00_data_audit/
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
DATASET     = Path("/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim")
FINAL_PARQ  = Path("/home/may33/.cache/vbti/detection_labels_final.parquet")
NO_OBJ_DS   = Path("/home/may33/.cache/huggingface/lerobot/eternalmay33/distill_no_objects")
NO_OBJ_PARQ = Path("/home/may33/.cache/vbti/detection_labels_final_no_objects.parquet")
OUT_DIR     = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/training/00_data_audit"
)

CAMERAS   = ["left", "right", "top", "gripper"]
NATIVE_W  = 640
NATIVE_H  = 480
FPS       = 30
SEED      = 42

COLOR_DUCK = (0, 255, 60)     # BGR green
COLOR_CUP  = (40, 40, 220)    # BGR red


# ── helpers ────────────────────────────────────────────────────────────

def get_video_path(ds_path: Path, cam: str, chunk: int, file_idx: int) -> Path:
    cam_key = f"observation.images.{cam}"
    return ds_path / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"


def load_episodes(ds_path: Path) -> pd.DataFrame:
    import json, pyarrow.parquet as pq
    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    ep_files = sorted((ds_path / "meta" / "episodes").rglob("*.parquet"))
    frames = [pq.read_table(f).to_pandas() for f in ep_files]
    return pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def decode_frame(ds_path: Path, episodes: pd.DataFrame, cam: str, ep_idx: int, frame_idx: int) -> np.ndarray:
    """Decode a single frame from the video file."""
    cam_key = f"observation.images.{cam}"
    row = episodes[episodes["episode_index"] == ep_idx].iloc[0]
    chunk   = int(row[f"videos/{cam_key}/chunk_index"])
    file_i  = int(row[f"videos/{cam_key}/file_index"])
    from_ts = float(row[f"videos/{cam_key}/from_timestamp"])
    start_f = round(from_ts * FPS)
    abs_f   = start_f + frame_idx

    video_path = get_video_path(ds_path, cam, chunk, file_i)
    import subprocess
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{abs_f})",
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    raw = proc.stdout
    if len(raw) < NATIVE_W * NATIVE_H * 3:
        raise RuntimeError(f"Frame decode failed: got {len(raw)} bytes for ep{ep_idx} fi{frame_idx} cam={cam}")
    arr = np.frombuffer(raw[:NATIVE_W*NATIVE_H*3], dtype=np.uint8).reshape(NATIVE_H, NATIVE_W, 3)
    return arr.copy()


def draw_detection(img: np.ndarray, row: pd.Series, cam: str) -> np.ndarray:
    """Overlay duck + cup bboxes and labels on img (RGB). Returns BGR copy."""
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for obj, color in [("duck", COLOR_DUCK), ("cup", COLOR_CUP)]:
        x1n = row.get(f"{cam}_{obj}_x1", np.nan)
        y1n = row.get(f"{cam}_{obj}_y1", np.nan)
        x2n = row.get(f"{cam}_{obj}_x2", np.nan)
        y2n = row.get(f"{cam}_{obj}_y2", np.nan)
        conf = row.get(f"{cam}_{obj}_conf", np.nan)
        trust = row.get(f"{cam}_{obj}_trust", 1)
        cx_n = row.get(f"{cam}_{obj}_cx", np.nan)
        cy_n = row.get(f"{cam}_{obj}_cy", np.nan)

        if any(pd.isna(v) for v in [x1n, y1n, x2n, y2n]):
            continue

        x1 = int(float(x1n) * NATIVE_W)
        y1 = int(float(y1n) * NATIVE_H)
        x2 = int(float(x2n) * NATIVE_W)
        y2 = int(float(y2n) * NATIVE_H)
        conf_f = float(conf) if pd.notna(conf) else 0.0
        trust_i = int(trust) if pd.notna(trust) else 1

        # bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # center dot
        if pd.notna(cx_n) and pd.notna(cy_n):
            cx = int(float(cx_n) * NATIVE_W)
            cy = int(float(cy_n) * NATIVE_H)
            cv2.circle(out, (cx, cy), 5, color, -1)

        # label
        label = f"{obj} conf={conf_f:.2f} trust={trust_i}"
        ty = max(14, y1 - 6)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def put_header(img: np.ndarray, text: str):
    """Write header text top-left (in-place, BGR img)."""
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


# ── main ───────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ── 1. Load final parquet ──────────────────────────────────────────
    print(f"Loading {FINAL_PARQ} ...")
    det = pd.read_parquet(FINAL_PARQ)
    print(f"  {len(det)} rows, {det['episode_index'].nunique()} episodes")

    episodes = load_episodes(DATASET)

    # ── 2. Pick 2 frames per camera (both duck+cup trust=1) ───────────
    picks: list[dict] = []   # {cam, ep_idx, frame_idx}
    for cam in CAMERAS:
        mask = (det[f"{cam}_duck_trust"] == 1) & (det[f"{cam}_cup_trust"] == 1)
        pool = det[mask].copy()
        # Try to pick from different episodes
        ep_list = pool["episode_index"].unique().tolist()
        rng.shuffle(ep_list)
        chosen_eps = ep_list[:2]
        for ep in chosen_eps:
            ep_pool = pool[pool["episode_index"] == ep]
            sample = ep_pool.sample(1, random_state=rng.randint(0, 99999))
            row = sample.iloc[0]
            picks.append({
                "cam": cam,
                "ep_idx": int(row["episode_index"]),
                "frame_idx": int(row["frame_index"]),
                "det_row": row,
            })
        print(f"  {cam}: picked ep={[p['ep_idx'] for p in picks if p['cam']==cam]} "
              f"frame={[p['frame_idx'] for p in picks if p['cam']==cam]}")

    # ── 3. Decode & render each frame ────────────────────────────────
    rendered: list[np.ndarray] = []   # BGR, 640×480
    cam_counts = {cam: 0 for cam in CAMERAS}
    for p in picks:
        cam      = p["cam"]
        ep_idx   = p["ep_idx"]
        frame_idx = p["frame_idx"]
        det_row  = p["det_row"]

        print(f"  Decoding cam={cam} ep={ep_idx} frame={frame_idx} ...")
        try:
            rgb = decode_frame(DATASET, episodes, cam, ep_idx, frame_idx)
        except Exception as e:
            print(f"  [WARN] decode failed: {e} — skipping")
            rendered.append(np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8))
            cam_counts[cam] += 1
            continue

        bgr = draw_detection(rgb, det_row, cam)
        header = f"cam={cam}  ep={ep_idx}  fr={frame_idx}"
        put_header(bgr, header)

        n = cam_counts[cam]
        fname = OUT_DIR / f"cam_{cam}_frame_{n}.png"
        cv2.imwrite(str(fname), bgr)
        print(f"  -> {fname}")
        rendered.append(bgr)
        cam_counts[cam] += 1

    # ── 4. 2×4 grid (row=cam_slot 0/1, col=camera L/R/T/G) ──────────
    # Layout: 4 cols × 2 rows → each cell = one pick
    # Order: [left_0, right_0, top_0, gripper_0]
    #        [left_1, right_1, top_1, gripper_1]
    assert len(rendered) == 8, f"Expected 8 frames, got {len(rendered)}"
    row0 = np.concatenate([rendered[i] for i in [0, 2, 4, 6]], axis=1)   # first pick each cam
    row1 = np.concatenate([rendered[i] for i in [1, 3, 5, 7]], axis=1)   # second pick each cam
    grid = np.concatenate([row0, row1], axis=0)
    grid_path = OUT_DIR / "data_audit_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"\n[grid] {grid_path}")

    # ── 5. No-objects negatives (2 frames, random cams) ──────────────
    print(f"\nLoading no_objects parquet {NO_OBJ_PARQ} ...")
    no_obj_det = pd.read_parquet(NO_OBJ_PARQ)
    no_obj_eps = load_episodes(NO_OBJ_DS)
    print(f"  {len(no_obj_det)} rows, {no_obj_det['episode_index'].nunique()} episodes")

    no_obj_cams = rng.choices(CAMERAS, k=2)
    for n, cam in enumerate(no_obj_cams):
        # pick a random row from no_obj
        sample_row = no_obj_det.sample(1, random_state=SEED + n).iloc[0]
        ep_idx    = int(sample_row["episode_index"])
        frame_idx = int(sample_row["frame_index"])
        print(f"  no_obj {n}: cam={cam} ep={ep_idx} frame={frame_idx}")
        try:
            rgb = decode_frame(NO_OBJ_DS, no_obj_eps, cam, ep_idx, frame_idx)
        except Exception as e:
            print(f"  [WARN] decode failed: {e}")
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        put_header(bgr, f"NO_OBJ cam={cam} ep={ep_idx} fr={frame_idx}")
        fname = OUT_DIR / f"no_obj_cam_{cam}_{n}.png"
        cv2.imwrite(str(fname), bgr)
        print(f"  -> {fname}")

    # ── 6. Print summary stats for findings ──────────────────────────
    print("\n=== Label stats for picked frames ===")
    for p in picks:
        r = p["det_row"]
        cam = p["cam"]
        for obj in ["duck", "cup"]:
            conf  = r.get(f"{cam}_{obj}_conf", float("nan"))
            trust = r.get(f"{cam}_{obj}_trust", float("nan"))
            x1    = r.get(f"{cam}_{obj}_x1", float("nan"))
            y1    = r.get(f"{cam}_{obj}_y1", float("nan"))
            x2    = r.get(f"{cam}_{obj}_x2", float("nan"))
            y2    = r.get(f"{cam}_{obj}_y2", float("nan"))
            def fmt(v):
                try: return f"{float(v):.3f}"
                except: return str(v)
            print(f"  cam={cam} ep={p['ep_idx']} fr={p['frame_idx']} {obj}: "
                  f"conf={fmt(conf)} trust={fmt(trust)} "
                  f"bbox=[{fmt(x1)}, {fmt(y1)}, {fmt(x2)}, {fmt(y2)}]")

    print(f"\n[done] All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
