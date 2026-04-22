"""Render distillation comparison artifacts for a trained run.

Reusable for any training run (baseline_raw, v10_clean, v11_*, ...). Loads
per-camera best.pt checkpoints from training/<run>/<cam>/best.pt, renders:

  1. comparison_student_vs_dino.png   — 3×4 grid (3 val episodes × 4 cams)
  2. tracking_ep<EP>.mp4               — full-episode 2×2 video (H.264)
  3. training_compare_<run>_vs_<compare>.png  — optional baseline comparison

Usage:
    python -u scripts/distill_compare.py --run v10_clean
    python -u scripts/distill_compare.py --run v11_foo --episode 215
    python -u scripts/distill_compare.py --run v10_clean --compare-with baseline_raw
    python -u scripts/distill_compare.py --run v10_clean --skip-video    # grid only
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vbti.logic.detection.distill_model import DistilledDetector
from vbti.logic.detection.process_dataset import (
    VideoReader,
    get_video_path,
    load_dataset_meta,
)

# ---- paths ----
DATASET_PATH = Path(
    "/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim"
)
TRAINING_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/training"
)

CAMERAS = ["left", "right", "top", "gripper"]
NATIVE_W, NATIVE_H = 640, 480
IMG_SIZE = 224
FPS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- colors ----
# Teacher = cool blues/purples (dashed). Student = warm yellows/greens (solid).
# Each distinct RGB so student and teacher never blend even if they overlap.
COLOR_T_DUCK_RGB = (0, 200, 255)    # cyan
COLOR_T_CUP_RGB  = (255, 80, 200)   # pink-magenta
COLOR_S_DUCK_RGB = (80, 255, 80)    # lime green
COLOR_S_CUP_RGB  = (255, 170, 0)    # amber orange

# Matplotlib (0-1 floats) equivalents
def _to_mpl(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

COLOR_T_DUCK_MPL = _to_mpl(COLOR_T_DUCK_RGB)
COLOR_T_CUP_MPL  = _to_mpl(COLOR_T_CUP_RGB)
COLOR_S_DUCK_MPL = _to_mpl(COLOR_S_DUCK_RGB)
COLOR_S_CUP_MPL  = _to_mpl(COLOR_S_CUP_RGB)

# ---- smoothing (video only) ----
# EMA to kill per-frame jitter. alpha = weight of new sample.
EMA_ALPHA_CX = 0.35
EMA_ALPHA_CY = 0.35
EMA_ALPHA_CONF = 0.5
DEFAULT_BBOX_SIZE = 60
CONF_DRAW_THRESHOLD = 0.15  # skip drawing student prediction below this confidence


# =============================================================================
# model loading + inference
# =============================================================================
def load_students(run: str) -> dict[str, DistilledDetector]:
    run_root = TRAINING_ROOT / run
    students: dict[str, DistilledDetector] = {}
    for cam in CAMERAS:
        ckpt_path = run_root / cam / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        model = DistilledDetector(pretrained=False)
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval().to(DEVICE)
        students[cam] = model
    return students


@torch.inference_mode()
def predict_student(model: DistilledDetector, frame_rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).to(DEVICE).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    out = model(x)[0].cpu().numpy()
    return out


@torch.inference_mode()
def predict_student_batch(
    model: DistilledDetector, frames_rgb: list[np.ndarray]
) -> np.ndarray:
    if not frames_rgb:
        return np.zeros((0, 6), dtype=np.float32)
    xs = [cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) for f in frames_rgb]
    arr = np.stack(xs).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).to(DEVICE).permute(0, 3, 1, 2)
    return model(t).cpu().numpy()


def smooth_predictions(preds: np.ndarray) -> np.ndarray:
    """EMA smoothing on (T, 6) student outputs to reduce per-frame jitter.

    Applies different alphas to coords vs conf. Confidence uses a higher alpha
    because it encodes semantic state changes (obj appears/disappears) that
    should not be blurred as much as positional jitter.
    """
    out = preds.copy()
    for t in range(1, len(out)):
        for idx, a in (
            (0, EMA_ALPHA_CX), (1, EMA_ALPHA_CY), (2, EMA_ALPHA_CONF),
            (3, EMA_ALPHA_CX), (4, EMA_ALPHA_CY), (5, EMA_ALPHA_CONF),
        ):
            out[t, idx] = a * preds[t, idx] + (1 - a) * out[t - 1, idx]
    return out


# =============================================================================
# dataset helpers
# =============================================================================
def find_episode_video(episodes: pd.DataFrame, ep_idx: int, cam: str) -> tuple[Path, int, int]:
    cam_key = f"observation.images.{cam}"
    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col = f"videos/{cam_key}/file_index"
    from_ts_col = f"videos/{cam_key}/from_timestamp"
    row = episodes[episodes["episode_index"] == ep_idx].iloc[0]
    chunk = int(row[chunk_col])
    file_idx = int(row[file_col])
    from_ts = float(row[from_ts_col])
    length = int(row["length"])
    video_path = get_video_path(DATASET_PATH, cam_key, chunk, file_idx)
    start_frame = round(from_ts * FPS)
    return video_path, start_frame, length


def read_episode_frames(ep_idx: int, cam: str, episodes: pd.DataFrame) -> np.ndarray:
    video_path, start_frame, length = find_episode_video(episodes, ep_idx, cam)
    frames = np.empty((length, NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    with VideoReader(video_path, NATIVE_W, NATIVE_H) as reader:
        reader.skip(start_frame)
        for i in range(length):
            fr = reader.read_one()
            if fr is None:
                frames = frames[:i]
                break
            frames[i] = fr
    return frames


def read_single_frame(ep_idx: int, frame_idx: int, cam: str, episodes: pd.DataFrame) -> np.ndarray:
    video_path, start_frame, _ = find_episode_video(episodes, ep_idx, cam)
    with VideoReader(video_path, NATIVE_W, NATIVE_H) as reader:
        reader.skip(start_frame + frame_idx)
        fr = reader.read_one()
    if fr is None:
        raise RuntimeError(f"could not read ep={ep_idx} frame={frame_idx} cam={cam}")
    return fr


# =============================================================================
# teacher bbox helpers
# =============================================================================
def _extract_teacher_bbox(
    teacher_row: pd.Series | None, cam: str, obj: str
) -> tuple[float, float, float, float] | None:
    """Return teacher bbox in pixel coords (x1, y1, x2, y2) or None."""
    if teacher_row is None:
        return None
    x1 = teacher_row.get(f"{cam}_{obj}_x1", np.nan)
    y1 = teacher_row.get(f"{cam}_{obj}_y1", np.nan)
    x2 = teacher_row.get(f"{cam}_{obj}_x2", np.nan)
    y2 = teacher_row.get(f"{cam}_{obj}_y2", np.nan)
    if any(v is None for v in (x1, y1, x2, y2)):
        return None
    try:
        x1f = float(x1); y1f = float(y1); x2f = float(x2); y2f = float(y2)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x1f, y1f, x2f, y2f])):
        return None
    return (x1f * NATIVE_W, y1f * NATIVE_H, x2f * NATIVE_W, y2f * NATIVE_H)


def _teacher_conf(teacher_row: pd.Series | None, cam: str, obj: str) -> float | None:
    if teacher_row is None:
        return None
    c = teacher_row.get(f"{cam}_{obj}_conf", np.nan)
    try:
        cf = float(c)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return cf if np.isfinite(cf) else None


# =============================================================================
# matplotlib rendering (static grid)
# =============================================================================
def draw_subplot(
    ax, frame_rgb: np.ndarray,
    teacher_row: pd.Series | None,
    student_out: np.ndarray,
    cam: str,
) -> None:
    ax.imshow(frame_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    title_parts = [cam]

    for obj, tcolor, scolor, s_idx in [
        ("duck", COLOR_T_DUCK_MPL, COLOR_S_DUCK_MPL, (0, 1, 2)),
        ("cup",  COLOR_T_CUP_MPL,  COLOR_S_CUP_MPL,  (3, 4, 5)),
    ]:
        t_bbox = _extract_teacher_bbox(teacher_row, cam, obj)
        t_cx = t_cy = None
        t_w = t_h = None

        if t_bbox is not None:
            x1p, y1p, x2p, y2p = t_bbox
            t_w = x2p - x1p
            t_h = y2p - y1p
            t_cx = 0.5 * (x1p + x2p)
            t_cy = 0.5 * (y1p + y2p)
            ax.add_patch(mpatches.Rectangle(
                (x1p, y1p), t_w, t_h,
                linewidth=1.5, edgecolor=tcolor, facecolor="none", linestyle="--",
            ))
            tconf = _teacher_conf(teacher_row, cam, obj)
            tl = f"T{obj[0]}" + (f" {tconf:.2f}" if tconf is not None else "")
            ax.text(
                x1p, max(0, y1p - 2), tl, color=tcolor, fontsize=6,
                verticalalignment="bottom",
                bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"),
            )

        s_cx = float(student_out[s_idx[0]]) * NATIVE_W
        s_cy = float(student_out[s_idx[1]]) * NATIVE_H
        s_conf = float(student_out[s_idx[2]])

        s_w = t_w if t_w is not None else float(DEFAULT_BBOX_SIZE)
        s_h = t_h if t_h is not None else float(DEFAULT_BBOX_SIZE)
        sx1 = s_cx - s_w / 2
        sy1 = s_cy - s_h / 2
        ax.add_patch(mpatches.Rectangle(
            (sx1, sy1), s_w, s_h,
            linewidth=1.5, edgecolor=scolor, facecolor="none", linestyle="-",
        ))
        sl = f"S{obj[0]} {s_conf:.2f}"
        ax.text(
            sx1, min(NATIVE_H - 1, sy1 + s_h + 2), sl, color=scolor, fontsize=6,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"),
        )

        if t_cx is not None and t_cy is not None:
            err = float(np.hypot(s_cx - t_cx, s_cy - t_cy))
            title_parts.append(f"{obj[0]}={err:.0f}px")
    ax.set_title(" ".join(title_parts), fontsize=8)


def _pick_frame_with_all_teachers(teacher: pd.DataFrame, ep_idx: int, ep_len: int) -> int:
    """Find a frame within episode ep_idx where the teacher has valid bboxes
    for BOTH duck and cup on ALL 4 cameras. Scan outward from the episode midpoint.
    Falls back to the midpoint if no fully-covered frame is found.
    """
    required_cols = [
        f"{cam}_{obj}_x1" for cam in CAMERAS for obj in ("duck", "cup")
    ]
    ep_t = teacher[teacher["episode_index"] == ep_idx]
    if len(ep_t) == 0:
        return ep_len // 2
    # mask: which frames have all required x1 cols non-null
    valid_mask = ep_t[required_cols].notna().all(axis=1)  # type: ignore[union-attr]
    valid_frames = ep_t.loc[valid_mask, "frame_index"].to_numpy()
    if len(valid_frames) == 0:
        print(f"  [grid] ep{ep_idx}: NO frame with full teacher coverage — using midpoint")
        return ep_len // 2
    # pick the valid frame closest to the episode midpoint
    mid = ep_len // 2
    best = int(valid_frames[np.argmin(np.abs(valid_frames - mid))])
    print(f"  [grid] ep{ep_idx}: picked fr{best} (closest to midpoint {mid}, "
          f"{len(valid_frames)} valid frames available)")
    return best


def build_comparison_grid(
    students: dict[str, DistilledDetector],
    teacher: pd.DataFrame,
    episodes: pd.DataFrame,
    run: str,
    grid_episodes: list[int],
) -> Path:
    rows = []
    for ep_idx in grid_episodes:
        ep_len = int(episodes[episodes["episode_index"] == ep_idx].iloc[0]["length"])
        frame_idx = _pick_frame_with_all_teachers(teacher, ep_idx, ep_len)
        rows.append((ep_idx, frame_idx))

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.3 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    for r, (ep_idx, frame_idx) in enumerate(rows):
        sub = teacher[(teacher["episode_index"] == ep_idx) & (teacher["frame_index"] == frame_idx)]
        trow = sub.iloc[0] if len(sub) > 0 else None
        for c, cam in enumerate(CAMERAS):
            frame_rgb = read_single_frame(ep_idx, frame_idx, cam, episodes)
            out = predict_student(students[cam], frame_rgb)
            draw_subplot(axes[r, c], frame_rgb, trow, out, cam)
        axes[r, 0].set_ylabel(f"ep{ep_idx} fr{frame_idx}", fontsize=10)

    fig.suptitle(
        f"{run} student vs G-DINO teacher\n"
        f"teacher (dashed): cyan=duck, pink=cup  |  student (solid): lime=duck, amber=cup",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = TRAINING_ROOT / run / "comparison_student_vs_dino.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[grid] saved {out_path}")
    return out_path


# =============================================================================
# opencv rendering (video)
# =============================================================================
def draw_dashed_rect(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    x1, y1 = p1
    x2, y2 = p2

    def dashed_line(a: tuple[int, int], b: tuple[int, int]) -> None:
        total = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if total == 0:
            return
        dx = (b[0] - a[0]) / total
        dy = (b[1] - a[1]) / total
        s = 0.0
        while s < total:
            e = min(s + dash_len, total)
            p_s = (int(a[0] + dx * s), int(a[1] + dy * s))
            p_e = (int(a[0] + dx * e), int(a[1] + dy * e))
            cv2.line(img, p_s, p_e, color, thickness)
            s = e + gap_len

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def _draw_crosshair(img: np.ndarray, cx: int, cy: int, color: tuple[int, int, int], r: int = 8, thickness: int = 2) -> None:
    """Draw a crosshair (+) at (cx, cy) — compact point marker for center predictions."""
    cv2.line(img, (cx - r, cy), (cx + r, cy), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r), (cx, cy + r), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, (cx - r, cy), (cx + r, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r), (cx, cy + r), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)


def draw_overlay_on_frame(
    frame_rgb: np.ndarray,
    teacher_row: pd.Series | None,
    student_out: np.ndarray,
    cam: str,
    point_only: bool = False,
    show_teacher: bool = True,
) -> np.ndarray:
    img = frame_rgb.copy()
    errs: dict[str, float | None] = {"duck": None, "cup": None}

    for obj, tcolor, scolor, s_idx in [
        ("duck", COLOR_T_DUCK_RGB, COLOR_S_DUCK_RGB, (0, 1, 2)),
        ("cup",  COLOR_T_CUP_RGB,  COLOR_S_CUP_RGB,  (3, 4, 5)),
    ]:
        t_bbox = _extract_teacher_bbox(teacher_row, cam, obj) if show_teacher else None
        t_cx = t_cy = None
        t_w = t_h = None

        if t_bbox is not None:
            x1p, y1p, x2p, y2p = t_bbox
            x1i, y1i = int(x1p), int(y1p)
            x2i, y2i = int(x2p), int(y2p)
            t_w = float(x2i - x1i)
            t_h = float(y2i - y1i)
            t_cx = 0.5 * (x1i + x2i)
            t_cy = 0.5 * (y1i + y2i)
            if point_only:
                _draw_crosshair(img, int(t_cx), int(t_cy), tcolor)
            else:
                draw_dashed_rect(img, (x1i, y1i), (x2i, y2i), tcolor, thickness=2)
            tconf = _teacher_conf(teacher_row, cam, obj)
            tl = f"T{obj[0]}" + (f" {tconf:.2f}" if tconf is not None else "")
            ty = max(12, y1i - 4) if not point_only else max(12, int(t_cy) - 10)
            tx = x1i if not point_only else int(t_cx) + 6
            cv2.putText(img, tl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, tl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tcolor, 1, cv2.LINE_AA)

        s_cx = float(student_out[s_idx[0]]) * NATIVE_W
        s_cy = float(student_out[s_idx[1]]) * NATIVE_H
        s_conf = float(student_out[s_idx[2]])

        if s_conf >= CONF_DRAW_THRESHOLD:
            if point_only:
                _draw_crosshair(img, int(s_cx), int(s_cy), scolor)
                sl = f"S{obj[0]} {s_conf:.2f}"
                cv2.putText(img, sl, (int(s_cx) + 6, int(s_cy) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, sl, (int(s_cx) + 6, int(s_cy) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, scolor, 1, cv2.LINE_AA)
            else:
                s_w = t_w if t_w is not None else float(DEFAULT_BBOX_SIZE)
                s_h = t_h if t_h is not None else float(DEFAULT_BBOX_SIZE)
                sx1 = int(max(0, min(NATIVE_W - 1, s_cx - s_w / 2)))
                sy1 = int(max(0, min(NATIVE_H - 1, s_cy - s_h / 2)))
                sx2 = int(max(0, min(NATIVE_W - 1, s_cx + s_w / 2)))
                sy2 = int(max(0, min(NATIVE_H - 1, s_cy + s_h / 2)))
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), scolor, 2)
                sl = f"S{obj[0]} {s_conf:.2f}"
                sy_text = min(NATIVE_H - 3, sy2 + 14)
                cv2.putText(img, sl, (sx1, sy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, sl, (sx1, sy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, scolor, 1, cv2.LINE_AA)

        if t_cx is not None and t_cy is not None and s_conf >= CONF_DRAW_THRESHOLD:
            errs[obj] = float(np.hypot(s_cx - t_cx, s_cy - t_cy))

    err_d = errs["duck"]
    err_c = errs["cup"]
    err_d_s = f"{err_d:.0f}" if err_d is not None else "-"
    err_c_s = f"{err_c:.0f}" if err_c is not None else "-"
    label = f"{cam}  d={err_d_s}  c={err_c_s}px" if show_teacher else cam
    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_tracking_video(
    ep_idx: int,
    students: dict[str, DistilledDetector],
    teacher: pd.DataFrame,
    episodes: pd.DataFrame,
    run: str,
    point_only: bool = False,
    show_teacher: bool = True,
) -> Path:
    print(f"[video] decoding ep{ep_idx} frames (4 cams)...")
    cam_frames: dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        cam_frames[cam] = read_episode_frames(ep_idx, cam, episodes)
    T = min(len(cam_frames[c]) for c in CAMERAS)
    print(f"[video] {T} frames per cam")

    t_ep = teacher[teacher["episode_index"] == ep_idx].set_index("frame_index")

    print("[video] running student inference + smoothing...")
    cam_preds: dict[str, np.ndarray] = {}
    B = 128
    for cam in CAMERAS:
        frames = cam_frames[cam][:T]
        outs = np.zeros((T, 6), dtype=np.float32)
        for start in range(0, T, B):
            batch = [frames[i] for i in range(start, min(start + B, T))]
            outs[start:start + len(batch)] = predict_student_batch(students[cam], batch)
        cam_preds[cam] = smooth_predictions(outs)

    grid_h = 2 * NATIVE_H
    grid_w = 2 * NATIVE_W
    suffix = ""
    if point_only:
        suffix += "_point"
    if not show_teacher:
        suffix += "_noteacher"
    out_dir = TRAINING_ROOT / run
    tmp_path = out_dir / f"tracking_ep{ep_idx}{suffix}.tmp.mp4"
    final_path = out_dir / f"tracking_ep{ep_idx}{suffix}.mp4"

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter at {tmp_path}")

    layout = [("left", 0, 0), ("right", 0, 1), ("top", 1, 0), ("gripper", 1, 1)]
    print("[video] compositing...")
    for t in range(T):
        trow = t_ep.loc[t] if t in t_ep.index else None
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for cam, rr, cc in layout:
            frame = cam_frames[cam][t]
            out = cam_preds[cam][t]
            overlay = draw_overlay_on_frame(frame, trow, out, cam, point_only=point_only, show_teacher=show_teacher)
            canvas[rr * NATIVE_H:(rr + 1) * NATIVE_H, cc * NATIVE_W:(cc + 1) * NATIVE_W] = overlay
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[video] tmp mp4v at {tmp_path}")

    print("[video] ffmpeg -> libx264 yuv420p (Obsidian-compatible)...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "22", "-preset", "fast",
        str(final_path),
    ], check=True)
    print(f"[video] saved {final_path}")
    try:
        tmp_path.unlink()
    except Exception:
        pass
    return final_path


# =============================================================================
# training curves comparison (optional, vs another run)
# =============================================================================
def plot_training_compare(run: str, compare_with: str) -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=False)
    metrics = [
        ("duck_err_median_px", "val duck median px error"),
        ("cup_err_median_px", "val cup median px error"),
        ("duck_false_positive_rate", "val duck false-positive rate"),
    ]
    run_root = TRAINING_ROOT / run
    cmp_root = TRAINING_ROOT / compare_with

    for r, cam in enumerate(CAMERAS):
        a_df = pd.read_csv(cmp_root / cam / "metrics.csv")
        b_df = pd.read_csv(run_root / cam / "metrics.csv")
        for c, (col, title) in enumerate(metrics):
            ax = axes[r, c]
            ax.plot(a_df["epoch"], a_df[col], color="C0", marker="o", markersize=3, label=compare_with)
            ax.plot(b_df["epoch"], b_df[col], color="C1", marker="o", markersize=3, label=run)
            ax.set_title(f"{cam} — {title}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Training curves: {compare_with} vs {run}", fontsize=13, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = run_root / f"training_compare_{compare_with}_vs_{run}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[curves] saved {out_path}")
    return out_path


# =============================================================================
# driver
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training run name (e.g. v10_clean)")
    parser.add_argument("--episode", type=int, default=210, help="episode for tracking video (default 210)")
    parser.add_argument(
        "--grid-episodes", type=int, nargs="+", default=[205, 220, 235],
        help="val episodes for the 3-row grid plot",
    )
    parser.add_argument("--compare-with", type=str, default=None, help="other run to compare curves against")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--point-only", action="store_true",
                        help="draw center crosshair instead of bounding box")
    parser.add_argument("--no-teacher", action="store_true",
                        help="omit teacher annotations entirely")
    args = parser.parse_args()

    run_root = TRAINING_ROOT / args.run
    if not run_root.exists():
        raise FileNotFoundError(f"run dir not found: {run_root}")

    print(f"Device: {DEVICE}")
    print(f"Run: {args.run}")

    print("Loading dataset meta + teacher parquet...")
    _, episodes = load_dataset_meta(DATASET_PATH)
    teacher = pd.read_parquet(DATASET_PATH / "detection_results.parquet")

    print(f"Loading {args.run} students (4 cams)...")
    students = load_students(args.run)

    if not args.skip_grid:
        build_comparison_grid(students, teacher, episodes, args.run, args.grid_episodes)
    if not args.skip_video:
        build_tracking_video(
            args.episode, students, teacher, episodes, args.run,
            point_only=args.point_only,
            show_teacher=not args.no_teacher,
        )
    if args.compare_with:
        plot_training_compare(args.run, args.compare_with)

    print("done.")


if __name__ == "__main__":
    main()
