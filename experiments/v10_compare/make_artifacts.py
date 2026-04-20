"""Generate v10_clean comparison artifacts: grid plot, tracking video, training curves.

Outputs written under:
  /home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/detection/distillation/training/v10_clean/
"""
from __future__ import annotations

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
V10_ROOT = TRAINING_ROOT / "v10_clean"
BASELINE_ROOT = TRAINING_ROOT / "baseline_raw"

CAMERAS = ["left", "right", "top", "gripper"]
NATIVE_W, NATIVE_H = 640, 480
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- model loading ----
def load_student(run_root: Path, cam: str) -> DistilledDetector:
    ckpt_path = run_root / cam / "best.pt"
    model = DistilledDetector(pretrained=False)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval().to(DEVICE)
    return model


# ---- inference ----
@torch.inference_mode()
def predict_student(model: DistilledDetector, frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
    """Run student on a single native-resolution RGB frame.

    Returns (6,) numpy array: [duck_cx, duck_cy, duck_conf, cup_cx, cup_cy, cup_conf]
    with cx/cy still normalized in [0,1].
    """
    small = cv2.resize(frame_bgr_or_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
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
    xs = []
    for f in frames_rgb:
        small = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        xs.append(small)
    arr = np.stack(xs).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).to(DEVICE).permute(0, 3, 1, 2)
    out = model(t).cpu().numpy()
    return out


# ---- dataset helpers ----
def find_episode_video(episodes: pd.DataFrame, ep_idx: int, cam: str) -> tuple[Path, int, int]:
    """Return (video_path, start_frame, length) for ep_idx/cam.

    start_frame is the frame index in the video file (which may contain multiple episodes).
    """
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
    start_frame = round(from_ts * 30)  # fps=30
    return video_path, start_frame, length


def read_episode_frames(ep_idx: int, cam: str, episodes: pd.DataFrame) -> np.ndarray:
    """Decode ALL frames of episode ep_idx for camera cam. Returns (T, H, W, 3) uint8 RGB."""
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
        raise RuntimeError(f"Could not read ep={ep_idx} frame={frame_idx} cam={cam}")
    return fr


# ---- artifact 3: training curves compare ----
def plot_training_compare():
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=False)
    metrics_to_plot = [
        ("duck_err_median_px", "val duck median px error"),
        ("cup_err_median_px", "val cup median px error"),
        ("duck_false_positive_rate", "val duck false-positive rate"),
    ]
    for r, cam in enumerate(CAMERAS):
        base_df = pd.read_csv(BASELINE_ROOT / cam / "metrics.csv")
        v10_df = pd.read_csv(V10_ROOT / cam / "metrics.csv")
        for c, (col, title) in enumerate(metrics_to_plot):
            ax = axes[r, c]
            ax.plot(base_df["epoch"], base_df[col], color="C0", marker="o", markersize=3, label="baseline_raw")
            ax.plot(v10_df["epoch"], v10_df[col], color="C1", marker="o", markersize=3, label="v10_clean")
            ax.set_title(f"{cam} — {title}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=9)
    fig.suptitle("Training curves: baseline_raw vs v10_clean", fontsize=13, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = V10_ROOT / "training_compare_baseline_vs_v10.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[curves] saved {out_path}")
    return out_path


DEFAULT_BBOX_SIZE = 60  # px, used when teacher bbox is missing


# ---- artifact 1: comparison grid ----
def draw_subplot(ax, frame_rgb: np.ndarray, teacher_row: pd.Series | None, student_out: np.ndarray, cam: str):
    ax.imshow(frame_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    title_parts = [cam]
    for obj_name, color in [
        ("duck", "cyan"),
        ("cup", "magenta"),
    ]:
        # teacher bbox (dashed)
        t_cx = t_cy = None
        t_w = t_h = None
        t_conf = None
        if teacher_row is not None:
            x1 = teacher_row.get(f"{cam}_{obj_name}_x1", np.nan)
            y1 = teacher_row.get(f"{cam}_{obj_name}_y1", np.nan)
            x2 = teacher_row.get(f"{cam}_{obj_name}_x2", np.nan)
            y2 = teacher_row.get(f"{cam}_{obj_name}_y2", np.nan)
            if np.isfinite([x1, y1, x2, y2]).all():
                x1p = float(x1) * NATIVE_W
                y1p = float(y1) * NATIVE_H
                x2p = float(x2) * NATIVE_W
                y2p = float(y2) * NATIVE_H
                t_w = x2p - x1p
                t_h = y2p - y1p
                t_cx = 0.5 * (x1p + x2p)
                t_cy = 0.5 * (y1p + y2p)
                ax.add_patch(mpatches.Rectangle(
                    (x1p, y1p), t_w, t_h,
                    linewidth=1.4, edgecolor=color, facecolor="none", linestyle="--",
                ))
                conf_val = teacher_row.get(f"{cam}_{obj_name}_conf", np.nan)
                if conf_val is not None and np.isfinite(conf_val):
                    t_conf = float(conf_val)
                # teacher label above bbox
                tl = f"T {obj_name[0]}"
                if t_conf is not None:
                    tl += f" {t_conf:.2f}"
                ax.text(
                    x1p, max(0, y1p - 2), tl, color=color, fontsize=6,
                    verticalalignment="bottom",
                    bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"),
                )

        # student center
        if obj_name == "duck":
            s_cx = float(student_out[0]) * NATIVE_W
            s_cy = float(student_out[1]) * NATIVE_H
            s_conf = float(student_out[2])
        else:
            s_cx = float(student_out[3]) * NATIVE_W
            s_cy = float(student_out[4]) * NATIVE_H
            s_conf = float(student_out[5])

        # student bbox same size as teacher, anchored at student center
        has_teacher = t_w is not None and t_h is not None
        s_w = t_w if has_teacher else float(DEFAULT_BBOX_SIZE)
        s_h = t_h if has_teacher else float(DEFAULT_BBOX_SIZE)
        sx1 = s_cx - s_w / 2
        sy1 = s_cy - s_h / 2
        ax.add_patch(mpatches.Rectangle(
            (sx1, sy1), s_w, s_h,
            linewidth=1.4, edgecolor=color, facecolor="none", linestyle="-",
        ))
        sl = f"S {obj_name[0]} {s_conf:.2f}"
        if not has_teacher:
            sl += " (no teacher)"
        ax.text(
            sx1, min(NATIVE_H - 1, sy1 + s_h + 2), sl, color=color, fontsize=6,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"),
        )

        # px err
        if t_cx is not None and t_cy is not None:
            err = float(np.hypot(s_cx - t_cx, s_cy - t_cy))
            title_parts.append(f"{obj_name[0]}={err:.0f}px")
    ax.set_title(" ".join(title_parts), fontsize=8)


def build_comparison_grid(students: dict[str, DistilledDetector], teacher: pd.DataFrame, episodes: pd.DataFrame):
    # choose 3 val episodes spread across range 200..243
    chosen = [
        (205, "early"),
        (220, "mid"),
        (235, "late"),
    ]
    # pick frame near middle of each episode
    rows = []
    for ep_idx, tag in chosen:
        ep_len = int(episodes[episodes["episode_index"] == ep_idx].iloc[0]["length"])
        frame_idx = ep_len // 2
        rows.append((ep_idx, frame_idx, tag))

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for r, (ep_idx, frame_idx, tag) in enumerate(rows):
        # teacher row for this (ep, frame)
        trow = teacher[(teacher["episode_index"] == ep_idx) & (teacher["frame_index"] == frame_idx)]
        trow = trow.iloc[0] if len(trow) > 0 else None
        for c, cam in enumerate(CAMERAS):
            frame_rgb = read_single_frame(ep_idx, frame_idx, cam, episodes)
            out = predict_student(students[cam], frame_rgb)
            draw_subplot(axes[r, c], frame_rgb, trow, out, cam)
        axes[r, 0].set_ylabel(f"ep{ep_idx} fr{frame_idx} ({tag})", fontsize=10)
    fig.suptitle(
        "v10_clean student vs G-DINO teacher — cyan=duck, magenta=cup. dashed=teacher, solid=student",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = V10_ROOT / "comparison_student_vs_dino.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[grid] saved {out_path}")
    return out_path


# ---- artifact 2: tracking video ep210 ----
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
        nd = max(1, int(total // (dash_len + gap_len)))
        dx = (b[0] - a[0]) / total
        dy = (b[1] - a[1]) / total
        for i in range(nd + 1):
            s = i * (dash_len + gap_len)
            if s >= total:
                break
            e = min(s + dash_len, total)
            p_s = (int(a[0] + dx * s), int(a[1] + dy * s))
            p_e = (int(a[0] + dx * e), int(a[1] + dy * e))
            cv2.line(img, p_s, p_e, color, thickness)

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


def draw_overlay_on_frame(
    frame_rgb: np.ndarray, teacher_row: pd.Series | None, student_out: np.ndarray, cam: str
) -> np.ndarray:
    """Return a copy of frame_rgb with overlays: teacher dashed bbox, student solid bbox (same size)."""
    img = frame_rgb.copy()
    errs: dict[str, float | None] = {}
    for obj_name, color_rgb in [
        ("duck", (0, 255, 255)),   # cyan
        ("cup", (255, 0, 255)),    # magenta
    ]:
        t_cx = t_cy = None
        t_w = t_h = None
        t_conf = None
        has_teacher = False
        if teacher_row is not None:
            x1 = teacher_row.get(f"{cam}_{obj_name}_x1", np.nan)
            y1 = teacher_row.get(f"{cam}_{obj_name}_y1", np.nan)
            x2 = teacher_row.get(f"{cam}_{obj_name}_x2", np.nan)
            y2 = teacher_row.get(f"{cam}_{obj_name}_y2", np.nan)
            if np.isfinite([x1, y1, x2, y2]).all():
                has_teacher = True
                x1p = int(float(x1) * NATIVE_W)
                y1p = int(float(y1) * NATIVE_H)
                x2p = int(float(x2) * NATIVE_W)
                y2p = int(float(y2) * NATIVE_H)
                t_w = float(x2p - x1p)
                t_h = float(y2p - y1p)
                t_cx = 0.5 * (x1p + x2p)
                t_cy = 0.5 * (y1p + y2p)
                draw_dashed_rect(img, (x1p, y1p), (x2p, y2p), color_rgb, thickness=2)
                conf_val = teacher_row.get(f"{cam}_{obj_name}_conf", np.nan)
                if conf_val is not None and np.isfinite(conf_val):
                    t_conf = float(conf_val)
                # teacher label above
                tl = f"T{obj_name[0]}"
                if t_conf is not None:
                    tl += f" {t_conf:.2f}"
                ty = max(12, y1p - 4)
                cv2.putText(img, tl, (x1p, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, tl, (x1p, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_rgb, 1, cv2.LINE_AA)

        # student center
        if obj_name == "duck":
            s_cx = float(student_out[0]) * NATIVE_W
            s_cy = float(student_out[1]) * NATIVE_H
            s_conf = float(student_out[2])
        else:
            s_cx = float(student_out[3]) * NATIVE_W
            s_cy = float(student_out[4]) * NATIVE_H
            s_conf = float(student_out[5])

        s_w = t_w if has_teacher and t_w is not None else float(DEFAULT_BBOX_SIZE)
        s_h = t_h if has_teacher and t_h is not None else float(DEFAULT_BBOX_SIZE)
        sx1_f = s_cx - s_w / 2
        sy1_f = s_cy - s_h / 2
        sx2_f = s_cx + s_w / 2
        sy2_f = s_cy + s_h / 2
        # clip for drawing
        sx1 = int(max(0, min(NATIVE_W - 1, sx1_f)))
        sy1 = int(max(0, min(NATIVE_H - 1, sy1_f)))
        sx2 = int(max(0, min(NATIVE_W - 1, sx2_f)))
        sy2 = int(max(0, min(NATIVE_H - 1, sy2_f)))
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color_rgb, 2)
        sl = f"S{obj_name[0]} {s_conf:.2f}"
        if not has_teacher:
            sl += " (no T)"
        sy_text = min(NATIVE_H - 3, sy2 + 14)
        cv2.putText(img, sl, (sx1, sy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, sl, (sx1, sy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_rgb, 1, cv2.LINE_AA)

        # error text
        if t_cx is not None and t_cy is not None:
            errs[obj_name] = float(np.hypot(s_cx - t_cx, s_cy - t_cy))
        else:
            errs[obj_name] = None
    # cam label + errors
    err_d = errs.get("duck")
    err_c = errs.get("cup")
    err_d_s = f"{err_d:.0f}" if err_d is not None else "-"
    err_c_s = f"{err_c:.0f}" if err_c is not None else "-"
    label = f"{cam}  d={err_d_s}  c={err_c_s}px"
    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_tracking_video(
    ep_idx: int,
    students: dict[str, DistilledDetector],
    teacher: pd.DataFrame,
    episodes: pd.DataFrame,
    fps: int = 30,
):
    # Preload all frames for all 4 cams
    print(f"[video] decoding ep{ep_idx} frames for all 4 cams...")
    cam_frames: dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        cam_frames[cam] = read_episode_frames(ep_idx, cam, episodes)
    T = min(len(cam_frames[c]) for c in CAMERAS)
    print(f"[video] {T} frames")

    # teacher rows for this ep
    t_ep = teacher[teacher["episode_index"] == ep_idx].set_index("frame_index")

    # Batch per-cam inference: run each cam's model across its frames
    print("[video] running student inference per cam...")
    cam_preds: dict[str, np.ndarray] = {}
    B = 128
    for cam in CAMERAS:
        frames = cam_frames[cam][:T]
        outs = np.zeros((T, 6), dtype=np.float32)
        for start in range(0, T, B):
            batch = [frames[i] for i in range(start, min(start + B, T))]
            outs[start:start + len(batch)] = predict_student_batch(students[cam], batch)
        cam_preds[cam] = outs

    # Layout: 2x2 grid = (2*480, 2*640) = 960 x 1280 (h x w)
    grid_h = 2 * NATIVE_H
    grid_w = 2 * NATIVE_W
    tmp_path = V10_ROOT / "tracking_ep210.tmp.mp4"
    final_path = V10_ROOT / "tracking_ep210.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {tmp_path}")

    layout = [("left", 0, 0), ("right", 0, 1), ("top", 1, 0), ("gripper", 1, 1)]

    print("[video] compositing...")
    for t in range(T):
        trow = t_ep.loc[t] if t in t_ep.index else None
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for cam, rr, cc in layout:
            frame = cam_frames[cam][t]
            out = cam_preds[cam][t]
            overlay = draw_overlay_on_frame(frame, trow, out, cam)
            canvas[rr * NATIVE_H:(rr + 1) * NATIVE_H, cc * NATIVE_W:(cc + 1) * NATIVE_W] = overlay
        # cv2 wants BGR
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[video] wrote tmp mp4v file {tmp_path}")

    # Re-encode to libx264/yuv420p for Obsidian
    print("[video] ffmpeg re-encode to libx264...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "22", "-preset", "fast",
            str(final_path),
        ],
        check=True,
    )
    print(f"[video] saved {final_path}")
    try:
        tmp_path.unlink()
    except Exception:
        pass
    return final_path


# ---- driver ----
def main():
    print(f"Device: {DEVICE}")

    print("Loading dataset meta + teacher parquet...")
    info, episodes = load_dataset_meta(DATASET_PATH)
    teacher = pd.read_parquet(DATASET_PATH / "detection_results.parquet")

    print("Loading v10_clean students for all 4 cams...")
    students = {cam: load_student(V10_ROOT, cam) for cam in CAMERAS}

    grid_path = build_comparison_grid(students, teacher, episodes)
    video_path = build_tracking_video(210, students, teacher, episodes)

    print("\n=== OUTPUTS ===")
    print(grid_path)
    print(video_path)


if __name__ == "__main__":
    main()
