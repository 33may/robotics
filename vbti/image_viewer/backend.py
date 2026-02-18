"""Processing backend for COLMAP Frame Curator. Zero Gradio imports."""

import cv2
import json
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vbti.utils.video_utils import score_frame, get_video_info


@dataclass
class FrameData:
    frame_idx: int
    score: float
    timestamp_sec: float
    full_path: str
    thumb_path: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return FrameData(**d)


def _ffprobe_duration(path: str) -> float | None:
    """Get duration from ffprobe (reliable for HEVC/.mov where OpenCV isn't)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        pass
    return None


def probe_video(path: str) -> dict:
    """Get video metadata. Uses OpenCV for resolution/fps, ffprobe for duration."""
    info = get_video_info(path)

    # OpenCV's frame count / duration is unreliable for HEVC/.mov — use ffprobe
    ffprobe_dur = _ffprobe_duration(path)
    if ffprobe_dur and ffprobe_dur > info["duration_sec"] * 1.5:
        info["duration_sec"] = ffprobe_dur
        info["total_frames"] = int(ffprobe_dur * info["fps"])

    return {
        "fps": info["fps"],
        "total_frames": info["total_frames"],
        "width": info["width"],
        "height": info["height"],
        "duration_sec": round(info["duration_sec"], 2),
    }


def init_work_dir(video_path: str) -> str:
    """Create temp working directory with full/ and thumbs/ subdirs."""
    stem = Path(video_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(f"/tmp/colmap_curator_{stem}_{ts}")
    (work_dir / "full").mkdir(parents=True, exist_ok=True)
    (work_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def extract_frames(video_path: str, work_dir: str, k_per_second: int,
                   progress_cb=None) -> list[dict]:
    """
    Extract ALL frames from video in a single pass (constant memory).

    Processes one 1-second batch at a time: scores all frames in the batch,
    picks top-k as preselected, writes all to disk, then releases the batch.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Don't trust CAP_PROP_FRAME_COUNT (wrong for HEVC/.mov) — use ffprobe estimate for progress only
    est_total = _ffprobe_duration(video_path)
    est_total = int(est_total * fps) if est_total else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_dir = Path(work_dir) / "full"
    thumb_dir = Path(work_dir) / "thumbs"

    frames_out = []
    batch = []  # (frame_idx, frame_bgr, score) — holds at most `fps` frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = score_frame(frame)
        batch.append((frame_idx, frame, score))

        # Process batch when we hit a full second
        if len(batch) == fps:
            _process_batch(batch, k_per_second, fps, full_dir, thumb_dir, frames_out)
            batch = []

        if progress_cb and frame_idx % 30 == 0:
            progress_cb(frame_idx, est_total, "Extracting frames")

        frame_idx += 1

    # Process remaining frames
    if batch:
        _process_batch(batch, k_per_second, fps, full_dir, thumb_dir, frames_out)

    cap.release()

    if progress_cb:
        progress_cb(frame_idx, frame_idx, "Done")

    return frames_out


def _process_batch(batch, k_per_second, fps, full_dir, thumb_dir, frames_out):
    """Score-rank a 1-second batch, write all to disk, mark top-k as preselected."""
    # Determine top-k by score
    ranked = sorted(batch, key=lambda x: x[2], reverse=True)
    preselected = {item[0] for item in ranked[:k_per_second]}

    # Write all frames in chronological order
    for idx, frame_bgr, score in batch:
        timestamp = idx / fps if fps > 0 else 0.0

        full_path = str(full_dir / f"frame_{idx:06d}.jpg")
        cv2.imwrite(full_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        h, w = frame_bgr.shape[:2]
        thumb = cv2.resize(frame_bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        thumb_path = str(thumb_dir / f"thumb_{idx:06d}.jpg")
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])

        frames_out.append({
            "frame_idx": idx,
            "score": round(score, 2),
            "timestamp_sec": round(timestamp, 2),
            "full_path": full_path,
            "thumb_path": thumb_path,
            "preselected": idx in preselected,
        })


def get_gallery_data(frames: list[dict], index_set: dict,
                     show_accepted: bool = True) -> list[tuple]:
    """
    Build (thumb_path, caption) tuples for gr.Gallery.

    frames: list of FrameData dicts
    index_set: dict of {frame_idx: True} for accepted or rejected
    """
    result = []
    for fd in frames:
        if fd["frame_idx"] in index_set:
            caption = f"#{fd['frame_idx']} | {fd['score']:.1f} | {fd['timestamp_sec']:.1f}s"
            result.append((fd["thumb_path"], caption))
    return result


def generate_histogram(frames: list[dict], dark=True):
    """Score distribution histogram. Returns matplotlib Figure."""
    bg = "#1f2937" if dark else "white"
    fg = "#e2e8f0" if dark else "black"

    if not frames:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=bg)
        ax.set_facecolor(bg)
        ax.text(0.5, 0.5, "No frames", ha="center", va="center", color=fg)
        return fig

    scores = [fd["score"] for fd in frames]
    scores = np.array(scores)

    fig, ax = plt.subplots(figsize=(6, 3), facecolor=bg)
    ax.set_facecolor(bg)
    ax.hist(scores, bins=min(30, len(scores)), edgecolor=bg, color="#3b82f6", alpha=0.8)
    ax.axvline(scores.mean(), color="#ef4444", linestyle="--",
               label=f"mean={scores.mean():.1f}")
    ax.set_xlabel("Sharpness (Laplacian variance)", color=fg)
    ax.set_ylabel("Count", color=fg)
    ax.set_title(f"Score distribution — {len(scores)} frames", color=fg)
    ax.tick_params(colors=fg)
    ax.legend(facecolor=bg, edgecolor="#334155", labelcolor=fg)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


def export_accepted(frames: list[dict], accepted: dict, output_dir: str,
                    fmt: str = "jpg", quality: int = 90,
                    progress_cb=None) -> tuple[str, int]:
    """
    Export accepted frames with sequential COLMAP naming.

    Returns (output_dir, count).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    accepted_frames = [fd for fd in frames if fd["frame_idx"] in accepted]
    # Already chronological from extract_frames

    for i, fd in enumerate(accepted_frames, start=1):
        ext = fmt
        src = fd["full_path"]
        dst = out / f"frame_{i:05d}.{ext}"

        if fmt == "png":
            # Re-encode as PNG
            img = cv2.imread(src)
            cv2.imwrite(str(dst), img)
        else:
            if quality == 90:
                # Source is already q=90 JPEG, just copy
                shutil.copy2(src, str(dst))
            else:
                img = cv2.imread(src)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if progress_cb:
            progress_cb(i, len(accepted_frames), "Exporting")

    return str(out), len(accepted_frames)


def find_frame_by_gallery_index(frames: list[dict], index_set: dict,
                                gallery_idx: int) -> dict | None:
    """Map a gallery click index back to the FrameData dict."""
    visible = [fd for fd in frames if fd["frame_idx"] in index_set]
    if 0 <= gallery_idx < len(visible):
        return visible[gallery_idx]
    return None
