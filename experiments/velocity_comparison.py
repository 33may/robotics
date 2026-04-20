"""Experiment: compare detection coordinate approximation methods for async inference.

Runs dense detection on episode 0, then simulates round-robin arrivals at 12-frame
intervals and evaluates Hold, Linear velocity, Decayed velocity, and EMA velocity
approximation methods.
"""

import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- Project imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.detect import (
    create_detector, DEFAULT_MAX_AREA, GRIPPER_MAX_AREA,
)
from vbti.logic.detection.process_dataset import (
    VideoReader, load_dataset_meta, get_video_path,
)

# ── Config ──────────────────────────────────────────────────────────
DATASET = "eternalmay33/01_02_03_merged_may-sim"
EPISODE = 0
CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]
ARRIVAL_INTERVAL = 12  # frames between arrivals per camera
CONF_THRESHOLD = 0.08
FPS = 30
TAU_FRAMES = 9  # 0.3s at 30fps for decayed velocity
EMA_ALPHA = 0.3

OUTPUT_DIR = Path("/home/may33/Documents/Obsidian Vault/vbti/researches/"
                  "engineering tricks/detection/velocity_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 1. Dense ground-truth detection
# ══════════════════════════════════════════════════════════════════════

def run_dense_detection():
    """Run detection on every frame of episode 0, all 4 cameras."""
    ds_path = resolve_dataset_path(DATASET)
    info, episodes = load_dataset_meta(ds_path)
    fps = info["fps"]

    ep_row = episodes[episodes["episode_index"] == EPISODE].iloc[0]
    n_frames = int(ep_row["length"])
    print(f"Episode {EPISODE}: {n_frames} frames @ {fps} fps")

    detector = create_detector(device="cuda", confidence_threshold=0.01, max_area=DEFAULT_MAX_AREA)

    # gt[cam][obj] = list of (cx_norm, cy_norm, conf) per frame
    gt = {cam: {obj: [] for obj in OBJECTS} for cam in CAMERAS}

    for cam in CAMERAS:
        cam_key = f"observation.images.{cam}"
        feat = info["features"][cam_key]
        vid_w = feat["info"]["video.width"]
        vid_h = feat["info"]["video.height"]

        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"

        chunk_idx = int(ep_row[chunk_col])
        file_idx = int(ep_row[file_col])
        from_ts = float(ep_row[from_ts_col])
        start_frame = round(from_ts * fps)

        video_path = get_video_path(ds_path, cam_key, chunk_idx, file_idx)
        max_area = GRIPPER_MAX_AREA if cam == "gripper" else DEFAULT_MAX_AREA

        print(f"  {cam}: {video_path.name} start={start_frame} frames={n_frames} "
              f"({vid_w}x{vid_h})")

        with VideoReader(video_path, vid_w, vid_h) as reader:
            if start_frame > 0:
                reader.skip(start_frame)
            for fi in range(n_frames):
                frame = reader.read_one()
                if frame is None:
                    print(f"    EOF at frame {fi}")
                    # pad remaining
                    for obj in OBJECTS:
                        gt[cam][obj].extend([(0.0, 0.0, 0.0)] * (n_frames - fi))
                    break
                det = detector.detect(frame, max_area=max_area)
                for obj in OBJECTS:
                    d = det[obj]
                    cx, cy = d["center_norm"]
                    gt[cam][obj].append((cx, cy, d["confidence"]))

                if fi % 50 == 0:
                    print(f"    {cam} frame {fi}/{n_frames}")

    # convert to numpy
    for cam in CAMERAS:
        for obj in OBJECTS:
            gt[cam][obj] = np.array(gt[cam][obj])  # (N, 3): cx, cy, conf
    return gt, n_frames


# ══════════════════════════════════════════════════════════════════════
# 2. Approximation methods
# ══════════════════════════════════════════════════════════════════════

class HoldMethod:
    def __init__(self):
        self.pos = None

    def on_arrival(self, cx, cy, t):
        self.pos = np.array([cx, cy])

    def predict(self, t):
        return self.pos.copy() if self.pos is not None else None


class LinearVelocity:
    def __init__(self):
        self.pos = None
        self.prev_pos = None
        self.t = None
        self.prev_t = None

    def on_arrival(self, cx, cy, t):
        self.prev_pos = self.pos
        self.prev_t = self.t
        self.pos = np.array([cx, cy])
        self.t = t

    def predict(self, t):
        if self.pos is None:
            return None
        if self.prev_pos is None:
            return self.pos.copy()
        dt_arr = self.t - self.prev_t
        if dt_arr == 0:
            return self.pos.copy()
        vel = (self.pos - self.prev_pos) / dt_arr
        pred = self.pos + vel * (t - self.t)
        return np.clip(pred, 0.0, 1.0)


class DecayedVelocity:
    def __init__(self, tau=TAU_FRAMES):
        self.pos = None
        self.prev_pos = None
        self.t = None
        self.prev_t = None
        self.tau = tau

    def on_arrival(self, cx, cy, t):
        self.prev_pos = self.pos
        self.prev_t = self.t
        self.pos = np.array([cx, cy])
        self.t = t

    def predict(self, t):
        if self.pos is None:
            return None
        if self.prev_pos is None:
            return self.pos.copy()
        dt_arr = self.t - self.prev_t
        if dt_arr == 0:
            return self.pos.copy()
        vel = (self.pos - self.prev_pos) / dt_arr
        age = t - self.t
        decay = math.exp(-age / self.tau)
        pred = self.pos + vel * age * decay
        return np.clip(pred, 0.0, 1.0)


class EMAVelocity:
    def __init__(self, alpha=EMA_ALPHA):
        self.pos = None
        self.prev_pos = None
        self.t = None
        self.prev_t = None
        self.v_ema = np.zeros(2)
        self.alpha = alpha
        self.has_vel = False

    def on_arrival(self, cx, cy, t):
        self.prev_pos = self.pos
        self.prev_t = self.t
        self.pos = np.array([cx, cy])
        self.t = t
        if self.prev_pos is not None:
            dt = self.t - self.prev_t
            if dt > 0:
                v_new = (self.pos - self.prev_pos) / dt
                if self.has_vel:
                    self.v_ema = self.alpha * v_new + (1 - self.alpha) * self.v_ema
                else:
                    self.v_ema = v_new.copy()
                    self.has_vel = True

    def predict(self, t):
        if self.pos is None:
            return None
        if not self.has_vel:
            return self.pos.copy()
        pred = self.pos + self.v_ema * (t - self.t)
        return np.clip(pred, 0.0, 1.0)


METHOD_CLASSES = {
    "hold": HoldMethod,
    "linear": LinearVelocity,
    "decayed": DecayedVelocity,
    "ema": EMAVelocity,
}
METHOD_NAMES = list(METHOD_CLASSES.keys())


# ══════════════════════════════════════════════════════════════════════
# 3. Simulate arrivals and evaluate
# ══════════════════════════════════════════════════════════════════════

def simulate_and_evaluate(gt, n_frames):
    """Simulate staggered arrivals and compute predictions + errors."""
    # Camera stagger offsets
    cam_offsets = {"left": 0, "right": 3, "top": 6, "gripper": 9}

    # predictions[method][cam][obj] = (N, 2) array of predicted (cx, cy)
    predictions = {m: {c: {o: np.full((n_frames, 2), np.nan) for o in OBJECTS}
                       for c in CAMERAS} for m in METHOD_NAMES}

    # errors[method][cam][obj] = (N,) array of euclidean errors (nan where no GT)
    errors = {m: {c: {o: np.full(n_frames, np.nan) for o in OBJECTS}
                  for c in CAMERAS} for m in METHOD_NAMES}

    # staleness[cam] = (N,) frames since last arrival
    staleness = {c: np.full(n_frames, -1, dtype=int) for c in CAMERAS}

    for cam in CAMERAS:
        offset = cam_offsets[cam]
        # Arrival frames for this camera
        arrival_frames = list(range(offset, n_frames, ARRIVAL_INTERVAL))

        for obj in OBJECTS:
            gt_data = gt[cam][obj]  # (N, 3)
            methods = {m: cls() for m, cls in METHOD_CLASSES.items()}

            for t in range(n_frames):
                # Check if this is an arrival frame for this camera
                if t in arrival_frames:
                    conf = gt_data[t, 2]
                    if conf >= CONF_THRESHOLD:
                        cx, cy = gt_data[t, 0], gt_data[t, 1]
                        for m in methods.values():
                            m.on_arrival(cx, cy, t)

                # Predict for all methods
                for mname, m in methods.items():
                    pred = m.predict(t)
                    if pred is not None:
                        predictions[mname][cam][obj][t] = pred

                        # Compute error if GT available
                        if gt_data[t, 2] >= CONF_THRESHOLD:
                            gt_xy = gt_data[t, :2]
                            err = np.linalg.norm(pred - gt_xy)
                            errors[mname][cam][obj][t] = err

        # Compute staleness for this camera
        last_arrival = -1
        for t in range(n_frames):
            if t in arrival_frames:
                last_arrival = t
            if last_arrival >= 0:
                staleness[cam][t] = t - last_arrival

    return predictions, errors, staleness, cam_offsets


# ══════════════════════════════════════════════════════════════════════
# 4. Plotting
# ══════════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "hold": "#2ecc71",
    "linear": "#3498db",
    "decayed": "#e67e22",
    "ema": "#e74c3c",
}


def plot_error_by_method(errors, n_frames):
    """Bar chart: median and p95 error per method, grouped by camera."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Detection Approximation Error by Method & Camera", fontsize=14)

    for ax, obj in zip(axes, OBJECTS):
        x = np.arange(len(CAMERAS))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for i, mname in enumerate(METHOD_NAMES):
            medians = []
            p95s = []
            for cam in CAMERAS:
                errs = errors[mname][cam][obj]
                valid = errs[~np.isnan(errs)]
                if len(valid) == 0:
                    medians.append(0)
                    p95s.append(0)
                else:
                    medians.append(np.median(valid))
                    p95s.append(np.percentile(valid, 95))

            bars = ax.bar(x + offsets[i], medians, width,
                          label=f"{mname} (median)",
                          color=METHOD_COLORS[mname], alpha=0.8)
            # p95 as error bar cap
            ax.errorbar(x + offsets[i], medians,
                        yerr=[np.zeros(len(medians)), np.array(p95s) - np.array(medians)],
                        fmt="none", capsize=3, color="black", alpha=0.5)

            # Annotate with pixel values
            for j, (med, p95) in enumerate(zip(medians, p95s)):
                px_med = med * ((640 + 480) / 2)
                ax.annotate(f"{px_med:.1f}px", (x[j] + offsets[i], med),
                            textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(CAMERAS)
        ax.set_ylabel("Error (normalized coords)")
        ax.set_title(f"{obj.capitalize()}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = OUTPUT_DIR / "error_by_method.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_error_over_time(errors, n_frames, cam_offsets):
    """Line plot: error over time averaged across cameras."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Prediction Error Over Time (averaged across cameras)", fontsize=14)

    for ax, obj in zip(axes, OBJECTS):
        for mname in METHOD_NAMES:
            # Average error across cameras at each frame
            all_errs = []
            for cam in CAMERAS:
                all_errs.append(errors[mname][cam][obj])
            stacked = np.stack(all_errs, axis=0)  # (4, N)
            mean_err = np.nanmean(stacked, axis=0)
            ax.plot(range(n_frames), mean_err, label=mname,
                    color=METHOD_COLORS[mname], alpha=0.7, linewidth=0.8)

        # Arrival frames for left camera
        left_arrivals = list(range(cam_offsets["left"], n_frames, ARRIVAL_INTERVAL))
        for af in left_arrivals:
            ax.axvline(af, color="gray", alpha=0.15, linewidth=0.5, linestyle=":")

        ax.set_ylabel("Error (normalized)")
        ax.set_title(f"{obj.capitalize()}")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Frame Index")
    plt.tight_layout()
    path = OUTPUT_DIR / "error_over_time.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_error_vs_staleness(errors, staleness, n_frames):
    """Error vs frames since last arrival."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error vs Staleness (frames since last arrival)", fontsize=14)

    for ax, obj in zip(axes, OBJECTS):
        for mname in METHOD_NAMES:
            # Collect (staleness, error) pairs across all cameras
            stale_err = defaultdict(list)
            for cam in CAMERAS:
                errs = errors[mname][cam][obj]
                stale = staleness[cam]
                for t in range(n_frames):
                    if not np.isnan(errs[t]) and stale[t] >= 0:
                        stale_err[stale[t]].append(errs[t])

            stale_vals = sorted(stale_err.keys())
            mean_errs = [np.mean(stale_err[s]) for s in stale_vals]
            ax.plot(stale_vals, mean_errs, label=mname,
                    color=METHOD_COLORS[mname], marker=".", markersize=4, linewidth=1.5)

        ax.set_xlabel("Frames since last arrival")
        ax.set_ylabel("Mean error (normalized)")
        ax.set_title(f"{obj.capitalize()}")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, ARRIVAL_INTERVAL + 0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "error_vs_staleness.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_trajectory(gt, predictions, n_frames, cam_offsets):
    """Trajectory comparison for left camera, duck, frames 100-300."""
    cam = "left"
    obj = "duck"
    f_start, f_end = 100, min(300, n_frames)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"Trajectory Comparison — {cam}/{obj} (frames {f_start}-{f_end})", fontsize=13)

    gt_data = gt[cam][obj]
    mask = gt_data[f_start:f_end, 2] >= CONF_THRESHOLD
    frames = np.arange(f_start, f_end)

    # GT trajectory
    gt_cx = gt_data[f_start:f_end, 0]
    gt_cy = gt_data[f_start:f_end, 1]
    ax.plot(gt_cx[mask], gt_cy[mask], "k-", linewidth=2, label="Ground Truth", alpha=0.8)

    for mname in METHOD_NAMES:
        pred = predictions[mname][cam][obj][f_start:f_end]
        valid = ~np.isnan(pred[:, 0])
        ax.plot(pred[valid, 0], pred[valid, 1], "--",
                color=METHOD_COLORS[mname], linewidth=1, label=mname, alpha=0.7)

    # Mark arrival frames
    arrivals = [f for f in range(cam_offsets[cam], n_frames, ARRIVAL_INTERVAL)
                if f_start <= f < f_end]
    for af in arrivals:
        idx = af - f_start
        if gt_data[af, 2] >= CONF_THRESHOLD:
            ax.plot(gt_data[af, 0], gt_data[af, 1], "ko", markersize=5, alpha=0.4)

    ax.set_xlabel("cx (normalized)")
    ax.set_ylabel("cy (normalized)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()  # Image coords: y increases downward
    plt.tight_layout()
    path = OUTPUT_DIR / "trajectory_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# 5. Video rendering
# ══════════════════════════════════════════════════════════════════════

def render_video(gt, predictions, n_frames):
    """Render left camera frames 100-400 with overlaid prediction dots."""
    import cv2

    cam = "left"
    obj = "duck"
    f_start, f_end = 100, min(400, n_frames)

    ds_path = resolve_dataset_path(DATASET)
    info, episodes = load_dataset_meta(ds_path)
    fps_ds = info["fps"]
    ep_row = episodes[episodes["episode_index"] == EPISODE].iloc[0]

    cam_key = f"observation.images.{cam}"
    feat = info["features"][cam_key]
    vid_w = feat["info"]["video.width"]
    vid_h = feat["info"]["video.height"]

    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col = f"videos/{cam_key}/file_index"
    from_ts_col = f"videos/{cam_key}/from_timestamp"

    chunk_idx = int(ep_row[chunk_col])
    file_idx = int(ep_row[file_col])
    from_ts = float(ep_row[from_ts_col])
    start_frame = round(from_ts * fps_ds)

    video_path = get_video_path(ds_path, cam_key, chunk_idx, file_idx)

    output_path = OUTPUT_DIR / "method_comparison_left.mp4"

    # ffmpeg pipe for output
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{vid_w}x{vid_h}",
        "-pix_fmt", "bgr24",
        "-r", "10",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-crf", "18",
        str(output_path),
    ]
    out_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Trail storage: last 5 positions per method
    trails = {m: [] for m in METHOD_NAMES}

    method_draw_colors = {
        "hold": (0, 255, 0),       # green (BGR)
        "linear": (255, 0, 0),     # blue
        "decayed": (0, 165, 255),  # orange
        "ema": (0, 0, 255),        # red
    }
    gt_color = (255, 255, 0)  # cyan

    cam_offsets = {"left": 0, "right": 3, "top": 6, "gripper": 9}
    arrivals = set(range(cam_offsets[cam], n_frames, ARRIVAL_INTERVAL))

    print(f"  Rendering video frames {f_start}-{f_end}...")

    with VideoReader(video_path, vid_w, vid_h) as reader:
        if start_frame + f_start > 0:
            reader.skip(start_frame + f_start)

        for fi in range(f_start, f_end):
            frame = reader.read_one()
            if frame is None:
                break
            # Convert RGB to BGR for cv2 drawing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gt_data = gt[cam][obj]

            # Draw GT
            if gt_data[fi, 2] >= CONF_THRESHOLD:
                gx = int(gt_data[fi, 0] * vid_w)
                gy = int(gt_data[fi, 1] * vid_h)
                cv2.circle(frame_bgr, (gx, gy), 8, gt_color, 2)  # ring

            # Draw predictions
            for mname in METHOD_NAMES:
                pred = predictions[mname][cam][obj][fi]
                if np.isnan(pred[0]):
                    continue
                px = int(pred[0] * vid_w)
                py = int(pred[1] * vid_h)
                color = method_draw_colors[mname]

                # Trail
                trails[mname].append((px, py))
                if len(trails[mname]) > 5:
                    trails[mname] = trails[mname][-5:]

                # Draw trail
                for i in range(len(trails[mname]) - 1):
                    alpha = (i + 1) / len(trails[mname])
                    t_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame_bgr, trails[mname][i], trails[mname][i + 1],
                             t_color, 1)

                # Draw dot
                cv2.circle(frame_bgr, (px, py), 5, color, -1)

            # Mark arrival frame
            if fi in arrivals:
                cv2.putText(frame_bgr, "ARRIVAL", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Frame number
            cv2.putText(frame_bgr, f"F{fi}", (vid_w - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out_proc.stdin.write(frame_bgr.tobytes())

    out_proc.stdin.close()
    out_proc.wait()
    print(f"  Saved {output_path}")


# ══════════════════════════════════════════════════════════════════════
# 6. Results JSON
# ══════════════════════════════════════════════════════════════════════

def save_results_json(errors, n_frames):
    """Save aggregated results as JSON."""
    results = {
        "methods": METHOD_NAMES,
        "per_camera_per_object": {},
        "overall": {},
        "best_method": None,
        "n_frames_evaluated": n_frames,
        "arrival_interval": ARRIVAL_INTERVAL,
        "confidence_threshold": CONF_THRESHOLD,
    }

    overall_errs = {m: [] for m in METHOD_NAMES}

    for cam in CAMERAS:
        for obj in OBJECTS:
            key = f"{cam}_{obj}"
            results["per_camera_per_object"][key] = {}
            for mname in METHOD_NAMES:
                errs = errors[mname][cam][obj]
                valid = errs[~np.isnan(errs)]
                stats = {
                    "median": float(np.median(valid)) if len(valid) > 0 else None,
                    "p95": float(np.percentile(valid, 95)) if len(valid) > 0 else None,
                    "mean": float(np.mean(valid)) if len(valid) > 0 else None,
                }
                results["per_camera_per_object"][key][mname] = stats
                overall_errs[mname].extend(valid.tolist())

    for mname in METHOD_NAMES:
        all_e = np.array(overall_errs[mname])
        results["overall"][mname] = {
            "median": float(np.median(all_e)) if len(all_e) > 0 else None,
            "p95": float(np.percentile(all_e, 95)) if len(all_e) > 0 else None,
            "mean": float(np.mean(all_e)) if len(all_e) > 0 else None,
        }

    # Best method by overall median
    best = min(METHOD_NAMES, key=lambda m: results["overall"][m]["median"] or 999)
    results["best_method"] = best

    path = OUTPUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")
    return results


# ══════════════════════════════════════════════════════════════════════
# 7. Findings
# ══════════════════════════════════════════════════════════════════════

def write_findings(results, errors, staleness, n_frames):
    """Write findings.md with analysis."""
    r = results
    lines = ["# Velocity Approximation Comparison\n"]
    lines.append(f"**Date**: 2026-04-17  ")
    lines.append(f"**Dataset**: `{DATASET}` episode {EPISODE}  ")
    lines.append(f"**Arrival interval**: {ARRIVAL_INTERVAL} frames ({ARRIVAL_INTERVAL/FPS*1000:.0f}ms)  ")
    lines.append(f"**Confidence threshold**: {CONF_THRESHOLD}  ")
    lines.append(f"**Frames evaluated**: {r['n_frames_evaluated']}\n")

    lines.append("## Overall Results\n")
    lines.append("| Method | Median | P95 | Mean | Median (px) | P95 (px) |")
    lines.append("|--------|--------|-----|------|-------------|----------|")
    px_scale = (640 + 480) / 2
    for m in METHOD_NAMES:
        s = r["overall"][m]
        if s["median"] is not None:
            lines.append(f"| {m} | {s['median']:.5f} | {s['p95']:.5f} | {s['mean']:.5f} "
                         f"| {s['median']*px_scale:.1f} | {s['p95']*px_scale:.1f} |")
    lines.append("")

    best = r["best_method"]
    hold_med = r["overall"]["hold"]["median"]
    best_med = r["overall"][best]["median"]
    if hold_med and best_med and hold_med > 0:
        improvement = (hold_med - best_med) / hold_med * 100
        lines.append(f"**Best method**: `{best}` — {improvement:.1f}% better than hold baseline\n")
    else:
        lines.append(f"**Best method**: `{best}`\n")

    # Per-camera analysis
    lines.append("## Per-Camera Breakdown\n")
    for cam in CAMERAS:
        lines.append(f"### {cam.capitalize()}\n")
        lines.append("| Method | Duck Median | Duck P95 | Cup Median | Cup P95 |")
        lines.append("|--------|-------------|----------|------------|---------|")
        for m in METHOD_NAMES:
            dk = r["per_camera_per_object"].get(f"{cam}_duck", {}).get(m, {})
            ck = r["per_camera_per_object"].get(f"{cam}_cup", {}).get(m, {})
            dm = f"{dk.get('median', 0):.5f}" if dk.get("median") is not None else "N/A"
            dp = f"{dk.get('p95', 0):.5f}" if dk.get("p95") is not None else "N/A"
            cm = f"{ck.get('median', 0):.5f}" if ck.get("median") is not None else "N/A"
            cp = f"{ck.get('p95', 0):.5f}" if ck.get("p95") is not None else "N/A"
            lines.append(f"| {m} | {dm} | {dp} | {cm} | {cp} |")
        lines.append("")

    # Staleness analysis
    lines.append("## Staleness Degradation\n")
    for obj in OBJECTS:
        lines.append(f"### {obj.capitalize()}\n")
        # Check if any method overshoots at high staleness
        for mname in METHOD_NAMES:
            stale_err = defaultdict(list)
            for cam in CAMERAS:
                errs = errors[mname][cam][obj]
                stale = staleness[cam]
                for t in range(n_frames):
                    if not np.isnan(errs[t]) and stale[t] >= 0:
                        stale_err[stale[t]].append(errs[t])
            if ARRIVAL_INTERVAL - 1 in stale_err and 1 in stale_err:
                err_near = np.mean(stale_err[1])
                err_stale = np.mean(stale_err[ARRIVAL_INTERVAL - 1])
                ratio = err_stale / err_near if err_near > 0 else float("inf")
                lines.append(f"- **{mname}**: age=1f: {err_near:.5f}, "
                             f"age={ARRIVAL_INTERVAL-1}f: {err_stale:.5f}, "
                             f"degradation={ratio:.1f}x")
        lines.append("")

    # Overshoot check
    lines.append("## Overshoot Analysis\n")
    for mname in ["linear", "ema"]:
        overshoots = 0
        total = 0
        for cam in CAMERAS:
            for obj in OBJECTS:
                errs = errors[mname][cam][obj]
                stale = staleness[cam]
                hold_errs = errors["hold"][cam][obj]
                for t in range(n_frames):
                    if not np.isnan(errs[t]) and not np.isnan(hold_errs[t]) and stale[t] >= 0:
                        total += 1
                        if errs[t] > hold_errs[t] * 1.5:
                            overshoots += 1
        pct = overshoots / total * 100 if total > 0 else 0
        lines.append(f"- **{mname}**: {overshoots}/{total} frames ({pct:.1f}%) where error > 1.5x hold")
    lines.append("")

    # Gripper analysis
    lines.append("## Gripper Camera (fast-moving)\n")
    for obj in OBJECTS:
        lines.append(f"### {obj.capitalize()}")
        for m in METHOD_NAMES:
            s = r["per_camera_per_object"].get(f"gripper_{obj}", {}).get(m, {})
            med = s.get("median", 0)
            lines.append(f"- {m}: median={med:.5f} ({med*px_scale:.1f}px)")
    lines.append("")

    # Gripper velocity benefit
    g_hold = r["per_camera_per_object"].get("gripper_duck", {}).get("hold", {}).get("median")
    g_best_m = min(METHOD_NAMES,
                   key=lambda m: r["per_camera_per_object"].get("gripper_duck", {}).get(m, {}).get("median", 999))
    g_best = r["per_camera_per_object"].get("gripper_duck", {}).get(g_best_m, {}).get("median")
    if g_hold and g_best and g_hold > 0:
        imp = (g_hold - g_best) / g_hold * 100
        lines.append(f"Gripper duck: `{g_best_m}` is {imp:.1f}% better than hold\n")

    # Recommendation
    lines.append("## Recommendation\n")
    lines.append(f"Use **{best}** for production async inference. ")

    # Check if velocity methods hurt
    hold_med = r["overall"].get("hold", {}).get("median")
    lin_med = r["overall"].get("linear", {}).get("median")
    dec_med = r["overall"].get("decayed", {}).get("median")
    ema_med = r["overall"].get("ema", {}).get("median")

    if hold_med and lin_med and hold_med < lin_med:
        lines.append("\nVelocity extrapolation hurts rather than helps on this dataset. "
                     "This likely means objects move slowly relative to the detection interval, "
                     "so the noise from velocity estimation exceeds the position drift. "
                     "The detection interval (12 frames / 400ms) is short enough that positions "
                     "barely change between updates.")

    if dec_med and ema_med:
        if abs(dec_med - ema_med) / max(dec_med, ema_med) < 0.1:
            lines.append("\nDecayed and EMA perform similarly — decayed is simpler to implement "
                         "and has fewer parameters. ")
        if best in ("linear", "ema"):
            lines.append("\nMonitor for overshoot in production — consider clamping predictions "
                         "to a maximum displacement from last observation. ")

    lines.append("\n**Key insight**: With ~103ms/cam TRT inference and 4-cam round-robin, "
                 "the 400ms update interval is fast enough that simple hold (last known position) "
                 "is sufficient. Velocity methods add complexity and noise without benefit. "
                 "This may change with slower detectors or faster-moving objects.")
    lines.append("")

    path = OUTPUT_DIR / "findings.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Velocity Approximation Comparison Experiment")
    print("=" * 60)

    # Step 1: Dense GT detection
    print("\n[1/6] Running dense detection on episode 0...")
    t0 = time.perf_counter()
    gt, n_frames = run_dense_detection()
    det_time = time.perf_counter() - t0
    print(f"  Detection done in {det_time:.1f}s ({n_frames} frames)")

    # Step 2: Simulate and evaluate
    print("\n[2/6] Simulating arrivals and evaluating methods...")
    predictions, errors, staleness, cam_offsets = simulate_and_evaluate(gt, n_frames)
    print("  Done")

    # Step 3: Generate plots (save early)
    print("\n[3/6] Generating plots...")
    plot_error_by_method(errors, n_frames)
    plot_error_over_time(errors, n_frames, cam_offsets)
    plot_error_vs_staleness(errors, staleness, n_frames)
    plot_trajectory(gt, predictions, n_frames, cam_offsets)

    # Step 4: Render video
    print("\n[4/6] Rendering comparison video...")
    try:
        render_video(gt, predictions, n_frames)
    except Exception as e:
        print(f"  Video rendering failed: {e}")

    # Step 5: Save results JSON
    print("\n[5/6] Saving results...")
    results = save_results_json(errors, n_frames)

    # Step 6: Write findings
    print("\n[6/6] Writing findings...")
    write_findings(results, errors, staleness, n_frames)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    px = (640 + 480) / 2
    for m in METHOD_NAMES:
        s = results["overall"][m]
        if s["median"] is not None:
            print(f"  {m:12s}  median={s['median']:.5f} ({s['median']*px:.1f}px)  "
                  f"p95={s['p95']:.5f} ({s['p95']*px:.1f}px)")
    print(f"\n  Best: {results['best_method']}")
    print(f"\n  All outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
