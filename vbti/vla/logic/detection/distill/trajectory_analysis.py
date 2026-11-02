"""Trajectory consistency analysis across episodes.

Runs dense detection (stride=3) on sampled episodes, then plots and
computes statistics to measure trajectory consistency across episodes.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.detect import (
    create_detector, DEFAULT_MAX_AREA, GRIPPER_MAX_AREA,
)
from vbti.logic.detection.process_dataset import (
    VideoReader, load_dataset_meta, get_video_path,
)

# ── Config ──────────────────────────────────────────────────────────
DATASET = "eternalmay33/01_02_03_merged_may-sim"
EPISODES = [0, 10, 20, 50, 100, 200]
CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]
STRIDE = 3
CONF_THRESHOLD = 0.08
RESAMPLE_N = 100

OUT_DIR = Path.home() / "Documents" / "Obsidian Vault" / "vbti" / "researches" / "engineering tricks" / "detection" / "trajectory_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_detection_on_episode(detector, ds_path, info, episodes_df, ep_idx, cameras, cam_resolution, fps):
    """Run detection on a single episode for all cameras. Returns dict[cam] -> list of per-frame dicts."""
    ep_row = episodes_df[episodes_df["episode_index"] == ep_idx].iloc[0]
    n_frames = int(ep_row["length"])

    # Pre-allocate: cam -> frame_idx -> {duck_cx, duck_cy, duck_conf, cup_cx, ...}
    results = {cam: [None] * n_frames for cam in cameras}

    for cam in cameras:
        cam_key = f"observation.images.{cam}"
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"

        chunk_idx = int(ep_row[chunk_col])
        file_idx = int(ep_row[file_col])
        from_ts = float(ep_row[from_ts_col])
        start_frame = round(from_ts * fps)

        vid_w, vid_h = cam_resolution[cam]
        video_path = get_video_path(ds_path, cam_key, chunk_idx, file_idx)
        max_area = GRIPPER_MAX_AREA if cam == "gripper" else DEFAULT_MAX_AREA

        with VideoReader(video_path, vid_w, vid_h) as reader:
            for local_idx, frame in reader.read_range(start_frame, n_frames, STRIDE):
                det = detector.detect(frame, max_area=max_area)
                results[cam][local_idx] = det

    # Interpolate skipped frames (simple linear on cx/cy for detected frames)
    interpolated = {}
    for cam in cameras:
        cam_data = {
            "duck_cx": np.full(n_frames, np.nan),
            "duck_cy": np.full(n_frames, np.nan),
            "duck_conf": np.full(n_frames, 0.0),
            "cup_cx": np.full(n_frames, np.nan),
            "cup_cy": np.full(n_frames, np.nan),
            "cup_conf": np.full(n_frames, 0.0),
        }

        for fi in range(n_frames):
            det = results[cam][fi]
            if det is None:
                continue
            for obj in OBJECTS:
                cx, cy = det[obj]["center_norm"]
                conf = det[obj]["confidence"]
                cam_data[f"{obj}_cx"][fi] = cx
                cam_data[f"{obj}_cy"][fi] = cy
                cam_data[f"{obj}_conf"][fi] = conf

        # Linear interpolation for cx/cy (only between detected frames)
        for obj in OBJECTS:
            for coord in ["cx", "cy"]:
                arr = cam_data[f"{obj}_{coord}"]
                valid = ~np.isnan(arr)
                if valid.sum() >= 2:
                    x_valid = np.where(valid)[0]
                    y_valid = arr[valid]
                    f_interp = interpolate.interp1d(
                        x_valid, y_valid, kind="linear",
                        bounds_error=False, fill_value=(y_valid[0], y_valid[-1])
                    )
                    cam_data[f"{obj}_{coord}"] = f_interp(np.arange(n_frames))
                elif valid.sum() == 1:
                    cam_data[f"{obj}_{coord}"][:] = arr[valid][0]
                # else stays NaN

            # For conf, also interpolate (for filtering later we use the raw sampled values)
            conf_arr = cam_data[f"{obj}_conf"]
            valid = conf_arr > 0
            if valid.sum() >= 2:
                x_valid = np.where(valid)[0]
                y_valid = conf_arr[valid]
                f_interp = interpolate.interp1d(
                    x_valid, y_valid, kind="linear",
                    bounds_error=False, fill_value=(y_valid[0], y_valid[-1])
                )
                cam_data[f"{obj}_conf"] = f_interp(np.arange(n_frames))

        interpolated[cam] = cam_data

    return interpolated, n_frames


def resample_trajectory(cx, cy, conf, n_points=RESAMPLE_N, conf_thresh=CONF_THRESHOLD):
    """Resample a trajectory to n_points, only using frames with conf >= threshold.
    Returns (cx_resampled, cy_resampled, valid_fraction)."""
    valid = conf >= conf_thresh
    if valid.sum() < 2:
        return np.full(n_points, np.nan), np.full(n_points, np.nan), 0.0

    # Normalize frame indices to 0-1
    idxs = np.where(valid)[0]
    t_valid = idxs / (len(cx) - 1) if len(cx) > 1 else np.zeros(len(idxs))
    cx_valid = cx[valid]
    cy_valid = cy[valid]

    t_out = np.linspace(0, 1, n_points)

    f_cx = interpolate.interp1d(t_valid, cx_valid, kind="linear",
                                 bounds_error=False, fill_value=(cx_valid[0], cx_valid[-1]))
    f_cy = interpolate.interp1d(t_valid, cy_valid, kind="linear",
                                 bounds_error=False, fill_value=(cy_valid[0], cy_valid[-1]))

    return f_cx(t_out), f_cy(t_out), valid.mean()


def main():
    print(f"=== Trajectory Consistency Analysis ===")
    print(f"Episodes: {EPISODES}")
    print(f"Cameras: {CAMERAS}")
    print(f"Stride: {STRIDE}, Conf threshold: {CONF_THRESHOLD}")

    ds_path = resolve_dataset_path(DATASET)
    info, episodes_df = load_dataset_meta(ds_path)
    fps = info["fps"]

    cam_resolution = {}
    for cam in CAMERAS:
        feat = info["features"][f"observation.images.{cam}"]
        cam_resolution[cam] = (feat["info"]["video.width"], feat["info"]["video.height"])

    detector = create_detector(device="cuda", confidence_threshold=0.05)

    # ── Run detection ───────────────────────────────────────────────
    # all_data[ep_idx][cam] = {duck_cx, duck_cy, duck_conf, cup_cx, ...}
    all_data = {}
    ep_lengths = {}

    for ep_idx in EPISODES:
        t0 = time.perf_counter()
        print(f"\nProcessing episode {ep_idx}...")
        data, n_frames = run_detection_on_episode(
            detector, ds_path, info, episodes_df, ep_idx, CAMERAS, cam_resolution, fps
        )
        all_data[ep_idx] = data
        ep_lengths[ep_idx] = n_frames
        dt = time.perf_counter() - t0
        print(f"  {n_frames} frames, {dt:.1f}s")

    # ── Resample all trajectories to RESAMPLE_N points ──────────────
    # resampled[cam][obj] = {ep_idx: (cx, cy, valid_frac)}
    resampled = {cam: {obj: {} for obj in OBJECTS} for cam in CAMERAS}

    for ep_idx in EPISODES:
        for cam in CAMERAS:
            for obj in OBJECTS:
                d = all_data[ep_idx][cam]
                cx_r, cy_r, vf = resample_trajectory(
                    d[f"{obj}_cx"], d[f"{obj}_cy"], d[f"{obj}_conf"]
                )
                resampled[cam][obj][ep_idx] = (cx_r, cy_r, vf)

    # ── Save plot FIRST ─────────────────────────────────────────────
    print("\nGenerating plot...")
    colors = plt.cm.tab10(np.linspace(0, 1, len(EPISODES)))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Trajectory Consistency: duck-in-cup task across episodes", fontsize=14, fontweight="bold")

    for col_idx, cam in enumerate(CAMERAS):
        for row_idx, obj in enumerate(OBJECTS):
            ax = axes[row_idx, col_idx]

            # Collect all resampled trajectories
            all_cx = []
            all_cy = []

            for i, ep_idx in enumerate(EPISODES):
                cx_r, cy_r, vf = resampled[cam][obj][ep_idx]
                if vf < 0.1:
                    continue
                all_cx.append(cx_r)
                all_cy.append(cy_r)
                ax.plot(cx_r, cy_r, color=colors[i], alpha=0.5, linewidth=1.0,
                        label=f"ep {ep_idx}")
                # Mark start and end
                ax.scatter(cx_r[0], cy_r[0], color=colors[i], s=20, zorder=5, marker="o")
                ax.scatter(cx_r[-1], cy_r[-1], color=colors[i], s=20, zorder=5, marker="x")

            # Mean trajectory
            if len(all_cx) >= 2:
                mean_cx = np.nanmean(all_cx, axis=0)
                mean_cy = np.nanmean(all_cy, axis=0)
                std_cx = np.nanstd(all_cx, axis=0)
                std_cy = np.nanstd(all_cy, axis=0)

                ax.plot(mean_cx, mean_cy, color="black", linewidth=2.5, alpha=0.9,
                        label="mean", zorder=10)

                # +/- 1 std band (as an ellipse-like corridor)
                ax.fill_between(mean_cx, mean_cy - std_cy, mean_cy + std_cy,
                                alpha=0.15, color="gray", zorder=1)

            ax.set_title(f"{cam} / {obj}", fontsize=10, fontweight="bold")
            ax.set_xlabel("cx (norm)")
            ax.set_ylabel("cy (norm)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()  # image coords: y increases downward
            ax.set_aspect("equal")

            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=6, loc="upper right", ncol=2)

    plt.tight_layout()
    plot_path = OUT_DIR / "trajectory_consistency.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # ── Compute statistics ──────────────────────────────────────────
    print("\nComputing statistics...")
    trajectory_stats = {}

    for cam in CAMERAS:
        for obj in OBJECTS:
            key = f"{cam}_{obj}"
            all_cx = []
            all_cy = []
            valid_eps = []

            for ep_idx in EPISODES:
                cx_r, cy_r, vf = resampled[cam][obj][ep_idx]
                if vf < 0.1:
                    continue
                all_cx.append(cx_r)
                all_cy.append(cy_r)
                valid_eps.append(ep_idx)

            if len(all_cx) < 2:
                trajectory_stats[key] = {
                    "valid_episodes": len(all_cx),
                    "mean_trajectory_length": 0.0,
                    "std_endpoint_cx": 0.0,
                    "std_endpoint_cy": 0.0,
                    "mean_pairwise_corr_cx": 0.0,
                    "verdict": "insufficient_data",
                }
                continue

            all_cx = np.array(all_cx)
            all_cy = np.array(all_cy)

            # Mean trajectory length (sum of Euclidean steps)
            lengths = []
            for i in range(len(all_cx)):
                dx = np.diff(all_cx[i])
                dy = np.diff(all_cy[i])
                lengths.append(np.sum(np.sqrt(dx**2 + dy**2)))
            mean_length = float(np.mean(lengths))

            # Std of endpoints
            std_end_cx = float(np.std(all_cx[:, -1]))
            std_end_cy = float(np.std(all_cy[:, -1]))

            # Pairwise correlation of cx sequences
            n_eps = len(all_cx)
            corrs = []
            for i in range(n_eps):
                for j in range(i + 1, n_eps):
                    # Only correlate if both have variance
                    if np.std(all_cx[i]) > 1e-6 and np.std(all_cx[j]) > 1e-6:
                        r, _ = stats.pearsonr(all_cx[i], all_cx[j])
                        corrs.append(r)
            mean_corr = float(np.mean(corrs)) if corrs else 0.0

            verdict = "consistent" if mean_corr > 0.7 else "variable"

            trajectory_stats[key] = {
                "valid_episodes": len(valid_eps),
                "mean_trajectory_length": round(mean_length, 4),
                "std_endpoint_cx": round(std_end_cx, 4),
                "std_endpoint_cy": round(std_end_cy, 4),
                "mean_pairwise_corr_cx": round(mean_corr, 4),
                "verdict": verdict,
            }

    stats_path = OUT_DIR / "trajectory_stats.json"
    with open(stats_path, "w") as f:
        json.dump(trajectory_stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # ── Print summary ───────────────────────────────────────────────
    print("\n=== Summary ===")
    for key, st in trajectory_stats.items():
        print(f"  {key}: corr={st['mean_pairwise_corr_cx']:.3f} "
              f"length={st['mean_trajectory_length']:.4f} "
              f"end_std=({st['std_endpoint_cx']:.4f}, {st['std_endpoint_cy']:.4f}) "
              f"-> {st['verdict']}")


if __name__ == "__main__":
    main()
