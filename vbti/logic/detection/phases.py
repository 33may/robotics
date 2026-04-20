"""Task phase detection from gripper + joint trajectories.

Detects 5 phases of a pick-and-place manipulation task using a
velocity-based state machine approach:

    0 = reach     — arm moving toward object, gripper open
    1 = pregrasp  — fine positioning near object
    2 = grasp     — gripper actively closing
    3 = transport  — gripper closed, arm moving to drop location
    4 = release   — gripper opening at drop location

Usage:
    python -m vbti.logic.detection.phases eternalmay33/02_black_full_center
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PHASE_NAMES = ["reach", "pregrasp", "grasp", "transport", "release"]
MIN_PHASE_FRAMES = 3


# ------------------------------------------------------------------
# Smoothing helpers
# ------------------------------------------------------------------

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average smoothing with edge-padding."""
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _group_consecutive(indices: np.ndarray, gap: int = 3) -> list[np.ndarray]:
    """Group frame indices into contiguous events (allowing small gaps)."""
    if len(indices) == 0:
        return []
    groups = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= gap:
            current.append(indices[i])
        else:
            groups.append(np.array(current))
            current = [indices[i]]
    groups.append(np.array(current))
    return groups


# ------------------------------------------------------------------
# Core detection
# ------------------------------------------------------------------

def _find_grasp_region(grip_smooth: np.ndarray, grip_vel: np.ndarray, fps: int):
    """Find the main grasp region: the largest closing transition followed by
    a sustained closed period.

    Returns (grasp_start, grasp_end, closed_end) or None if not found.
    - grasp_start: first frame of closing transition
    - grasp_end: last frame of closing transition (gripper reaches closed)
    - closed_end: last frame where gripper stays closed before opening
    """
    T = len(grip_smooth)

    # Find all significant closing frames
    close_thresh = -0.3
    neg_frames = np.where(grip_vel < close_thresh)[0]
    if len(neg_frames) == 0:
        close_thresh = -0.15
        neg_frames = np.where(grip_vel < close_thresh)[0]
        if len(neg_frames) == 0:
            return None

    # Group into closing events (allow gaps up to fps/3 ~10 frames)
    closing_events = _group_consecutive(neg_frames, gap=max(fps // 3, 5))

    # Score each event by total displacement
    events_scored = []
    for event in closing_events:
        start, end = event[0], event[-1]
        displacement = grip_smooth[start] - grip_smooth[end]  # positive = closing
        events_scored.append((event, displacement))

    if not events_scored:
        return None

    # Sort by displacement (biggest closing first)
    events_scored.sort(key=lambda x: x[1], reverse=True)
    best_event, best_disp = events_scored[0]

    if best_disp < 1.0:
        return None

    grasp_start = int(best_event[0])
    grasp_end = int(best_event[-1])

    # The closed level is the gripper value right after the grasp.
    # Find the minimum gripper value in the region after grasp_end to account
    # for the gripper still settling.
    settle_end = min(grasp_end + fps, T)
    closed_level = np.min(grip_smooth[grasp_end:settle_end])
    # Tight tolerance: small noise margin (3 degrees or 15% of displacement)
    closed_tolerance = min(max(best_disp * 0.15, 3.0), 8.0)

    # Find the end of the sustained closed region.
    # The gripper is "closed" while it stays below closed_level + tolerance.
    # To handle grasp retries (close, open briefly, close again), we look
    # for closing events after the main one. If there's another significant
    # closing within fps*2 frames that returns to a closed level, include it.
    closed_end = grasp_end
    i = grasp_end + 1
    while i < T:
        if grip_smooth[i] <= closed_level + closed_tolerance:
            closed_end = i
            i += 1
        else:
            # Gripper went above tolerance. Check if there's another closing
            # event soon (grasp retry). Look for a subsequent closing that
            # returns to a closed-ish level and stays there.
            retry_window = min(i + fps * 3, T)
            found_retry = False
            # The retry might close to a slightly different level, so use
            # a more generous threshold: below the midpoint between
            # closed_level and the open level before the main grasp.
            open_level = grip_smooth[grasp_start]
            retry_closed_thresh = closed_level + (open_level - closed_level) * 0.35
            for j in range(i, retry_window):
                if grip_smooth[j] <= retry_closed_thresh:
                    # Check it stays closed for at least 0.5s
                    stay_count = 0
                    for k in range(j, min(j + fps, T)):
                        if grip_smooth[k] <= retry_closed_thresh:
                            stay_count += 1
                    if stay_count >= fps // 2:
                        # Update closed_level to the new level
                        new_settle = min(j + fps // 2, T)
                        closed_level = np.min(grip_smooth[j:new_settle])
                        closed_tolerance = min(max(
                            (open_level - closed_level) * 0.15, 3.0), 8.0)
                        closed_end = j
                        i = j + 1
                        found_retry = True
                        break
            if not found_retry:
                break

    # Also extend grasp_start backwards if there were earlier closing movements
    # that are part of the approach (gripper pre-closing slightly)
    pre_level = grip_smooth[grasp_start]
    for i in range(grasp_start - 1, max(grasp_start - fps, -1), -1):
        if grip_vel[i] < close_thresh * 0.5:
            grasp_start = i
        else:
            break

    return grasp_start, grasp_end, closed_end


def detect_phases(
    gripper: np.ndarray,
    joint_positions: np.ndarray,
    fps: int = 30,
    smooth_window: int = 5,
) -> np.ndarray:
    """Detect task phases from gripper + joint trajectories.

    Args:
        gripper: 1D array of gripper positions (degrees).
                 Higher = more open, lower = more closed.
        joint_positions: 2D array (T, 5) of arm joint positions (no gripper).
        fps: Recording frame rate.
        smooth_window: Window size for moving average smoothing.

    Returns:
        1D int array of phase indices (0-4), same length as input.
        Returns all -1 if detection fails.
    """
    T = len(gripper)
    if T < 10:
        return np.full(T, -1, dtype=np.int32)

    labels = np.full(T, -1, dtype=np.int32)

    # --- Step 1: smooth gripper and compute velocity ---
    grip_smooth = _smooth(gripper, smooth_window)
    grip_vel = np.gradient(grip_smooth)

    # --- Step 2: find grasp region ---
    result = _find_grasp_region(grip_smooth, grip_vel, fps)
    if result is None:
        return np.full(T, -1, dtype=np.int32)

    grasp_start, grasp_end, closed_end = result

    # Ensure minimum grasp duration
    if grasp_end - grasp_start < MIN_PHASE_FRAMES:
        grasp_end = min(grasp_start + MIN_PHASE_FRAMES, T - 1)
    if closed_end < grasp_end:
        closed_end = grasp_end

    # --- Step 3: find RELEASE event ---
    # Release starts when gripper begins sustained opening after the closed region.
    # The gripper must reach substantially above the closed level.
    closed_level = grip_smooth[closed_end]
    grasp_displacement = grip_smooth[grasp_start] - closed_level
    release_thresh = closed_level + max(grasp_displacement * 0.3, 3.0)

    release_start = None
    for i in range(closed_end + 1, T):
        if grip_smooth[i] >= release_thresh:
            # Walk back to find start of the opening
            release_start = i
            for j in range(i - 1, closed_end, -1):
                if grip_vel[j] <= 0.05:
                    release_start = j + 1
                    break
            else:
                release_start = closed_end + 1
            break

    # --- Step 4: separate REACH from PREGRASP ---
    if joint_positions.shape[0] != T:
        return np.full(T, -1, dtype=np.int32)

    joint_vel = np.gradient(joint_positions, axis=0)  # (T, 5)
    arm_speed = np.linalg.norm(joint_vel, axis=1)     # (T,)
    arm_speed_smooth = _smooth(arm_speed, smooth_window * 2)

    pre_grasp_region = arm_speed_smooth[:grasp_start]
    if len(pre_grasp_region) < MIN_PHASE_FRAMES * 2:
        reach_end = max(grasp_start // 2, MIN_PHASE_FRAMES)
    else:
        peak_speed = np.max(pre_grasp_region)
        if peak_speed < 0.1:
            reach_end = grasp_start // 2
        else:
            threshold = peak_speed * 0.30
            peak_idx = np.argmax(pre_grasp_region)
            reach_end = grasp_start  # default

            # Look for sustained low-speed period after the peak
            below = pre_grasp_region[peak_idx:] < threshold
            if np.any(below):
                min_sustained = max(fps // 6, 3)
                count = 0
                for i in range(len(below)):
                    if below[i]:
                        count += 1
                        if count >= min_sustained:
                            reach_end = peak_idx + i - count + 1
                            break
                    else:
                        count = 0

            # Fallback: cumulative displacement
            if reach_end >= grasp_start - MIN_PHASE_FRAMES:
                cum_disp = np.cumsum(arm_speed_smooth[:grasp_start])
                total_disp = cum_disp[-1] if len(cum_disp) > 0 else 1.0
                if total_disp > 0:
                    frac_idx = np.searchsorted(cum_disp, total_disp * 0.70)
                    reach_end = int(frac_idx)

    # Clamp reach_end
    reach_end = max(MIN_PHASE_FRAMES, min(reach_end, grasp_start - MIN_PHASE_FRAMES))

    # --- Step 5: assign labels ---
    labels[:reach_end] = 0                                  # reach
    labels[reach_end:grasp_start] = 1                       # pregrasp
    labels[grasp_start:grasp_end + 1] = 2                   # grasp
    labels[grasp_end + 1:closed_end + 1] = 2                # extended grasp (retries)

    # But we want grasp to be just the closing transition, and transport
    # to be the sustained closed period. Redefine:
    # grasp = grasp_start to grasp_end (the main closing)
    # If there were retries, include them in grasp too
    labels[grasp_start:closed_end + 1] = 2                  # entire grasp region

    # Actually, separate: the initial closing is grasp, the sustained closed
    # period is transport. But retries (close-open-close) should stay as grasp.
    # Use a simpler rule: grasp = grasp_start to closed_end if closed_end is
    # close to grasp_end. Otherwise, grasp = closing, rest = transport.
    # Threshold: if closed_end - grasp_end < fps (1 second), it's all grasp.
    if closed_end - grasp_end < fps:
        # All part of the grasp
        labels[grasp_start:closed_end + 1] = 2
        transport_start = closed_end + 1
    else:
        # Grasp is the closing transition + a small settle window
        settle = min(fps // 3, closed_end - grasp_end)
        grasp_region_end = grasp_end + settle
        labels[grasp_start:grasp_region_end + 1] = 2        # grasp
        labels[grasp_region_end + 1:closed_end + 1] = 3     # transport
        transport_start = grasp_region_end + 1

    if release_start is not None and release_start > transport_start:
        # Fill transport from end of grasp/extended-grasp to release
        labels[transport_start:release_start] = 3            # transport
        labels[release_start:] = 4                           # release
    else:
        labels[transport_start:] = 3                         # no release

    # --- Step 6: enforce minimum phase lengths ---
    for phase_id in range(5):
        mask = labels == phase_id
        if mask.any() and mask.sum() < MIN_PHASE_FRAMES:
            phase_frames = np.where(mask)[0]
            if phase_id > 0:
                labels[phase_frames] = phase_id - 1
            elif phase_id < 4:
                labels[phase_frames] = phase_id + 1

    # Fill any remaining -1 gaps
    if np.any(labels == -1):
        for i in range(1, T):
            if labels[i] == -1 and labels[i - 1] != -1:
                labels[i] = labels[i - 1]
        for i in range(T - 2, -1, -1):
            if labels[i] == -1 and labels[i + 1] != -1:
                labels[i] = labels[i + 1]

    return labels


def detect_phases_episode(state: np.ndarray, fps: int = 30) -> np.ndarray:
    """Convenience wrapper: takes full state array (T, 6) and splits gripper/joints.

    Args:
        state: 2D array (T, 6) — joints 0-4 + gripper at index 5.
        fps: Recording frame rate.

    Returns:
        1D int array of phase indices (0-4), same length as input.
    """
    if state.ndim != 2 or state.shape[1] < 6:
        return np.full(len(state), -1, dtype=np.int32)

    gripper = state[:, 5]
    joint_positions = state[:, :5]
    return detect_phases(gripper, joint_positions, fps=fps)


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------

def process_phases_dataset(
    dataset_path: str,
    fps: int = 30,
    root: str = None,
) -> Path:
    """Run phase detection on all episodes in a LeRobot dataset.

    Reads observation.state from parquet data files, runs detect_phases_episode
    on each episode, and saves results to {dataset_path}/phase_labels.parquet.

    Args:
        dataset_path: Repo ID or filesystem path.
        fps: Recording FPS (default 30).
        root: Override dataset root path.

    Returns:
        Path to the output parquet file.
    """
    from vbti.logic.dataset import resolve_dataset_path

    ds_path = resolve_dataset_path(dataset_path, root=root)
    output_path = ds_path / "phase_labels.parquet"

    print(f"[phases] Dataset: {ds_path}")
    print(f"[phases] Output:  {output_path}")

    # Load info for FPS
    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    dataset_fps = info.get("fps", fps)

    # Load all data files
    data_dir = ds_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files in {data_dir}")

    print(f"[phases] Loading {len(parquet_files)} data files...")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    episode_ids = sorted(df["episode_index"].unique())
    print(f"[phases] {len(episode_ids)} episodes, {len(df)} total frames")

    t0 = time.time()
    rows = []
    success = 0
    fail = 0

    for ep_idx in episode_ids:
        ep_data = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep_data["observation.state"].values)
        phases = detect_phases_episode(states, fps=dataset_fps)

        failed = np.all(phases == -1)
        if failed:
            fail += 1
        else:
            success += 1

        for i, (fi, ph) in enumerate(zip(ep_data["frame_index"].values, phases)):
            rows.append({
                "frame_index": int(fi),
                "episode_index": int(ep_idx),
                "phase": int(ph),
                "phase_name": PHASE_NAMES[ph] if ph >= 0 else "unknown",
            })

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    print(f"[phases] Done in {elapsed:.1f}s — {success} ok, {fail} failed")
    print(f"[phases] Saved {len(result_df)} rows to {output_path}")

    return output_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect task phases in a LeRobot dataset.")
    parser.add_argument("dataset", help="Dataset repo ID or path")
    parser.add_argument("--fps", type=int, default=30, help="FPS (default: 30)")
    parser.add_argument("--root", default=None, help="Override dataset root")
    args = parser.parse_args()

    process_phases_dataset(args.dataset, fps=args.fps, root=args.root)


if __name__ == "__main__":
    main()
