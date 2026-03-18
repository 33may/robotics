"""
Trim rest frames from LeRobot datasets.

Real robot datasets often have long idle segments at the start and end of
each episode (robot sitting in rest position). This biases training toward
"do nothing" behavior. This module creates a new dataset with only the
active (moving) portions of each episode.

Usage:
    # Preview what would be trimmed (fast — reads parquet only, shows plot)
    python vbti/logic/dataset/trim_utils.py preview \
        --dataset_path=~/.cache/huggingface/lerobot/eternalmay33/so101_real_pick_place_50eps

    # Create trimmed dataset
    python vbti/logic/dataset/trim_utils.py trim \
        --dataset_path=~/.cache/huggingface/lerobot/eternalmay33/so101_real_pick_place_50eps

    # Custom threshold (higher = more aggressive trim)
    python vbti/logic/dataset/trim_utils.py trim \
        --dataset_path=~/.cache/huggingface/lerobot/eternalmay33/so101_real_pick_place_50eps \
        --dist_thresh=10.0
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Rest detection ────────────────────────────────────────────────────────────

def _detect_rest_frames(actions: np.ndarray, dist_thresh: float = 5.0) -> tuple[int, int]:
    """Detect rest regions at start/end of an episode.

    Compares each frame to the mean of the first/last 5 frames.
    When max joint delta exceeds dist_thresh, that's where motion begins/ends.

    Returns (trim_start, trim_end) — the active frame range.
    """
    n = len(actions)
    if n < 10:
        return 0, n

    start_pose = actions[:5].mean(axis=0)
    end_pose = actions[-5:].mean(axis=0)
    dist_from_start = np.abs(actions - start_pose).max(axis=1)
    dist_from_end = np.abs(actions - end_pose).max(axis=1)

    trim_start = 0
    for i in range(n):
        if dist_from_start[i] > dist_thresh:
            trim_start = max(0, i - 2)
            break

    trim_end = n
    for i in range(n - 1, -1, -1):
        if dist_from_end[i] > dist_thresh:
            trim_end = min(n, i + 3)
            break

    return trim_start, trim_end


# ── Parquet reading (fast, no video) ─────────────────────────────────────────

def _load_episodes_from_parquet(dataset_path: Path) -> list[np.ndarray]:
    """Load actions grouped by episode from parquet files."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files in {data_dir}")

    episodes = {}
    for pq in tqdm(parquet_files, desc="  Reading parquet", leave=False):
        df = pd.read_parquet(pq)
        if "action" not in df.columns:
            continue
        for ep_idx, group in df.groupby("episode_index"):
            group = group.sort_values("frame_index")
            arr = np.stack(group["action"].values)
            if ep_idx in episodes:
                episodes[ep_idx] = np.concatenate([episodes[ep_idx], arr])
            else:
                episodes[ep_idx] = arr

    print(f"  Loaded {len(episodes)} episodes")
    return [episodes[k] for k in sorted(episodes.keys())]


# ── Preview ───────────────────────────────────────────────────────────────────

def preview(dataset_path: str, dist_thresh: float = 5.0, n_eps: int = 10):
    """Preview trimming — table + trajectory plot with shaded rest regions.

    Args:
        dataset_path: path to LeRobot dataset root
        dist_thresh: max joint delta to consider "at rest" (dataset units)
        n_eps: number of episodes to show in table and plot
    """
    dataset_path = Path(dataset_path).expanduser().resolve()
    episodes = _load_episodes_from_parquet(dataset_path)
    n_total = len(episodes)
    show_n = min(n_eps, n_total)

    total_frames = 0
    kept_frames = 0
    trim_info = []

    print(f"\n  Dataset: {dataset_path.name}")
    print(f"  Episodes: {n_total}  Threshold: {dist_thresh}")
    print(f"\n  {'Ep':>4}  {'Total':>6}  {'Start':>6}  {'End':>6}  {'Kept':>6}  {'Cut %':>6}")
    print(f"  {'─' * 42}")

    for ep_idx in range(n_total):
        actions = episodes[ep_idx]
        n = len(actions)
        start, end = _detect_rest_frames(actions, dist_thresh)
        kept = end - start
        total_frames += n
        kept_frames += kept
        cut_pct = 100 * (n - kept) / n if n > 0 else 0
        trim_info.append((ep_idx, n, start, end, kept, cut_pct))

        if ep_idx < show_n:
            print(f"  {ep_idx:>4}  {n:>6}  {start:>6}  {end:>6}  {kept:>6}  {cut_pct:>5.1f}%")

    if n_total > show_n:
        print(f"  ... {n_total - show_n} more episodes")

    cut_total = 100 * (total_frames - kept_frames) / total_frames if total_frames > 0 else 0
    print(f"\n  Total: {kept_frames}/{total_frames} frames kept ({cut_total:.1f}% cut)")

    # Plot
    _plot_trim_preview(episodes, trim_info, show_n, dist_thresh, dataset_path)


def _plot_trim_preview(episodes, trim_info, n_eps, dist_thresh, dataset_path):
    """Plot trajectories with rest regions shaded red."""
    import matplotlib.pyplot as plt

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548"]

    fig, axes = plt.subplots(n_eps, 1, figsize=(16, 3 * n_eps), sharex=False)
    if n_eps == 1:
        axes = [axes]
    fig.suptitle(f"Trim Preview — {dataset_path.name} (thresh={dist_thresh})",
                 fontsize=14, fontweight="bold")

    for row in range(n_eps):
        ep_idx, n, start, end, kept, cut_pct = trim_info[row]
        actions = episodes[ep_idx]
        ax = axes[row]

        if start > 0:
            ax.axvspan(0, start, alpha=0.15, color="red",
                       label="rest (cut)" if row == 0 else None)
        if end < n:
            ax.axvspan(end, n, alpha=0.15, color="red")

        n_joints = min(actions.shape[1], len(joint_names))
        for j in range(n_joints):
            ax.plot(actions[:, j], color=colors[j % len(colors)], alpha=0.7,
                    linewidth=0.8, label=joint_names[j] if row == 0 else None)

        ax.set_ylabel(f"ep {ep_idx}")
        ax.set_xlim(0, n)
        ax.text(0.01, 0.95, f"{n}f, active={start}:{end} ({cut_pct:.0f}% cut)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].legend(loc="upper right", fontsize=7, ncol=len(joint_names))
    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    plt.show()


# ── Trim ──────────────────────────────────────────────────────────────────────

def trim(dataset_path: str, repo_id: str, output_repo_id: str,
         output_root: str = None, dist_thresh: float = 5.0,
         min_active_frames: int = 20, push_to_hub: bool = False):
    """Create a new LeRobot dataset with rest frames removed.

    Args:
        dataset_path: path to source LeRobot dataset on disk
        repo_id: source dataset repo_id (e.g. eternalmay33/so101_real_pick_place_50eps)
        output_repo_id: repo_id for output (e.g. eternalmay33/so101_real_trimmed)
        output_root: output directory (default: standard lerobot cache)
        dist_thresh: max joint delta to consider "at rest" (dataset units)
        min_active_frames: skip episodes with fewer active frames than this
        push_to_hub: push result to HuggingFace Hub after trimming
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import load_episodes

    dataset_path = Path(dataset_path).expanduser().resolve()

    # ── Step 1: Read metadata ─────────────────────────────────
    print(f"\n[1/4] Reading metadata...")
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    fps = info.get("fps", 30)
    robot_type = info.get("robot_type", "so101_follower")

    episodes_meta = load_episodes(dataset_path)
    print(f"  Source: {repo_id} ({len(episodes_meta)} episodes, {info.get('total_frames', '?')} frames)")
    print(f"  Output: {output_repo_id}")

    # ── Step 2: Detect rest frames from parquet (fast) ────────
    print(f"\n[2/4] Detecting rest frames (threshold={dist_thresh})...")
    episodes_actions = _load_episodes_from_parquet(dataset_path)

    trim_plan = []
    total_frames = 0
    total_active = 0
    skipped = 0

    for ep_idx, ep_actions in enumerate(episodes_actions):
        n = len(ep_actions)
        total_frames += n
        start, end = _detect_rest_frames(ep_actions, dist_thresh)
        active = end - start

        if active < min_active_frames:
            skipped += 1
            continue

        # Get global frame range from episode metadata
        ep_meta = episodes_meta[ep_idx]
        global_from = ep_meta["dataset_from_index"]
        trim_plan.append((ep_idx, start, end, n, active, global_from))
        total_active += active

    cut_pct = 100 * (total_frames - total_active) / total_frames if total_frames else 0
    print(f"  Episodes: {len(trim_plan)} keep, {skipped} skip")
    print(f"  Frames: {total_active}/{total_frames} to copy ({cut_pct:.1f}% cut)")

    # ── Step 3: Create output dataset ─────────────────────────
    print(f"\n[3/4] Creating output dataset...")
    source = LeRobotDataset(repo_id, root=str(dataset_path))
    meta = LeRobotDatasetMetadata(repo_id, root=str(dataset_path))

    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=meta.features,
        root=output_root,
    )
    print(f"  Output: {output_repo_id} → {output.root}")

    # ── Step 4: Copy active frames ────────────────────────────
    print(f"\n[4/4] Copying frames (this decodes video, may take a while)...")
    kept_frames = 0
    saved_eps = 0

    # Identify which feature keys add_frame expects (exclude auto-generated ones)
    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    feature_keys = [k for k in meta.features if k not in auto_keys]

    pbar = tqdm(trim_plan, desc="  Episodes")
    for ep_idx, start, end, n, active, global_from in pbar:
        for offset in range(start, end):
            global_idx = global_from + offset
            item = source[global_idx]

            # Build frame dict for add_frame
            # __getitem__ already adds "task" as a string
            frame_dict = {"task": item["task"]}
            for key in feature_keys:
                if key in item:
                    val = item[key]
                    if hasattr(val, "numpy"):
                        val = val.numpy()
                    # __getitem__ returns images as (C,H,W), add_frame expects (H,W,C)
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[0] in (1, 3):
                        val = val.transpose(1, 2, 0)
                    frame_dict[key] = val

            output.add_frame(frame_dict)

        output.save_episode()
        kept_frames += active
        saved_eps += 1
        pbar.set_postfix(ep=f"{saved_eps}/{len(trim_plan)}", frames=kept_frames)

    # Finalize
    output.finalize()

    final_cut = 100 * (total_frames - kept_frames) / total_frames if total_frames else 0
    print(f"\n{'=' * 50}")
    print(f"TRIM COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Episodes: {saved_eps} saved, {skipped} skipped")
    print(f"  Frames:   {kept_frames}/{total_frames} kept ({final_cut:.1f}% cut)")
    print(f"  Output:   {output.root}")

    if push_to_hub:
        output.push_to_hub()
        print(f"  Pushed: {output_repo_id}")

    return str(output.root)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "preview": preview,
        "trim":    trim,
    })
