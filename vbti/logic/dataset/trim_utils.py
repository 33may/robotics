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

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def _parse_joints(joints) -> list[int] | None:
    """Parse joint specification to column indices.

    Args:
        joints: None (all), list of ints, or comma-separated names/indices
                e.g. "shoulder_lift,elbow_flex" or "1,2"
    Returns:
        list of column indices, or None for all joints
    """
    if joints is None:
        return None
    if isinstance(joints, (list, tuple)):
        if all(isinstance(j, int) for j in joints):
            return list(joints)
        joints = ",".join(str(j) for j in joints)
    indices = []
    for part in str(joints).split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
        elif part in JOINT_NAMES:
            indices.append(JOINT_NAMES.index(part))
        else:
            # Try partial match
            matches = [i for i, n in enumerate(JOINT_NAMES) if part in n]
            if matches:
                indices.extend(matches)
            else:
                raise ValueError(f"Unknown joint: {part}. Available: {JOINT_NAMES}")
    return indices


def _detect_rest_frames(actions: np.ndarray, dist_thresh: float = 5.0,
                         joints: list[int] | None = None) -> tuple[int, int]:
    """Detect rest regions at start/end of an episode.

    Compares each frame to the mean of the first/last 5 frames.
    When max joint delta exceeds dist_thresh, that's where motion begins/ends.

    Args:
        actions: (N, n_joints) array
        dist_thresh: threshold in degrees
        joints: which joint indices to check (None = all, "any" mode).
                e.g. [1, 2] checks only shoulder_lift and elbow_flex

    Returns (trim_start, trim_end) — the active frame range.
    """
    n = len(actions)
    if n < 10:
        return 0, n

    # Select joints to check
    if joints is not None:
        check = actions[:, joints]
    else:
        check = actions

    start_pose = check[:5].mean(axis=0)
    end_pose = check[-5:].mean(axis=0)
    dist_from_start = np.abs(check - start_pose).max(axis=1)
    dist_from_end = np.abs(check - end_pose).max(axis=1)

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

def preview(dataset_path: str, dist_thresh: float = 5.0, n_eps: int = 6,
            joints: str = None, save: str = None):
    """Preview trimming — trajectories with shaded rest + before/after distributions.

    Args:
        dataset_path: path or repo_id
        dist_thresh: max joint delta to consider "at rest" (dataset units)
        n_eps: number of episodes to show in trajectory panel
        joints: which joints to check for rest detection (default: all).
                e.g. "shoulder_lift,elbow_flex" or "1,2"
        save: save to file instead of showing
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    joint_indices = _parse_joints(joints)

    resolved = _resolve_dataset_path(dataset_path)
    episodes = _load_episodes_from_parquet(resolved)
    n_total = len(episodes)
    show_n = min(n_eps, n_total)

    total_frames = 0
    kept_frames = 0
    trim_info = []
    all_original = []
    all_trimmed = []

    print(f"\n  Dataset: {resolved.name}")
    print(f"  Episodes: {n_total}  Threshold: {dist_thresh}")
    print(f"\n  {'Ep':>4}  {'Total':>6}  {'Start':>6}  {'End':>6}  {'Kept':>6}  {'Cut %':>6}")
    print(f"  {'─' * 42}")

    for ep_idx in range(n_total):
        actions = episodes[ep_idx]
        n = len(actions)
        start, end = _detect_rest_frames(actions, dist_thresh, joints=joint_indices)
        kept = end - start
        total_frames += n
        kept_frames += kept
        cut_pct = 100 * (n - kept) / n if n > 0 else 0
        trim_info.append((ep_idx, n, start, end, kept, cut_pct))
        all_original.append(actions)
        if end > start:
            all_trimmed.append(actions[start:end])

        if ep_idx < show_n:
            print(f"  {ep_idx:>4}  {n:>6}  {start:>6}  {end:>6}  {kept:>6}  {cut_pct:>5.1f}%")

    if n_total > show_n:
        print(f"  ... {n_total - show_n} more episodes")

    cut_total = 100 * (total_frames - kept_frames) / total_frames if total_frames > 0 else 0
    print(f"\n  Total: {kept_frames}/{total_frames} frames kept ({cut_total:.1f}% cut)")

    # ── Plot: trajectories (top) + distributions (bottom) ─────────
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548"]

    orig_all = np.concatenate(all_original, axis=0)
    trim_all = np.concatenate(all_trimmed, axis=0) if all_trimmed else orig_all
    n_joints = min(orig_all.shape[1], len(joint_names))

    fig = plt.figure(figsize=(18, 3 * show_n + 12))
    gs = gridspec.GridSpec(show_n + 2, 3, figure=fig, height_ratios=[3] * show_n + [0.3, 7])

    fig.suptitle(f"Trim Preview — {resolved.name} (thresh={dist_thresh})",
                 fontsize=14, fontweight="bold", y=0.995)

    # Trajectory rows (span full width)
    for row in range(show_n):
        ax = fig.add_subplot(gs[row, :])
        ep_idx, n, start, end, kept, cut_pct = trim_info[row]
        actions = episodes[ep_idx]

        if start > 0:
            ax.axvspan(0, start, alpha=0.15, color="red",
                       label="rest (cut)" if row == 0 else None)
        if end < n:
            ax.axvspan(end, n, alpha=0.15, color="red")

        for j in range(n_joints):
            ax.plot(actions[:, j], color=colors[j % len(colors)], alpha=0.7,
                    linewidth=0.8, label=joint_names[j] if row == 0 else None)

        ax.set_ylabel(f"ep {ep_idx}")
        ax.set_xlim(0, n)
        ax.text(0.01, 0.95, f"{n}f, active={start}:{end} ({cut_pct:.0f}% cut)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        if row == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=n_joints)
        if row == show_n - 1:
            ax.set_xlabel("frame")

    # Distribution row (2x3 grid for 6 joints)
    for j in range(n_joints):
        ax = fig.add_subplot(gs[show_n + 1, j % 3]) if j < 3 else None
        if j >= 3:
            # Need a second distribution row — reuse the grid
            break

    # Actually build 2x3 sub-grid for distributions
    dist_gs = gs[show_n + 1, :].subgridspec(2, 3, hspace=0.5, wspace=0.3)
    for j in range(n_joints):
        ax = fig.add_subplot(dist_gs[j // 3, j % 3])
        ax.hist(orig_all[:, j], bins=120, alpha=0.35, color="#999999",
                label="original", edgecolor="#999999", linewidth=0.3)
        ax.hist(trim_all[:, j], bins=120, alpha=0.55, color=colors[j],
                label="trimmed", edgecolor=colors[j], linewidth=0.3)
        orig_mean = orig_all[:, j].mean()
        trim_mean = trim_all[:, j].mean()
        ax.set_title(f"{joint_names[j]}  (μ {orig_mean:.0f}→{trim_mean:.0f})", fontsize=11)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("degrees", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        if j == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


# ── Trim ──────────────────────────────────────────────────────────────────────

def trim(repo_id: str, output_repo_id: str,
         root: str = None, output_root: str = None,
         dist_thresh: float = 5.0, min_active_frames: int = 20,
         push_to_hub: bool = False):
    """Create a new LeRobot dataset with rest frames removed.

    Args:
        repo_id: source dataset repo_id (e.g. eternalmay33/so101_real_pick_place_50eps)
        output_repo_id: repo_id for output (e.g. eternalmay33/so101_real_trimmed)
        root: path to source dataset on disk (default: standard lerobot cache)
        output_root: output directory (default: standard lerobot cache)
        dist_thresh: max joint delta to consider "at rest" (dataset units)
        min_active_frames: skip episodes with fewer active frames than this
        push_to_hub: push result to HuggingFace Hub after trimming
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import load_episodes

    if root:
        dataset_path = Path(root).expanduser().resolve()
    else:
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
    dataset_path = dataset_path.resolve()

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


# ── Plotting utilities ────────────────────────────────────────────────────────

def trajectories(dataset_path: str, episodes: list[int] = None, ep: str = None,
                  n_episodes: int = 6, save: str = None):
    """Plot joint trajectories for episodes, with rest regions shaded.

    Args:
        dataset_path: path or repo_id
        episodes: list of episode indices (e.g. [0, 5, 37])
        ep: comma-separated episode indices (e.g. "0,5,37" or "10-20" for range)
        n_episodes: how many episodes to show if none specified
        save: save to file instead of showing
    """
    import matplotlib.pyplot as plt
    from vbti.logic.dataset.check_utils import lerobot_info

    resolved = _resolve_dataset_path(dataset_path)
    all_episodes = _load_episodes_from_parquet(resolved)
    info_d = lerobot_info(str(resolved))
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    n_joints = all_episodes[0].shape[1] if all_episodes else 0
    if len(joint_names) < n_joints:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    # Parse episode selection: --ep="0,5,37" or --ep="10-20"
    if ep is not None:
        parsed = []
        raw = str(ep).strip("()[] ")
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                parsed.extend(range(int(a), int(b) + 1))
            else:
                parsed.append(int(part))
        episodes = parsed

    ep_indices = episodes if episodes is not None else list(range(min(n_episodes, len(all_episodes))))
    ep_indices = [i for i in ep_indices if i < len(all_episodes)]

    fig, axes = plt.subplots(len(ep_indices), 1, figsize=(16, 3 * len(ep_indices)), sharex=False)
    if len(ep_indices) == 1:
        axes = [axes]
    fig.suptitle(f"Joint Trajectories — {resolved.name}", fontsize=14, fontweight="bold")
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548"]

    for row, ep_i in enumerate(ep_indices):
        ax = axes[row]
        actions = all_episodes[ep_i]
        n_frames = len(actions)
        trim_start, trim_end = _detect_rest_frames(actions)
        if trim_start > 0:
            ax.axvspan(0, trim_start, alpha=0.15, color="red", label="rest" if row == 0 else None)
        if trim_end < n_frames:
            ax.axvspan(trim_end, n_frames, alpha=0.15, color="red")
        for j in range(n_joints):
            ax.plot(actions[:, j], color=colors[j % len(colors)], alpha=0.7,
                    linewidth=0.8, label=joint_names[j] if row == 0 else None)
        ax.set_ylabel(f"ep {ep_i}")
        ax.set_xlim(0, n_frames)
        ax.text(0.01, 0.95, f"{n_frames}f, active={trim_start}:{trim_end}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].legend(loc="upper right", fontsize=7, ncol=n_joints)
    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


def compare_trimmed(datasets: dict[str, str], dist_thresh: float = 5.0,
                    bins: int = 80, save: str = None):
    """Compare action distributions after trimming rest poses."""
    import matplotlib.pyplot as plt
    from vbti.logic.dataset.check_utils import lerobot_info

    loaded = {}
    for label, path in datasets.items():
        episodes = _load_episodes_from_parquet(_resolve_dataset_path(path))
        all_trimmed = []
        total_frames = 0
        trimmed_frames = 0
        for ep in episodes:
            total_frames += len(ep)
            start, end = _detect_rest_frames(ep, dist_thresh=dist_thresh)
            if end > start:
                all_trimmed.append(ep[start:end])
                trimmed_frames += end - start
        loaded[label] = np.concatenate(all_trimmed, axis=0)
        pct = 100 * trimmed_frames / total_frames if total_frames else 0
        print(f"  {label}: {trimmed_frames}/{total_frames} frames kept ({pct:.0f}%)")

    n_joints = next(iter(loaded.values())).shape[1]
    info_d = lerobot_info(str(_resolve_dataset_path(next(iter(datasets.values())))))
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    if len(joint_names) < n_joints:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Action Distributions (rest trimmed)", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n_joints:
            ax.set_visible(False)
            continue
        for j, (label, actions) in enumerate(loaded.items()):
            col = actions[:, i]
            color = colors[j % len(colors)]
            ax.hist(col, bins=bins, alpha=0.45, label=f"{label} (μ={col.mean():.1f})",
                    color=color, edgecolor=color, linewidth=0.5)
        ax.set_title(joint_names[i], fontsize=11)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


def _resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve a dataset path or repo_id to an absolute Path."""
    p = Path(dataset_path).expanduser()
    if p.exists():
        return p.resolve()
    # Try as repo_id in lerobot cache
    cache_root = Path.home() / ".cache/huggingface/lerobot"
    cache = cache_root / dataset_path
    if cache.exists():
        return cache.resolve()
    # Suggest similar repo_ids
    available = []
    if cache_root.exists():
        for author_dir in cache_root.iterdir():
            if author_dir.is_dir() and not author_dir.name.startswith("."):
                for ds_dir in author_dir.iterdir():
                    if ds_dir.is_dir():
                        available.append(f"{author_dir.name}/{ds_dir.name}")
    suggestions = [a for a in available if dataset_path.split("/")[-1][:5] in a]
    hint = f"\n  Did you mean: {', '.join(suggestions)}" if suggestions else f"\n  Available: {', '.join(available)}"
    raise FileNotFoundError(f"Dataset not found: {dataset_path}{hint}")


def distribution(dataset_path: str, bins: int = 80, save: str = None):
    """Plot action distribution for a single dataset.

    Args:
        dataset_path: path to dataset root OR repo_id (e.g. eternalmay33/so101_real_pick_place_50eps)
    """
    import matplotlib.pyplot as plt
    from vbti.logic.dataset.check_utils import lerobot_info

    resolved = _resolve_dataset_path(dataset_path)
    episodes = _load_episodes_from_parquet(resolved)
    actions = np.concatenate(episodes, axis=0)

    info_d = lerobot_info(str(resolved))
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    n_joints = actions.shape[1]
    if len(joint_names) < n_joints:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Action Distribution — {resolved.name}", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n_joints:
            ax.set_visible(False)
            continue
        col = actions[:, i]
        color = colors[i % len(colors)]
        ax.hist(col, bins=bins, alpha=0.7, color=color, edgecolor=color, linewidth=0.5)
        ax.set_title(f"{joint_names[i]}  (μ={col.mean():.1f}, σ={col.std():.1f})", fontsize=11)
        ax.set_xlabel("value (degrees)")
        ax.set_ylabel("count")
        ax.axvline(col.mean(), color="black", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


def drop_episodes(dataset_path: str, ep: str = None, min_std: float = None,
                   output: str = None, inplace: bool = False):
    """Remove episodes from a LeRobot dataset using LeRobot's native delete_episodes.

    Creates a new dataset with proper metadata (videos, parquet, stats, episodes).

    Args:
        dataset_path: path or repo_id
        ep: comma-separated episodes to drop (e.g. "37,4" or "10-20")
        min_std: also drop episodes where max joint std < this value (catches dead episodes)
        output: output repo_id (default: appends _filtered to original)
        inplace: if True, replaces the original dataset
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes
    import shutil

    resolved = _resolve_dataset_path(dataset_path)

    # Parse episodes to drop
    drop = set()
    if ep:
        raw = str(ep).strip("()[] ")
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                drop.update(range(int(a), int(b) + 1))
            else:
                drop.add(int(part))

    # Auto-detect dead episodes by min_std
    if min_std is not None:
        episodes_data = _load_episodes_from_parquet(resolved)
        for ep_idx, actions in enumerate(episodes_data):
            if actions.std(axis=0).max() < min_std:
                drop.add(ep_idx)
                print(f"  Auto-drop ep {ep_idx}: max joint std = {actions.std(axis=0).max():.3f}")

    if not drop:
        print("Nothing to drop.")
        return

    # Resolve repo_id from path
    # Convention: cache path is ~/.cache/huggingface/lerobot/{repo_id}
    cache_root = Path.home() / ".cache/huggingface/lerobot"
    try:
        repo_id = str(resolved.relative_to(cache_root))
    except ValueError:
        repo_id = dataset_path

    print(f"\n  Dataset: {repo_id}")
    print(f"  Dropping episodes: {sorted(drop)}")

    # Load dataset
    dataset = LeRobotDataset(repo_id, root=str(resolved))

    # Determine output
    if output:
        output_repo_id = output
    elif inplace:
        output_repo_id = repo_id + "_new"
    else:
        output_repo_id = repo_id + "_filtered"

    output_root = cache_root / output_repo_id

    # Use LeRobot's delete_episodes
    new_dataset = delete_episodes(
        dataset,
        episode_indices=sorted(drop),
        output_dir=output_root,
        repo_id=output_repo_id,
    )

    print(f"\n  Result: {new_dataset.meta.total_episodes} episodes, {new_dataset.meta.total_frames} frames")
    print(f"  Output: {output_root}")

    # If inplace, swap old and new
    if inplace:
        backup = resolved.with_name(resolved.name + "_backup")
        resolved.rename(backup)
        output_root.rename(resolved)
        shutil.rmtree(backup)
        print(f"  Replaced original dataset in-place.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "preview":         preview,
        "trim":            trim,
        "trajectories":    trajectories,
        "drop_episodes":   drop_episodes,
        "compare_trimmed": compare_trimmed,
        "distribution":    distribution,
    })
