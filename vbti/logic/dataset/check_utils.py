#!/usr/bin/env python3
"""LeRobot dataset inspection, comparison, and management utilities.

Usage:
    python -m vbti.logic.dataset.check_utils ls
    python -m vbti.logic.dataset.check_utils info may33/duck_cup --root=/path
    python -m vbti.logic.dataset.check_utils cameras may33/duck_cup --root=/path
    python -m vbti.logic.dataset.check_utils report /path/to/lerobot/dataset
    python -m vbti.logic.dataset.check_utils compare_actions --datasets real=/p1 sim=/p2
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


# ═══════════════════════════════════════════════════════════════════
# Dataset registry — list, find, info
# ═══════════════════════════════════════════════════════════════════


def ls():
    """List all locally available LeRobot datasets with basic info."""
    if not LEROBOT_CACHE.exists():
        print("  No cached datasets found")
        return

    datasets = []
    for author_dir in sorted(LEROBOT_CACHE.iterdir()):
        if not author_dir.is_dir() or author_dir.name.startswith("."):
            continue
        for ds_dir in sorted(author_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            # Resolve symlinks
            real_path = ds_dir.resolve()
            info_path = real_path / "meta" / "info.json"
            if not info_path.exists():
                continue
            try:
                with open(info_path) as f:
                    info = json.load(f)
                repo_id = f"{author_dir.name}/{ds_dir.name}"
                is_link = ds_dir.is_symlink()
                datasets.append({
                    "repo_id": repo_id,
                    "episodes": info.get("total_episodes", "?"),
                    "frames": info.get("total_frames", "?"),
                    "fps": info.get("fps", "?"),
                    "robot": info.get("robot_type", "?"),
                    "path": str(real_path),
                    "linked": is_link,
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not datasets:
        print("  No valid LeRobot datasets found")
        return

    print(f"\n  LeRobot Datasets ({len(datasets)}):\n")
    for ds in datasets:
        link_marker = " → " + ds["path"] if ds["linked"] else ""
        print(f"    {ds['repo_id']:40s}  {ds['episodes']:>4} eps  {ds['frames']:>6} frames  {ds['fps']:>2} fps  {ds['robot']}{link_marker}")
    print()


def info(repo_id: str, root: str = None):
    """Show detailed info for a LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    kwargs = {"repo_id": repo_id}
    if root:
        kwargs["root"] = root
    meta = LeRobotDatasetMetadata(**kwargs)

    print(f"\n  Dataset: {repo_id}")
    print(f"  Episodes: {meta.total_episodes}  Frames: {meta.total_frames}  FPS: {meta.fps}")

    # Features
    print(f"\n  Features:")
    for key, feat in meta.features.items():
        dtype = feat.get("dtype", "?")
        shape = feat.get("shape", "?")
        names = feat.get("names", [])
        extra = f"  names={names}" if names and dtype != "video" else ""
        print(f"    {key:40s}  dtype={dtype:8s}  shape={str(shape):20s}{extra}")

    # Stats
    if hasattr(meta, "stats") and meta.stats:
        print(f"\n  Stats:")
        for key in ["action", "observation.state"]:
            if key in meta.stats:
                s = meta.stats[key]
                print(f"    {key}:")
                for stat_name in ["min", "max", "mean", "std"]:
                    vals = s.get(stat_name, [])
                    if vals:
                        print(f"      {stat_name:5s}: {[round(v, 4) for v in vals]}")
    print()


def cameras(repo_id: str, root: str = None):
    """List cameras in a LeRobot dataset with shape and ordering info."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    kwargs = {"repo_id": repo_id}
    if root:
        kwargs["root"] = root
    meta = LeRobotDatasetMetadata(**kwargs)
    cam_features = {
        k: v for k, v in meta.features.items()
        if k.startswith("observation.images.")
    }
    if not cam_features:
        print(f"  No cameras found in {repo_id}")
        return

    print(f"\n  Dataset: {repo_id}")
    print(f"  Episodes: {meta.total_episodes}  Frames: {meta.total_frames}  FPS: {meta.fps}")
    print(f"  Cameras ({len(cam_features)}):\n")
    for i, (key, feat) in enumerate(cam_features.items()):
        name = key.replace("observation.images.", "")
        shape = feat.get("shape", "?")
        codec = feat.get("video_info", {}).get("video.codec", "?")
        print(f"    [{i}] {name:20s}  {str(shape):20s}  codec={codec}")
    print()


# ═══════════════════════════════════════════════════════════════════
# LeRobot inspection functions
# ═══════════════════════════════════════════════════════════════════


def lerobot_info(dataset_path: str | Path) -> dict:
    """Load and return info.json as dict."""
    with open(Path(dataset_path) / "meta" / "info.json") as f:
        return json.load(f)


def lerobot_stats(dataset_path: str | Path) -> dict:
    """Load and return stats.json as dict."""
    with open(Path(dataset_path) / "meta" / "stats.json") as f:
        return json.load(f)


def lerobot_features(dataset_path: str | Path) -> dict:
    return lerobot_info(dataset_path).get("features", {})


def lerobot_parquet_schema(dataset_path: str | Path) -> dict:
    """Read first data parquet and return column dtypes and shapes."""
    df = pd.read_parquet(Path(dataset_path) / "data" / "chunk-000" / "file-000.parquet")
    result = {}
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            arr = np.array(sample)
            result[col] = {"parquet_type": type(sample).__name__, "element_dtype": str(arr.dtype), "shape": arr.shape}
        else:
            result[col] = {"parquet_type": type(sample).__name__, "value_dtype": str(type(sample).__name__), "sample_value": sample}
    return result


def lerobot_action_stats(dataset_path: str | Path, max_files: int = 99) -> dict:
    """Compute detailed action statistics across all parquet files."""
    data_dir = Path(dataset_path) / "data"
    all_actions = []
    for pq in sorted(data_dir.rglob("*.parquet"))[:max_files]:
        df = pd.read_parquet(pq)
        if "action" in df.columns:
            all_actions.append(np.stack(df["action"].values))
    if not all_actions:
        return {"error": "no action column found"}

    actions = np.concatenate(all_actions, axis=0)
    info_d = lerobot_info(dataset_path)
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    stats = {"total_frames": len(actions), "num_joints": actions.shape[1], "dtype": str(actions.dtype), "joints": {}}
    for i in range(actions.shape[1]):
        col = actions[:, i]
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        stats["joints"][name] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std()), "abs_max": float(np.abs(col).max())}

    abs_max = np.abs(actions).max()
    stats["likely_unit"] = _detect_unit(abs_max)
    return stats


def lerobot_state_stats(dataset_path: str | Path, max_files: int = 99) -> dict:
    """Compute detailed observation.state statistics across all parquet files."""
    data_dir = Path(dataset_path) / "data"
    all_states = []
    for pq in sorted(data_dir.rglob("*.parquet"))[:max_files]:
        df = pd.read_parquet(pq)
        if "observation.state" in df.columns:
            all_states.append(np.stack(df["observation.state"].values))
    if not all_states:
        return {"error": "no observation.state column found"}

    states = np.concatenate(all_states, axis=0)
    info_d = lerobot_info(dataset_path)
    joint_names = info_d.get("features", {}).get("observation.state", {}).get("names", [])
    stats = {"total_frames": len(states), "num_joints": states.shape[1], "dtype": str(states.dtype), "joints": {}}
    for i in range(states.shape[1]):
        col = states[:, i]
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        stats["joints"][name] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std())}

    abs_max = np.abs(states).max()
    stats["likely_unit"] = _detect_unit(abs_max)
    return stats


def lerobot_episode_lengths(dataset_path: str | Path) -> list[dict]:
    ep_dir = Path(dataset_path) / "meta" / "episodes"
    if not ep_dir.exists():
        return []
    frames = []
    for pq in sorted(ep_dir.rglob("*.parquet")):
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            frames.append(row.to_dict())
    return frames


def lerobot_video_info(dataset_path: str | Path) -> dict:
    videos_dir = Path(dataset_path) / "videos"
    if not videos_dir.exists():
        return {}
    result = {}
    for cam_dir in sorted(videos_dir.iterdir()):
        if cam_dir.is_dir():
            files = list(cam_dir.rglob("*.mp4"))
            total_bytes = sum(f.stat().st_size for f in files)
            result[cam_dir.name] = {"num_files": len(files), "total_size_mb": round(total_bytes / 1e6, 1)}
    return result


def lerobot_action_state_correspondence(dataset_path: str | Path, episode_idx: int = 0) -> dict:
    """Check if action[t] ≈ state[t+1] - state[t] (delta) or action[t] ≈ state[t] (absolute)."""
    data_dir = Path(dataset_path) / "data"
    pq = sorted(data_dir.rglob("*.parquet"))[0]
    df = pd.read_parquet(pq)
    ep = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    if len(ep) < 10:
        return {"error": "episode too short"}

    actions = np.stack(ep["action"].values)
    states = np.stack(ep["observation.state"].values)
    abs_diff = np.abs(actions[:-1] - states[:-1]).mean(axis=0)
    state_deltas = states[1:] - states[:-1]
    delta_diff = np.abs(actions[:-1] - state_deltas).mean(axis=0)
    next_diff = np.abs(actions[:-1] - states[1:]).mean(axis=0)

    return {
        "action_vs_state_same_t": {"mean_abs_diff_per_joint": abs_diff.tolist(), "total_mean_diff": float(abs_diff.mean())},
        "action_vs_state_delta": {"mean_abs_diff_per_joint": delta_diff.tolist(), "total_mean_diff": float(delta_diff.mean())},
        "action_vs_next_state": {"mean_abs_diff_per_joint": next_diff.tolist(), "total_mean_diff": float(next_diff.mean())},
        "interpretation": (
            "absolute (action ≈ state)" if abs_diff.mean() < delta_diff.mean() and abs_diff.mean() < next_diff.mean()
            else "next-state (action ≈ state[t+1])" if next_diff.mean() < delta_diff.mean()
            else "delta (action ≈ state[t+1] - state[t])"
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════


def report(dataset_path: str | Path) -> str:
    """Full inspection report for a LeRobot dataset."""
    p = Path(dataset_path)
    lines = [f"{'='*70}", f"LEROBOT DATASET REPORT: {p.name}", f"Path: {p}", f"{'='*70}"]

    info_d = lerobot_info(p)
    lines += [f"\n── META ──",
              f"  Format:    LeRobot {info_d.get('codebase_version', '?')}",
              f"  Robot:     {info_d.get('robot_type', '?')}",
              f"  Episodes:  {info_d.get('total_episodes', '?')}",
              f"  Frames:    {info_d.get('total_frames', '?')}",
              f"  FPS:       {info_d.get('fps', '?')}"]

    lines.append(f"\n── FEATURES ──")
    for key, feat in info_d.get("features", {}).items():
        lines.append(f"  {key}: dtype={feat.get('dtype', '?')} shape={feat.get('shape', '?')}")
        if feat.get("names") and feat.get("dtype") != "video":
            lines.append(f"    names: {feat['names']}")

    lines.append(f"\n── ACTION STATS ──")
    act_stats = lerobot_action_stats(p)
    lines.append(f"  Likely unit: {act_stats.get('likely_unit', '?')}  dtype: {act_stats.get('dtype', '?')}")
    for name, js in act_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    lines.append(f"\n── STATE STATS ──")
    st_stats = lerobot_state_stats(p)
    lines.append(f"  Likely unit: {st_stats.get('likely_unit', '?')}")
    for name, js in st_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    lines.append(f"\n── ACTION-STATE CORRESPONDENCE ──")
    corr = lerobot_action_state_correspondence(p)
    lines.append(f"  Interpretation: {corr['interpretation']}")
    lines.append(f"  action vs state[t]   mean diff: {corr['action_vs_state_same_t']['total_mean_diff']:.6f}")
    lines.append(f"  action vs delta      mean diff: {corr['action_vs_state_delta']['total_mean_diff']:.6f}")
    lines.append(f"  action vs state[t+1] mean diff: {corr['action_vs_next_state']['total_mean_diff']:.6f}")

    lines.append(f"\n── NORMALIZATION STATS ──")
    try:
        stats = lerobot_stats(p)
        for key in ["action", "observation.state"]:
            if key in stats:
                s = stats[key]
                lines.append(f"  {key}:")
                lines.append(f"    min:  {[round(v, 4) for v in s.get('min', [])]}")
                lines.append(f"    max:  {[round(v, 4) for v in s.get('max', [])]}")
                lines.append(f"    mean: {[round(v, 4) for v in s.get('mean', [])]}")
                lines.append(f"    std:  {[round(v, 4) for v in s.get('std', [])]}")
    except FileNotFoundError:
        lines.append("  stats.json not found")

    lines.append(f"\n── VIDEOS ──")
    for cam, ci in lerobot_video_info(p).items():
        lines.append(f"  {cam}: {ci['num_files']} files, {ci['total_size_mb']} MB")

    lines.append(f"\n── EPISODE LENGTHS ──")
    eps = lerobot_episode_lengths(p)
    if eps:
        lengths = [e.get("length", e.get("num_frames", 0)) for e in eps]
        lines.append(f"  Count: {len(lengths)}, Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.1f}")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Compare / plot
# ═══════════════════════════════════════════════════════════════════


def _load_lerobot_actions(dataset_path: str | Path, max_files: int = 99) -> np.ndarray:
    data_dir = Path(dataset_path) / "data"
    all_actions = []
    for pq in sorted(data_dir.rglob("*.parquet"))[:max_files]:
        df = pd.read_parquet(pq)
        if "action" in df.columns:
            all_actions.append(np.stack(df["action"].values))
    return np.concatenate(all_actions, axis=0)


def _load_lerobot_episodes(dataset_path: str | Path, max_files: int = 99) -> list[np.ndarray]:
    """Load actions grouped by episode."""
    data_dir = Path(dataset_path) / "data"
    episodes = {}
    for pq in sorted(data_dir.rglob("*.parquet"))[:max_files]:
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
    return [episodes[k] for k in sorted(episodes.keys())]


def _detect_rest_frames(actions: np.ndarray, dist_thresh: float = 5.0,
                        window: int = 10) -> tuple[int, int]:
    """Detect rest regions at start/end of an episode."""
    n = len(actions)
    if n < window * 2:
        return 0, n
    start_pose = actions[:min(5, n)].mean(axis=0)
    end_pose = actions[max(0, n - 5):].mean(axis=0)
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
    return trim_start, min(trim_end, n)


def _detect_unit(abs_max: float) -> str:
    if abs_max <= 1.1:
        return "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        return "radians"
    elif abs_max <= 360:
        return "degrees"
    return f"unknown (abs_max={abs_max:.2f})"


def compare_actions(datasets: dict[str, str], bins: int = 80, save: str = None):
    """Plot overlaid action histograms for multiple LeRobot datasets."""
    import matplotlib.pyplot as plt

    loaded = {}
    for label, path in datasets.items():
        loaded[label] = _load_lerobot_actions(path)
        print(f"  {label}: {loaded[label].shape[0]} frames, {loaded[label].shape[1]} joints")

    n_joints = next(iter(loaded.values())).shape[1]
    first_path = Path(next(iter(datasets.values())))
    info_d = lerobot_info(first_path)
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    if len(joint_names) < n_joints:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Action Distributions", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n_joints:
            ax.set_visible(False)
            continue
        for j, (label, actions) in enumerate(loaded.items()):
            col = actions[:, i]
            color = colors[j % len(colors)]
            ax.hist(col, bins=bins, alpha=0.45, label=f"{label} (μ={col.mean():.1f})", color=color, edgecolor=color, linewidth=0.5)
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


def trajectories(dataset_path: str, episodes: list[int] = None, n_episodes: int = 6, save: str = None):
    """Plot joint trajectories for episodes, with rest regions shaded."""
    import matplotlib.pyplot as plt

    all_episodes = _load_lerobot_episodes(dataset_path)
    info_d = lerobot_info(dataset_path)
    joint_names = info_d.get("features", {}).get("action", {}).get("names", [])
    n_joints = all_episodes[0].shape[1] if all_episodes else 0
    if len(joint_names) < n_joints:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    if episodes is not None:
        ep_indices = [i for i in episodes if i < len(all_episodes)]
    else:
        ep_indices = list(range(min(n_episodes, len(all_episodes))))

    fig, axes = plt.subplots(len(ep_indices), 1, figsize=(16, 3 * len(ep_indices)), sharex=False)
    if len(ep_indices) == 1:
        axes = [axes]
    fig.suptitle(f"Joint Trajectories — {Path(dataset_path).name}", fontsize=14, fontweight="bold")
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

    loaded = {}
    for label, path in datasets.items():
        episodes = _load_lerobot_episodes(path)
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
    first_path = Path(next(iter(datasets.values())))
    info_d = lerobot_info(first_path)
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
            ax.hist(col, bins=bins, alpha=0.45, label=f"{label} (μ={col.mean():.1f})", color=color, edgecolor=color, linewidth=0.5)
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


# ═══════════════════════════════════════════════════════════════════
# Fire CLI
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "ls":              ls,
        "info":            info,
        "cameras":         cameras,
        "report":          lambda path: print(report(path)),
        "compare_actions": compare_actions,
        "compare_trimmed": compare_trimmed,
        "trajectories":    trajectories,
    })
