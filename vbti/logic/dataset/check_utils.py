#!/usr/bin/env python3
"""Dataset check, inspect, view and compare utilities.

Usage:
    python check_utils.py report_lerobot /path/to/dataset
    python check_utils.py report_hdf5 /path/to/file.hdf5
    python check_utils.py compare_actions --datasets real=/path sim=/path [--save plot.png]
    python check_utils.py info /path/to/file.hdf5
    python check_utils.py view /path/to/file.hdf5 0 cam_top
    python check_utils.py view /path/to/file.hdf5 0 all --sensor all
"""
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# LeRobot inspection
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
    info = lerobot_info(dataset_path)
    joint_names = info.get("features", {}).get("action", {}).get("names", [])
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
    info = lerobot_info(dataset_path)
    joint_names = info.get("features", {}).get("observation.state", {}).get("names", [])
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
# HDF5 inspection
# ═══════════════════════════════════════════════════════════════════


def hdf5_tree(hdf5_path: str | Path, episode: str = "demo_0") -> dict:
    result = {}
    def _walk(group, prefix=""):
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                result[full_key] = {"shape": item.shape, "dtype": str(item.dtype), "size_mb": round(item.nbytes / 1e6, 2)}
            else:
                _walk(item, full_key)

    with h5py.File(hdf5_path, "r") as f:
        target = None
        if episode in f["data"] and "actions" in f["data"][episode]:
            target = f["data"][episode]
        else:
            for ep_name in f["data"].keys():
                if "actions" in f["data"][ep_name]:
                    target = f["data"][ep_name]
                    break
        if target is None:
            target = f["data"][list(f["data"].keys())[0]]
        _walk(target)
    return result


def hdf5_episode_list(hdf5_path: str | Path) -> list[dict]:
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        def sort_key(name):
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0
        for ep_name in sorted(f["data"].keys(), key=sort_key):
            ep = f["data"][ep_name]
            for key in ["actions", "processed_actions"]:
                if key in ep:
                    n_frames = ep[key].shape[0]
                    break
            else:
                obs = ep.get("obs", ep)
                n_frames = 0
                for k in obs.keys():
                    if hasattr(obs[k], "shape"):
                        n_frames = obs[k].shape[0]
                        break
            episodes.append({"name": ep_name, "frames": n_frames})
    return episodes


def hdf5_action_stats(hdf5_path: str | Path) -> dict:
    all_actions = []
    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["data"].keys():
            ep = f["data"][ep_name]
            if "actions" not in ep:
                continue
            all_actions.append(np.array(ep["actions"]))

    actions = np.concatenate(all_actions, axis=0)
    stats = {"total_frames": len(actions), "num_joints": actions.shape[1], "dtype": str(actions.dtype), "joints": {}}
    for i in range(actions.shape[1]):
        col = actions[:, i]
        stats["joints"][f"joint_{i}"] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std()), "abs_max": float(np.abs(col).max())}
    abs_max = np.abs(actions).max()
    stats["likely_unit"] = _detect_unit(abs_max)
    return stats


def hdf5_state_stats(hdf5_path: str | Path, state_key: str = "obs/joint_pos") -> dict:
    all_states = []
    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["data"].keys():
            ep = f["data"][ep_name]
            try:
                node = ep
                for p in state_key.split("/"):
                    node = node[p]
                all_states.append(np.array(node))
            except KeyError:
                continue
    if not all_states:
        return {"error": f"key '{state_key}' not found"}

    states = np.concatenate(all_states, axis=0)
    stats = {"total_frames": len(states), "num_joints": states.shape[1], "dtype": str(states.dtype), "joints": {}}
    for i in range(states.shape[1]):
        col = states[:, i]
        stats["joints"][f"joint_{i}"] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std())}
    abs_max = np.abs(states).max()
    stats["likely_unit"] = _detect_unit(abs_max)
    return stats


def hdf5_image_stats(hdf5_path: str | Path, image_key: str = "obs/cam_top", episode: str = "demo_0") -> dict:
    with h5py.File(hdf5_path, "r") as f:
        ep = None
        for candidate_name in ([episode] if episode in f["data"] else []) + list(f["data"].keys()):
            if candidate_name in f["data"] and "actions" in f["data"][candidate_name]:
                ep = f["data"][candidate_name]
                break
        if ep is None:
            ep = f["data"][list(f["data"].keys())[0]]
        try:
            node = ep
            for part in image_key.split("/"):
                node = node[part]
            imgs = node
        except KeyError:
            return {"error": f"key '{image_key}' not found"}
        first = np.array(imgs[0])
        last = np.array(imgs[-1])
        return {
            "shape_per_frame": list(imgs.shape[1:]), "total_frames": imgs.shape[0], "dtype": str(imgs.dtype),
            "min": float(min(first.min(), last.min())), "max": float(max(first.max(), last.max())),
            "is_uint8": imgs.dtype == np.uint8, "channels_last": imgs.shape[-1] in (1, 3, 4),
            "size_per_frame_kb": round(np.prod(imgs.shape[1:]) * imgs.dtype.itemsize / 1024, 1),
        }


def hdf5_action_state_correspondence(hdf5_path: str | Path, episode: str = "demo_0") -> dict:
    with h5py.File(hdf5_path, "r") as f:
        ep = None
        for candidate in [episode] + list(f["data"].keys()):
            if candidate in f["data"] and "actions" in f["data"][candidate]:
                ep = f["data"][candidate]
                break
        if ep is None:
            return {"error": "no episode with actions found"}
        actions = np.array(ep["actions"])
        states = np.array(ep["obs/joint_pos"])

    abs_diff = np.abs(actions[:-1] - states[:-1]).mean(axis=0)
    delta_diff = np.abs(actions[:-1] - (states[1:] - states[:-1])).mean(axis=0)
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


def hdf5_processed_actions_stats(hdf5_path: str | Path) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        ep = None
        for ep_name in f["data"].keys():
            candidate = f["data"][ep_name]
            if "actions" in candidate and "processed_actions" in candidate:
                ep = candidate
                break
        if ep is None:
            return {"error": "no episode with processed_actions found"}
        actions = np.array(ep["actions"])
        proc = np.array(ep["processed_actions"])

    diff = proc - actions
    ratio = proc / (actions + 1e-10)
    return {
        "actions": {"shape": list(actions.shape), "dtype": str(actions.dtype), "min": actions.min(axis=0).tolist(), "max": actions.max(axis=0).tolist()},
        "processed_actions": {"shape": list(proc.shape), "dtype": str(proc.dtype), "min": proc.min(axis=0).tolist(), "max": proc.max(axis=0).tolist()},
        "difference": {"mean_abs_diff": np.abs(diff).mean(axis=0).tolist(), "are_identical": bool(np.allclose(actions, proc))},
        "ratio_mean": ratio.mean(axis=0).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# Report generators
# ═══════════════════════════════════════════════════════════════════


def report_lerobot(dataset_path: str | Path) -> str:
    """Full inspection report for a LeRobot dataset."""
    p = Path(dataset_path)
    lines = [f"{'='*70}", f"LEROBOT DATASET REPORT: {p.name}", f"Path: {p}", f"{'='*70}"]

    info = lerobot_info(p)
    lines += [f"\n── META ──",
              f"  Format:    LeRobot {info.get('codebase_version', '?')}",
              f"  Robot:     {info.get('robot_type', '?')}",
              f"  Episodes:  {info.get('total_episodes', '?')}",
              f"  Frames:    {info.get('total_frames', '?')}",
              f"  FPS:       {info.get('fps', '?')}"]

    lines.append(f"\n── FEATURES ──")
    for key, feat in info.get("features", {}).items():
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


def report_hdf5(hdf5_path: str | Path) -> str:
    """Full inspection report for an HDF5 dataset."""
    p = Path(hdf5_path)
    lines = [f"{'='*70}", f"HDF5 DATASET REPORT: {p.name}", f"Path: {p}", f"Size: {p.stat().st_size / 1e9:.2f} GB", f"{'='*70}"]

    eps = hdf5_episode_list(p)
    lengths = [e["frames"] for e in eps]
    valid_lengths = [l for l in lengths if l > 0]
    lines.append(f"\n── EPISODES ──")
    lines.append(f"  Count: {len(eps)} ({len(valid_lengths)} with data, {len(eps) - len(valid_lengths)} empty)")
    lines.append(f"  Total frames: {sum(lengths)}")
    if valid_lengths:
        lines.append(f"  Min: {min(valid_lengths)}, Max: {max(valid_lengths)}, Mean: {sum(valid_lengths)/len(valid_lengths):.1f}")

    lines.append(f"\n── EPISODE STRUCTURE (first episode) ──")
    tree = hdf5_tree(p)
    for key, info_key in tree.items():
        lines.append(f"  {key}: shape={info_key['shape']} dtype={info_key['dtype']} ({info_key['size_mb']} MB)")

    lines.append(f"\n── ACTION STATS ──")
    act_stats = hdf5_action_stats(p)
    lines.append(f"  Likely unit: {act_stats.get('likely_unit', '?')}  dtype: {act_stats.get('dtype', '?')}")
    for name, js in act_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    lines.append(f"\n── STATE STATS (obs/joint_pos) ──")
    st_stats = hdf5_state_stats(p, "obs/joint_pos")
    if "error" not in st_stats:
        lines.append(f"  Likely unit: {st_stats.get('likely_unit', '?')}")
        for name, js in st_stats.get("joints", {}).items():
            lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")
    else:
        lines.append(f"  {st_stats['error']}")

    lines.append(f"\n── ACTIONS vs PROCESSED_ACTIONS ──")
    proc_stats = hdf5_processed_actions_stats(p)
    if "error" not in proc_stats:
        lines.append(f"  Identical: {proc_stats['difference']['are_identical']}")
        lines.append(f"  Mean abs diff: {[f'{v:.6f}' for v in proc_stats['difference']['mean_abs_diff']]}")
    else:
        lines.append(f"  {proc_stats['error']}")

    lines.append(f"\n── ACTION-STATE CORRESPONDENCE ──")
    corr = hdf5_action_state_correspondence(p)
    lines.append(f"  Interpretation: {corr['interpretation']}")

    lines.append(f"\n── IMAGE STATS ──")
    for key in tree:
        if any(cam in key for cam in ["cam_", "wrist", "front", "gripper"]) and "depth" not in key:
            img_stats = hdf5_image_stats(p, key)
            if "error" not in img_stats:
                lines.append(f"  {key}: {img_stats['shape_per_frame']} {img_stats['dtype']} [{img_stats['min']}, {img_stats['max']}] {img_stats['size_per_frame_kb']}KB/frame")

    lines.append(f"\n── DEPTH STATS ──")
    for key in tree:
        if "depth" in key:
            img_stats = hdf5_image_stats(p, key)
            if "error" not in img_stats:
                lines.append(f"  {key}: {img_stats['shape_per_frame']} {img_stats['dtype']} [{img_stats['min']:.4f}, {img_stats['max']:.4f}]")

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
    """Load actions grouped by episode. Returns list of (n_frames, n_joints) arrays."""
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
    """Detect rest regions at start/end of an episode.

    Compares each frame to the start/end rest pose. A frame is "rest" if
    all joints are within dist_thresh of the rest pose. Uses a window to
    avoid cutting on brief pauses mid-episode.

    Returns (trim_start, trim_end) frame indices for the active region.
    """
    n = len(actions)
    if n < window * 2:
        return 0, n

    # Rest pose = average of first/last few frames
    start_pose = actions[:min(5, n)].mean(axis=0)
    end_pose = actions[max(0, n - 5):].mean(axis=0)

    # Distance from start/end rest pose per frame
    dist_from_start = np.abs(actions - start_pose).max(axis=1)
    dist_from_end = np.abs(actions - end_pose).max(axis=1)

    # Find first frame that's far enough from start rest pose
    trim_start = 0
    for i in range(n):
        if dist_from_start[i] > dist_thresh:
            # Walk back a few frames for margin
            trim_start = max(0, i - 2)
            break

    # Find last frame that's far enough from end rest pose
    trim_end = n
    for i in range(n - 1, -1, -1):
        if dist_from_end[i] > dist_thresh:
            trim_end = min(n, i + 3)
            break

    return trim_start, min(trim_end, n)


def compare_actions(datasets: dict[str, str], bins: int = 80, save: str = None):
    """Plot overlaid action histograms for multiple LeRobot datasets.

    Args:
        datasets: label=path pairs, e.g. --datasets real=/path sim=/path
        bins: Number of histogram bins.
        save: Save plot to file instead of showing.
    """
    import matplotlib.pyplot as plt

    loaded = {}
    for label, path in datasets.items():
        loaded[label] = _load_lerobot_actions(path)
        print(f"  {label}: {loaded[label].shape[0]} frames, {loaded[label].shape[1]} joints")

    n_joints = next(iter(loaded.values())).shape[1]

    first_path = Path(next(iter(datasets.values())))
    info = lerobot_info(first_path)
    joint_names = info.get("features", {}).get("action", {}).get("names", [])
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
        ax.axvline(x=-100, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=100, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


def trajectories(dataset_path: str, episodes: list[int] = None, n_episodes: int = 6, save: str = None):
    """Plot joint trajectories for episodes, with rest regions shaded.

    Shows where each episode starts/ends in rest position vs active motion.

    Args:
        dataset_path: Path to LeRobot dataset.
        episodes: Specific episode indices to plot, e.g. [0, 4, 9].
        n_episodes: Number of episodes to plot (used when episodes is None).
        save: Save plot to file instead of showing.
    """
    import matplotlib.pyplot as plt

    all_episodes = _load_lerobot_episodes(dataset_path)
    info = lerobot_info(dataset_path)
    joint_names = info.get("features", {}).get("action", {}).get("names", [])
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

        # Shade rest regions
        if trim_start > 0:
            ax.axvspan(0, trim_start, alpha=0.15, color="red", label="rest" if row == 0 else None)
        if trim_end < n_frames:
            ax.axvspan(trim_end, n_frames, alpha=0.15, color="red")

        # Plot each joint
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
    """Compare action distributions after trimming rest poses from episode start/end.

    Args:
        datasets: label=path pairs, e.g. --datasets real=/path sim=/path
        dist_thresh: Min distance from rest pose to consider "active" (in normalized units).
        bins: Number of histogram bins.
        save: Save plot to file instead of showing.
    """
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
        print(f"  {label}: {trimmed_frames}/{total_frames} frames kept ({pct:.0f}%), "
              f"trimmed {total_frames - trimmed_frames} rest frames")

    n_joints = next(iter(loaded.values())).shape[1]
    first_path = Path(next(iter(datasets.values())))
    info = lerobot_info(first_path)
    joint_names = info.get("features", {}).get("action", {}).get("names", [])
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
            ax.hist(col, bins=bins, alpha=0.45,
                    label=f"{label} (μ={col.mean():.1f})",
                    color=color, edgecolor=color, linewidth=0.5)
        ax.set_title(joint_names[i], fontsize=11)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.axvline(x=-100, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=100, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════
# HDF5 viewer (video playback)
# ═══════════════════════════════════════════════════════════════════

_NON_CAM_KEYS = {"joint_pos", "joint_vel", "joint_pos_rel", "joint_vel_rel", "actions"}
_SENSOR_SUFFIX = {"rgb": "", "depth": "_depth", "seg": "_seg"}


def _natural_sort(names):
    def key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    return sorted(names, key=key)


def _resolve_episode(f, episode):
    episodes = _natural_sort([k for k in f["data"].keys() if k.startswith("demo_")])
    if isinstance(episode, int):
        if episode < 0 or episode >= len(episodes):
            raise ValueError(f"Episode {episode} out of range [0, {len(episodes) - 1}]")
        return episodes[episode]
    if episode in f["data"]:
        return episode
    raise ValueError(f"Episode '{episode}' not found. Available: {episodes[:5]}...")


def _discover_cameras(obs_keys):
    return sorted(set(k.replace("_depth", "").replace("_seg", "") for k in obs_keys if k not in _NON_CAM_KEYS))


def _sensor_key(camera, sensor):
    if sensor not in _SENSOR_SUFFIX:
        raise ValueError(f"Unknown sensor '{sensor}'. Choose from: {list(_SENSOR_SUFFIX.keys())}")
    return f"{camera}{_SENSOR_SUFFIX[sensor]}"


def _prepare_frames(frames, sensor):
    if sensor == "rgb":
        return frames[:, :, :, ::-1].copy()
    elif sensor == "depth":
        sq = frames.squeeze()
        clipped = np.clip(sq, 0.01, 2.0)
        norm = ((clipped - 0.01) / (2.0 - 0.01) * 255).astype(np.uint8)
        return np.stack([cv2.applyColorMap(f, cv2.COLORMAP_TURBO) for f in norm])
    elif sensor == "seg":
        if frames.shape[-1] == 4:
            return frames[:, :, :, :3][:, :, :, ::-1].copy()
        return frames
    return frames


def _load_grid(f, ep_name, cameras, sensors):
    rows = []
    for cam in cameras:
        cols = []
        for sensor in sensors:
            full_key = f"data/{ep_name}/obs/{_sensor_key(cam, sensor)}"
            if full_key not in f:
                continue
            cols.append(_prepare_frames(f[full_key][:], sensor))
        if cols:
            min_t = min(c.shape[0] for c in cols)
            rows.append(np.concatenate([c[:min_t] for c in cols], axis=2))
    if not rows:
        raise ValueError("No valid camera/sensor combinations found")
    min_t = min(r.shape[0] for r in rows)
    max_w = max(r.shape[2] for r in rows)
    padded = []
    for r in rows:
        r = r[:min_t]
        if r.shape[2] < max_w:
            pad = np.zeros((r.shape[0], r.shape[1], max_w - r.shape[2], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=2)
        padded.append(r)
    return np.concatenate(padded, axis=1)


_BAR_H, _BAR_PAD = 32, 8
_BAR_COLOR, _PROGRESS_COLOR, _CURSOR_COLOR, _TEXT_COLOR = (80, 80, 80), (0, 180, 255), (255, 255, 255), (220, 220, 220)


def _draw_hud(frame, idx, n_frames, fps, paused, label):
    h, w = frame.shape[:2]
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - _BAR_H), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    x0, x1, y_mid = _BAR_PAD + 80, w - _BAR_PAD - 90, h - _BAR_H // 2
    cv2.line(out, (x0, y_mid), (x1, y_mid), _BAR_COLOR, 2)
    progress_x = x0 + int((x1 - x0) * idx / max(n_frames - 1, 1))
    cv2.line(out, (x0, y_mid), (progress_x, y_mid), _PROGRESS_COLOR, 2)
    cv2.circle(out, (progress_x, y_mid), 5, _CURSOR_COLOR, -1)
    cv2.putText(out, f"{idx}/{n_frames - 1}", (_BAR_PAD, y_mid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_COLOR, 1)
    status = "PAUSED" if paused else f"{fps}fps"
    cv2.putText(out, status, (x1 + 8, y_mid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if paused else _TEXT_COLOR, 1)
    cv2.putText(out, label, (_BAR_PAD, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_COLOR, 1)
    return out


def _timeline_click(event, x, y, flags, param):
    state = param
    if y < state["h"] - _BAR_H:
        return
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        x0, x1 = _BAR_PAD + 80, state["w"] - _BAR_PAD - 90
        if x0 <= x <= x1:
            state["idx"] = int((x - x0) / (x1 - x0) * (state["n_frames"] - 1))
            state["paused"] = True


def _play(frames, title, fps, save):
    n_frames, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    if save:
        has_ffmpeg = shutil.which("ffmpeg") is not None
        tmp_path = save if not has_ffmpeg else tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        if has_ffmpeg and tmp_path != save:
            subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", "18", "-preset", "fast", save], capture_output=True)
            os.remove(tmp_path)
        print(f"  Saved {n_frames} frames → {save}")
        return

    win = "viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(1920 / w, 1080 / h, 1.0)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))
    state = {"idx": 0, "n_frames": n_frames, "paused": True, "h": h, "w": w}
    cv2.setMouseCallback(win, _timeline_click, state)
    delay = max(1, int(1000 / fps))
    speed = 1
    print(f"\n  Controls: SPACE=play/pause  A/D=step  W/S=speed  R=restart  Q=quit\n")
    while True:
        display = _draw_hud(frames[state["idx"]], state["idx"], n_frames, fps * speed, state["paused"], title)
        cv2.imshow(win, display)
        key = cv2.waitKey(0 if state["paused"] else delay) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            state["paused"] = not state["paused"]
        elif key == ord("d") or key == 83:
            state["idx"] = min(state["idx"] + 1, n_frames - 1); state["paused"] = True
        elif key == ord("a") or key == 81:
            state["idx"] = max(state["idx"] - 1, 0); state["paused"] = True
        elif key == ord("w") or key == 82:
            speed = min(speed * 2, 16)
        elif key == ord("s") or key == 84:
            speed = max(speed // 2, 1)
        elif key == ord("r"):
            state["idx"] = 0
        elif key == ord("["):
            state["idx"] = max(state["idx"] - 30, 0)
        elif key == ord("]"):
            state["idx"] = min(state["idx"] + 30, n_frames - 1)
        else:
            if not state["paused"]:
                state["idx"] += speed
                if state["idx"] >= n_frames:
                    state["idx"] = n_frames - 1; state["paused"] = True
    cv2.destroyAllWindows()


def info(dataset_path: str):
    """Print HDF5 dataset structure: episodes, cameras, sensors, shapes."""
    with h5py.File(dataset_path, "r") as f:
        total = f["data"].attrs.get("total", "?")
        env_name = f["data"].attrs.get("env_name", "?")
        episodes = _natural_sort([k for k in f["data"].keys() if k.startswith("demo_")])
        print(f"\n  Dataset: {dataset_path}")
        print(f"  Task:    {env_name}")
        print(f"  Total frames: {total}")
        print(f"  Episodes: {len(episodes)}\n")
        for ep_name in episodes:
            ep = f[f"data/{ep_name}"]
            if "obs" not in ep:
                continue
            n = ep.attrs.get("num_samples", "?")
            success = ep.attrs.get("success", "?")
            print(f"  {ep_name}  ({n} frames, success={success})")
            for key in sorted(ep["obs"].keys()):
                ds = ep[f"obs/{key}"]
                print(f"    obs/{key:20s}  {str(ds.shape):20s}  {ds.dtype}")
            print()
            remaining = [e for e in episodes[1:] if "obs" in f[f"data/{e}"]]
            if remaining:
                lengths = [f["data"][e].attrs.get("num_samples", 0) for e in remaining]
                print(f"  ... {len(remaining)} more episodes ({min(lengths)}-{max(lengths)} frames each)")
            break


def view(dataset_path: str, episode: int = 0, camera: str = None,
         sensor: str = "rgb", fps: int = 30, save: str = None):
    """Play or save video from an HDF5 dataset episode.

    Args:
        dataset_path: Path to the HDF5 file.
        episode: Episode index or name (str like 'demo_5').
        camera: Camera name, space-separated list, or 'all'. None lists available.
        sensor: Sensor type: 'rgb', 'depth', 'seg', space-separated, or 'all'.
        fps: Playback framerate.
        save: Save to this mp4 path instead of playing.
    """
    with h5py.File(dataset_path, "r") as f:
        ep_name = _resolve_episode(f, episode)
        ep = f[f"data/{ep_name}"]
        if "obs" not in ep:
            print(f"  [WARN] {ep_name} has no obs data")
            return
        obs_keys = list(ep["obs"].keys())
        all_cams = _discover_cameras(obs_keys)
        if camera is None:
            print(f"\n  {ep_name} — available cameras:")
            for c in all_cams:
                types = [s for s in _SENSOR_SUFFIX if _sensor_key(c, s) in obs_keys or (s == "rgb" and c in obs_keys)]
                print(f"    {c:20s}  [{', '.join(types)}]")
            print(f"\n  Usage: view {dataset_path} {episode} <camera|all> [--sensor rgb|depth|seg|all]")
            return
        cameras = all_cams if camera == "all" else camera.split()
        sensors = list(_SENSOR_SUFFIX.keys()) if sensor == "all" else sensor.split()
        if len(cameras) == 1 and len(sensors) == 1:
            obs_key = _sensor_key(cameras[0], sensors[0])
            full_key = f"data/{ep_name}/obs/{obs_key}"
            if full_key not in f:
                print(f"  [ERROR] Key '{full_key}' not found. Available obs: {obs_keys}")
                return
            raw = f[full_key][:]
            print(f"\n  {ep_name}/{cameras[0]}/{sensors[0]}  shape={raw.shape}  dtype={raw.dtype}")
            frames = _prepare_frames(raw, sensors[0])
            title = f"{ep_name} | {cameras[0]} | {sensors[0]} ({frames.shape[0]}f)"
            _play(frames, title, fps, save)
            return
        cam_str = "all" if camera == "all" else "+".join(cameras)
        sen_str = "all" if sensor == "all" else "+".join(sensors)
        print(f"\n  {ep_name} — grid: [{cam_str}] x [{sen_str}]")
        frames = _load_grid(f, ep_name, cameras, sensors)
    title = f"{ep_name} | {cam_str} x {sen_str} ({frames.shape[0]}f)"
    _play(frames, title, fps, save)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _detect_unit(abs_max: float) -> str:
    if abs_max <= 1.1:
        return "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        return "radians"
    elif abs_max <= 360:
        return "degrees"
    return f"unknown (abs_max={abs_max:.2f})"


# ═══════════════════════════════════════════════════════════════════
# Fire CLI
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "report_lerobot": lambda path: print(report_lerobot(path)),
        "report_hdf5": lambda path: print(report_hdf5(path)),
        "compare_actions": compare_actions,
        "compare_trimmed": compare_trimmed,
        "trajectories": trajectories,
        "info": info,
        "view": view,
    })
