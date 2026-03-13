"""
Dataset inspection utilities for LeRobot and HDF5 datasets.

Provides detailed analysis of data types, ranges, units, and formats
to support reliable conversion between formats.

Usage:
    python -m vbti.utils.datasets.inspect lerobot /path/to/dataset
    python -m vbti.utils.datasets.inspect hdf5 /path/to/file.hdf5
"""
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── LeRobot inspection ──────────────────────────────────────────


def lerobot_info(dataset_path: str | Path) -> dict:
    """Load and return info.json as dict."""
    path = Path(dataset_path) / "meta" / "info.json"
    with open(path) as f:
        return json.load(f)


def lerobot_stats(dataset_path: str | Path) -> dict:
    """Load and return stats.json as dict."""
    path = Path(dataset_path) / "meta" / "stats.json"
    with open(path) as f:
        return json.load(f)


def lerobot_features(dataset_path: str | Path) -> dict:
    """Return features dict from info.json."""
    return lerobot_info(dataset_path).get("features", {})


def lerobot_parquet_schema(dataset_path: str | Path) -> dict:
    """Read first data parquet and return column dtypes and shapes."""
    path = Path(dataset_path) / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(path)
    result = {}
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            arr = np.array(sample)
            result[col] = {
                "parquet_type": type(sample).__name__,
                "element_dtype": str(arr.dtype),
                "shape": arr.shape,
            }
        else:
            result[col] = {
                "parquet_type": type(sample).__name__,
                "value_dtype": str(type(sample).__name__),
                "sample_value": sample,
            }
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

    stats = {
        "total_frames": len(actions),
        "num_joints": actions.shape[1],
        "dtype": str(actions.dtype),
        "joints": {},
    }

    for i in range(actions.shape[1]):
        col = actions[:, i]
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        stats["joints"][name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "abs_max": float(np.abs(col).max()),
        }

    # Heuristic: detect if values are degrees, radians, or normalized
    abs_max = np.abs(actions).max()
    if abs_max <= 1.1:
        stats["likely_unit"] = "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        stats["likely_unit"] = "radians"
    elif abs_max <= 360:
        stats["likely_unit"] = "degrees"
    else:
        stats["likely_unit"] = f"unknown (abs_max={abs_max:.2f})"

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

    stats = {
        "total_frames": len(states),
        "num_joints": states.shape[1],
        "dtype": str(states.dtype),
        "joints": {},
    }

    for i in range(states.shape[1]):
        col = states[:, i]
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        stats["joints"][name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }

    abs_max = np.abs(states).max()
    if abs_max <= 1.1:
        stats["likely_unit"] = "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        stats["likely_unit"] = "radians"
    elif abs_max <= 360:
        stats["likely_unit"] = "degrees"
    else:
        stats["likely_unit"] = f"unknown (abs_max={abs_max:.2f})"

    return stats


def lerobot_episode_lengths(dataset_path: str | Path) -> list[dict]:
    """Return list of {episode_index, length} from episode metadata."""
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
    """Return video file info (count, size, codec) per camera."""
    videos_dir = Path(dataset_path) / "videos"
    if not videos_dir.exists():
        return {}
    result = {}
    for cam_dir in sorted(videos_dir.iterdir()):
        if cam_dir.is_dir():
            files = list(cam_dir.rglob("*.mp4"))
            total_bytes = sum(f.stat().st_size for f in files)
            result[cam_dir.name] = {
                "num_files": len(files),
                "total_size_mb": round(total_bytes / 1e6, 1),
            }
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

    # Check absolute: action ≈ state
    abs_diff = np.abs(actions[:-1] - states[:-1]).mean(axis=0)

    # Check delta: action ≈ state[t+1] - state[t]
    state_deltas = states[1:] - states[:-1]
    delta_diff = np.abs(actions[:-1] - state_deltas).mean(axis=0)

    # Check next-state: action ≈ state[t+1]
    next_diff = np.abs(actions[:-1] - states[1:]).mean(axis=0)

    return {
        "action_vs_state_same_t": {
            "mean_abs_diff_per_joint": abs_diff.tolist(),
            "total_mean_diff": float(abs_diff.mean()),
        },
        "action_vs_state_delta": {
            "mean_abs_diff_per_joint": delta_diff.tolist(),
            "total_mean_diff": float(delta_diff.mean()),
        },
        "action_vs_next_state": {
            "mean_abs_diff_per_joint": next_diff.tolist(),
            "total_mean_diff": float(next_diff.mean()),
        },
        "interpretation": (
            "absolute (action ≈ state)" if abs_diff.mean() < delta_diff.mean() and abs_diff.mean() < next_diff.mean()
            else "next-state (action ≈ state[t+1])" if next_diff.mean() < delta_diff.mean()
            else "delta (action ≈ state[t+1] - state[t])"
        ),
    }


# ── HDF5 inspection ─────────────────────────────────────────────


def hdf5_tree(hdf5_path: str | Path, episode: str = "demo_0") -> dict:
    """Return full tree structure of an episode in the HDF5 file."""
    import h5py

    result = {}

    def _walk(group, prefix=""):
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                result[full_key] = {
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                    "size_mb": round(item.nbytes / 1e6, 2),
                }
            else:
                _walk(item, full_key)

    with h5py.File(hdf5_path, "r") as f:
        # Find an episode with actual data (not just initial_state)
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
    """Return list of episodes with frame counts."""
    import h5py
    import re
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        def sort_key(name):
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0
        for ep_name in sorted(f["data"].keys(), key=sort_key):
            ep = f["data"][ep_name]
            # Find first dataset to get frame count
            for key in ["actions", "processed_actions"]:
                if key in ep:
                    n_frames = ep[key].shape[0]
                    break
            else:
                # Walk into obs group
                obs = ep.get("obs", ep)
                for k in obs.keys():
                    if hasattr(obs[k], "shape"):
                        n_frames = obs[k].shape[0]
                        break
                else:
                    n_frames = 0
            episodes.append({"name": ep_name, "frames": n_frames})
    return episodes


def hdf5_action_stats(hdf5_path: str | Path) -> dict:
    """Compute detailed action statistics across all episodes."""
    import h5py
    all_actions = []
    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["data"].keys():
            ep = f["data"][ep_name]
            if "actions" not in ep:
                continue
            all_actions.append(np.array(ep["actions"]))

    actions = np.concatenate(all_actions, axis=0)
    stats = {
        "total_frames": len(actions),
        "num_joints": actions.shape[1],
        "dtype": str(actions.dtype),
        "joints": {},
    }
    for i in range(actions.shape[1]):
        col = actions[:, i]
        stats["joints"][f"joint_{i}"] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "abs_max": float(np.abs(col).max()),
        }

    abs_max = np.abs(actions).max()
    if abs_max <= 1.1:
        stats["likely_unit"] = "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        stats["likely_unit"] = "radians"
    elif abs_max <= 360:
        stats["likely_unit"] = "degrees"
    else:
        stats["likely_unit"] = f"unknown (abs_max={abs_max:.2f})"

    return stats


def hdf5_state_stats(hdf5_path: str | Path, state_key: str = "obs/joint_pos") -> dict:
    """Compute detailed state statistics across all episodes."""
    import h5py
    all_states = []
    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["data"].keys():
            ep = f["data"][ep_name]
            try:
                parts = state_key.split("/")
                node = ep
                for p in parts:
                    node = node[p]
                all_states.append(np.array(node))
            except KeyError:
                continue

    if not all_states:
        return {"error": f"key '{state_key}' not found"}

    states = np.concatenate(all_states, axis=0)
    stats = {
        "total_frames": len(states),
        "num_joints": states.shape[1],
        "dtype": str(states.dtype),
        "joints": {},
    }
    for i in range(states.shape[1]):
        col = states[:, i]
        stats["joints"][f"joint_{i}"] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }

    abs_max = np.abs(states).max()
    if abs_max <= 1.1:
        stats["likely_unit"] = "normalized [-1, 1]"
    elif abs_max <= 2 * math.pi + 0.5:
        stats["likely_unit"] = "radians"
    elif abs_max <= 360:
        stats["likely_unit"] = "degrees"
    else:
        stats["likely_unit"] = f"unknown (abs_max={abs_max:.2f})"

    return stats


def hdf5_image_stats(hdf5_path: str | Path, image_key: str = "obs/cam_top", episode: str = "demo_0") -> dict:
    """Check image dtype, shape, range for a single episode."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        # Find a valid episode with actual data
        ep = None
        for candidate_name in ([episode] if episode in f["data"] else []) + list(f["data"].keys()):
            if candidate_name in f["data"] and "actions" in f["data"][candidate_name]:
                ep = f["data"][candidate_name]
                break
        if ep is None:
            ep = f["data"][list(f["data"].keys())[0]]
        # Navigate nested path (e.g. "obs/cam_top")
        try:
            node = ep
            for part in image_key.split("/"):
                node = node[part]
            imgs = node
        except KeyError:
            return {"error": f"key '{image_key}' not found in {ep_name}"}
        # Read first and last frame only to avoid loading all into memory
        first = np.array(imgs[0])
        last = np.array(imgs[-1])
        return {
            "shape_per_frame": list(imgs.shape[1:]),
            "total_frames": imgs.shape[0],
            "dtype": str(imgs.dtype),
            "min": float(min(first.min(), last.min())),
            "max": float(max(first.max(), last.max())),
            "mean_first_frame": float(first.mean()),
            "is_uint8": imgs.dtype == np.uint8,
            "is_float_0_1": imgs.dtype in (np.float32, np.float64) and first.max() <= 1.0,
            "channels_last": imgs.shape[-1] in (1, 3, 4),
            "size_per_frame_kb": round(np.prod(imgs.shape[1:]) * imgs.dtype.itemsize / 1024, 1),
        }


def hdf5_action_state_correspondence(hdf5_path: str | Path, episode: str = "demo_0") -> dict:
    """Check if action[t] ≈ state[t] (absolute) or action[t] ≈ state[t+1]-state[t] (delta)."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        # Find an episode that has both actions and states
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
    state_deltas = states[1:] - states[:-1]
    delta_diff = np.abs(actions[:-1] - state_deltas).mean(axis=0)
    next_diff = np.abs(actions[:-1] - states[1:]).mean(axis=0)

    return {
        "action_vs_state_same_t": {
            "mean_abs_diff_per_joint": abs_diff.tolist(),
            "total_mean_diff": float(abs_diff.mean()),
        },
        "action_vs_state_delta": {
            "mean_abs_diff_per_joint": delta_diff.tolist(),
            "total_mean_diff": float(delta_diff.mean()),
        },
        "action_vs_next_state": {
            "mean_abs_diff_per_joint": next_diff.tolist(),
            "total_mean_diff": float(next_diff.mean()),
        },
        "interpretation": (
            "absolute (action ≈ state)" if abs_diff.mean() < delta_diff.mean() and abs_diff.mean() < next_diff.mean()
            else "next-state (action ≈ state[t+1])" if next_diff.mean() < delta_diff.mean()
            else "delta (action ≈ state[t+1] - state[t])"
        ),
    }


def hdf5_processed_actions_stats(hdf5_path: str | Path) -> dict:
    """Compare actions vs processed_actions to understand the transformation."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        # Find an episode with both actions and processed_actions
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

    ratio = proc / (actions + 1e-10)
    diff = proc - actions

    return {
        "actions": {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
        "processed_actions": {
            "shape": list(proc.shape),
            "dtype": str(proc.dtype),
            "min": proc.min(axis=0).tolist(),
            "max": proc.max(axis=0).tolist(),
        },
        "difference": {
            "mean_abs_diff": np.abs(diff).mean(axis=0).tolist(),
            "are_identical": bool(np.allclose(actions, proc)),
        },
        "ratio_mean": ratio.mean(axis=0).tolist(),
    }


# ── Report generators ────────────────────────────────────────────


def report_lerobot(dataset_path: str | Path) -> str:
    """Generate full inspection report for a LeRobot dataset."""
    p = Path(dataset_path)
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"LEROBOT DATASET REPORT: {p.name}")
    lines.append(f"Path: {p}")
    lines.append(f"{'='*70}")

    # Info
    info = lerobot_info(p)
    lines.append(f"\n── META ──")
    lines.append(f"  Format:    LeRobot {info.get('codebase_version', '?')}")
    lines.append(f"  Robot:     {info.get('robot_type', '?')}")
    lines.append(f"  Episodes:  {info.get('total_episodes', '?')}")
    lines.append(f"  Frames:    {info.get('total_frames', '?')}")
    lines.append(f"  FPS:       {info.get('fps', '?')}")
    lines.append(f"  Data size: {info.get('data_files_size_in_mb', '?')} MB")
    lines.append(f"  Video size:{info.get('video_files_size_in_mb', '?')} MB")

    # Features
    lines.append(f"\n── FEATURES ──")
    for key, feat in info.get("features", {}).items():
        dtype = feat.get("dtype", "?")
        shape = feat.get("shape", "?")
        names = feat.get("names", [])
        lines.append(f"  {key}: dtype={dtype} shape={shape}")
        if names and dtype != "video":
            lines.append(f"    names: {names}")

    # Parquet schema
    lines.append(f"\n── PARQUET SCHEMA ──")
    schema = lerobot_parquet_schema(p)
    for col, info_col in schema.items():
        lines.append(f"  {col}: {info_col}")

    # Action stats
    lines.append(f"\n── ACTION STATS ──")
    act_stats = lerobot_action_stats(p)
    lines.append(f"  Likely unit: {act_stats.get('likely_unit', '?')}")
    lines.append(f"  dtype: {act_stats.get('dtype', '?')}")
    lines.append(f"  Frames: {act_stats.get('total_frames', '?')}")
    for name, js in act_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    # State stats
    lines.append(f"\n── STATE STATS ──")
    st_stats = lerobot_state_stats(p)
    lines.append(f"  Likely unit: {st_stats.get('likely_unit', '?')}")
    for name, js in st_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    # Action-state correspondence
    lines.append(f"\n── ACTION-STATE CORRESPONDENCE ──")
    corr = lerobot_action_state_correspondence(p)
    lines.append(f"  Interpretation: {corr['interpretation']}")
    lines.append(f"  action vs state[t]  mean diff: {corr['action_vs_state_same_t']['total_mean_diff']:.6f}")
    lines.append(f"  action vs delta     mean diff: {corr['action_vs_state_delta']['total_mean_diff']:.6f}")
    lines.append(f"  action vs state[t+1] mean diff: {corr['action_vs_next_state']['total_mean_diff']:.6f}")

    # Normalization stats
    lines.append(f"\n── NORMALIZATION STATS (stats.json) ──")
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

    # Videos
    lines.append(f"\n── VIDEOS ──")
    vinfo = lerobot_video_info(p)
    for cam, ci in vinfo.items():
        lines.append(f"  {cam}: {ci['num_files']} files, {ci['total_size_mb']} MB")

    # Episode lengths
    lines.append(f"\n── EPISODE LENGTHS ──")
    eps = lerobot_episode_lengths(p)
    if eps:
        lengths = [e.get("length", e.get("num_frames", 0)) for e in eps]
        lines.append(f"  Count: {len(lengths)}")
        lines.append(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.1f}")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


def report_hdf5(hdf5_path: str | Path) -> str:
    """Generate full inspection report for an HDF5 dataset."""
    p = Path(hdf5_path)
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"HDF5 DATASET REPORT: {p.name}")
    lines.append(f"Path: {p}")
    lines.append(f"Size: {p.stat().st_size / 1e9:.2f} GB")
    lines.append(f"{'='*70}")

    # Episode list
    eps = hdf5_episode_list(p)
    lengths = [e["frames"] for e in eps]
    valid_lengths = [l for l in lengths if l > 0]
    lines.append(f"\n── EPISODES ──")
    lines.append(f"  Count: {len(eps)} ({len(valid_lengths)} with data, {len(eps) - len(valid_lengths)} empty)")
    lines.append(f"  Total frames: {sum(lengths)}")
    if valid_lengths:
        lines.append(f"  Min: {min(valid_lengths)}, Max: {max(valid_lengths)}, Mean: {sum(valid_lengths)/len(valid_lengths):.1f}")

    # Tree
    lines.append(f"\n── EPISODE STRUCTURE (first episode) ──")
    tree = hdf5_tree(p)
    for key, info_key in tree.items():
        lines.append(f"  {key}: shape={info_key['shape']} dtype={info_key['dtype']} ({info_key['size_mb']} MB)")

    # Action stats
    lines.append(f"\n── ACTION STATS ──")
    act_stats = hdf5_action_stats(p)
    lines.append(f"  Likely unit: {act_stats.get('likely_unit', '?')}")
    lines.append(f"  dtype: {act_stats.get('dtype', '?')}")
    for name, js in act_stats.get("joints", {}).items():
        lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")

    # State stats (joint_pos)
    lines.append(f"\n── STATE STATS (obs/joint_pos) ──")
    st_stats = hdf5_state_stats(p, "obs/joint_pos")
    if "error" not in st_stats:
        lines.append(f"  Likely unit: {st_stats.get('likely_unit', '?')}")
        for name, js in st_stats.get("joints", {}).items():
            lines.append(f"  {name:30s} [{js['min']:9.4f}, {js['max']:9.4f}]  mean={js['mean']:9.4f}  std={js['std']:7.4f}")
    else:
        lines.append(f"  {st_stats['error']}")

    # Processed actions comparison
    lines.append(f"\n── ACTIONS vs PROCESSED_ACTIONS ──")
    proc_stats = hdf5_processed_actions_stats(p)
    if "error" not in proc_stats:
        lines.append(f"  Identical: {proc_stats['difference']['are_identical']}")
        lines.append(f"  Actions range:    min={[f'{v:.4f}' for v in proc_stats['actions']['min']]}")
        lines.append(f"                    max={[f'{v:.4f}' for v in proc_stats['actions']['max']]}")
        lines.append(f"  Processed range:  min={[f'{v:.4f}' for v in proc_stats['processed_actions']['min']]}")
        lines.append(f"                    max={[f'{v:.4f}' for v in proc_stats['processed_actions']['max']]}")
        lines.append(f"  Mean abs diff:    {[f'{v:.6f}' for v in proc_stats['difference']['mean_abs_diff']]}")
    else:
        lines.append(f"  {proc_stats['error']}")

    # Action-state correspondence
    lines.append(f"\n── ACTION-STATE CORRESPONDENCE ──")
    corr = hdf5_action_state_correspondence(p)
    lines.append(f"  Interpretation: {corr['interpretation']}")
    lines.append(f"  action vs state[t]   mean diff: {corr['action_vs_state_same_t']['total_mean_diff']:.6f}")
    lines.append(f"  action vs delta      mean diff: {corr['action_vs_state_delta']['total_mean_diff']:.6f}")
    lines.append(f"  action vs state[t+1] mean diff: {corr['action_vs_next_state']['total_mean_diff']:.6f}")

    # Image stats for each camera
    lines.append(f"\n── IMAGE STATS ──")
    for key in tree:
        if any(cam in key for cam in ["cam_", "wrist", "front", "gripper"]) and "depth" not in key:
            img_stats = hdf5_image_stats(p, key)
            if "error" not in img_stats:
                lines.append(f"  {key}:")
                lines.append(f"    shape: {img_stats['shape_per_frame']} dtype={img_stats['dtype']}")
                lines.append(f"    range: [{img_stats['min']}, {img_stats['max']}]")
                lines.append(f"    uint8: {img_stats['is_uint8']}, channels_last: {img_stats['channels_last']}")
                lines.append(f"    size/frame: {img_stats['size_per_frame_kb']} KB")

    # Depth stats
    lines.append(f"\n── DEPTH STATS ──")
    for key in tree:
        if "depth" in key:
            img_stats = hdf5_image_stats(p, key)
            if "error" not in img_stats:
                lines.append(f"  {key}:")
                lines.append(f"    shape: {img_stats['shape_per_frame']} dtype={img_stats['dtype']}")
                lines.append(f"    range: [{img_stats['min']:.4f}, {img_stats['max']:.4f}]")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python inspect.py lerobot /path/to/dataset")
        print("  python inspect.py hdf5 /path/to/file.hdf5")
        sys.exit(1)

    fmt = sys.argv[1]
    path = sys.argv[2]

    if fmt == "lerobot":
        print(report_lerobot(path))
    elif fmt == "hdf5":
        print(report_hdf5(path))
    else:
        print(f"Unknown format: {fmt}. Use 'lerobot' or 'hdf5'")
