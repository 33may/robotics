"""
Inspect converted LeRobot dataset structure and verify data integrity.

Usage:
    python check_converted_dataset.py --repo_id=eternalmay33/lift_cube_3cams
    python check_converted_dataset.py --dataset_path=/path/to/dataset
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def format_size(num_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def inspect_lerobot_dataset(dataset_path: Path):
    print(f"\n{'='*60}")
    print(f"LEROBOT DATASET: {dataset_path.name}")
    print(f"{'='*60}")
    print(f"Path: {dataset_path}\n")

    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist!")
        return

    # ========== META INFO ==========
    print(f"{'='*60}")
    print("META INFO")
    print("-" * 40)

    info_path = dataset_path / "meta/info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)

        print(f"  Total episodes: {info.get('total_episodes', 'N/A')}")
        print(f"  Total frames: {info.get('total_frames', 'N/A')}")
        print(f"  FPS: {info.get('fps', 'N/A')}")
        print(f"  Robot type: {info.get('robot_type', 'N/A')}")
        print(f"  Chunks size: {info.get('chunks_size', 'N/A')}")

        if 'data_files_size_in_mb' in info:
            print(f"  Data size: {info['data_files_size_in_mb']:.1f} MB")
        if 'video_files_size_in_mb' in info:
            print(f"  Video size: {info['video_files_size_in_mb']:.1f} MB")
    else:
        print("  ERROR: info.json not found!")
        return

    # ========== FEATURES ==========
    print(f"\n{'='*60}")
    print("FEATURES")
    print("-" * 40)

    features = info.get('features', {})
    for key, feature in features.items():
        dtype = feature.get('dtype', 'N/A')
        shape = feature.get('shape', 'N/A')
        print(f"\n  {key}:")
        print(f"      dtype: {dtype}")
        print(f"      shape: {shape}")
        if 'names' in feature:
            print(f"      names: {feature['names']}")
        if 'video_info' in feature:
            vi = feature['video_info']
            print(f"      video: {vi.get('video.width')}x{vi.get('video.height')} @ {vi.get('video.fps')}fps, codec={vi.get('video.codec')}")

    # ========== STATS (Normalization) ==========
    print(f"\n{'='*60}")
    print("NORMALIZATION STATS")
    print("-" * 40)

    stats_path = dataset_path / "meta/stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        for key in ['action', 'observation.state']:
            if key in stats:
                s = stats[key]
                print(f"\n  {key}:")
                if 'min' in s:
                    min_vals = s['min']
                    max_vals = s['max']
                    mean_vals = s.get('mean', [])
                    std_vals = s.get('std', [])

                    print(f"      min:  {[f'{v:.2f}' for v in min_vals]}")
                    print(f"      max:  {[f'{v:.2f}' for v in max_vals]}")
                    if mean_vals:
                        print(f"      mean: {[f'{v:.2f}' for v in mean_vals]}")
                    if std_vals:
                        print(f"      std:  {[f'{v:.2f}' for v in std_vals]}")
    else:
        print("  WARNING: stats.json not found!")

    # ========== DATA PARQUET ==========
    print(f"\n{'='*60}")
    print("DATA INSPECTION")
    print("-" * 40)

    data_file = dataset_path / "data/chunk-000/file-000.parquet"
    if data_file.exists():
        try:
            df = pd.read_parquet(data_file)
            print(f"\n  Total frames in parquet: {len(df)}")
            print(f"  Columns: {list(df.columns)}")

            # Action ranges
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)
                print(f"\n  ACTION RANGES (per joint):")
                for i in range(actions.shape[1]):
                    min_v = actions[:, i].min()
                    max_v = actions[:, i].max()
                    mean_v = actions[:, i].mean()
                    std_v = actions[:, i].std()
                    print(f"      joint_{i}: [{min_v:8.2f}, {max_v:8.2f}] mean={mean_v:7.2f} std={std_v:6.2f}")

                # Check for issues
                action_magnitude = np.abs(actions).mean()
                print(f"\n  Mean action magnitude: {action_magnitude:.2f}")
                if action_magnitude < 1.0:
                    print("  WARNING: Actions are very small! May cause 'robot not moving'")
                elif action_magnitude > 200:
                    print("  WARNING: Actions are very large! May cause erratic behavior")
                else:
                    print("  OK: Action magnitude looks reasonable")

            # State ranges
            if 'observation.state' in df.columns:
                states = np.stack(df['observation.state'].values)
                print(f"\n  STATE RANGES (per joint):")
                for i in range(states.shape[1]):
                    min_v = states[:, i].min()
                    max_v = states[:, i].max()
                    print(f"      joint_{i}: [{min_v:8.2f}, {max_v:8.2f}]")

            # Sample frame
            print(f"\n  SAMPLE FRAME (index 10):")
            if len(df) > 10:
                sample = df.iloc[10]
                print(f"      episode_index: {sample.get('episode_index', 'N/A')}")
                print(f"      frame_index: {sample.get('frame_index', 'N/A')}")
                print(f"      timestamp: {sample.get('timestamp', 'N/A')}")
                if 'action' in sample:
                    action = sample['action']
                    print(f"      action: {action}")
                    print(f"      action dtype: {type(action).__name__}, element dtype: {np.array(action).dtype}")
                if 'observation.state' in sample:
                    state = sample['observation.state']
                    print(f"      state dtype: {type(state).__name__}, element dtype: {np.array(state).dtype}")

        except Exception as e:
            print(f"  ERROR reading parquet: {e}")
    else:
        print("  WARNING: data parquet not found!")

    # ========== EPISODES ==========
    print(f"\n{'='*60}")
    print("EPISODES")
    print("-" * 40)

    episodes_file = dataset_path / "meta/episodes.jsonl"
    if not episodes_file.exists():
        episodes_file = dataset_path / "meta/episodes/chunk-000/file-000.parquet"

    if episodes_file.exists() and episodes_file.suffix == '.parquet':
        try:
            ep_df = pd.read_parquet(episodes_file)
            print(f"\n  Total episodes: {len(ep_df)}")
            if 'length' in ep_df.columns:
                print(f"  Min frames: {ep_df['length'].min()}")
                print(f"  Max frames: {ep_df['length'].max()}")
                print(f"  Mean frames: {ep_df['length'].mean():.1f}")

                # Show first few episodes
                print(f"\n  First 5 episodes:")
                for i in range(min(5, len(ep_df))):
                    length = ep_df['length'].iloc[i]
                    print(f"      episode_{i}: {length} frames")
        except Exception as e:
            print(f"  ERROR reading episodes: {e}")

    # ========== IMAGE DTYPE CHECK ==========
    print(f"\n{'='*60}")
    print("IMAGE DTYPE CHECK (via LeRobot)")
    print("-" * 40)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        # Try to load dataset to check actual image dtypes
        ds = LeRobotDataset(dataset_path.parent.name + "/" + dataset_path.name)
        sample = ds[10]

        for key, value in sample.items():
            if 'image' in key:
                if isinstance(value, torch.Tensor):
                    print(f"  {key}:")
                    print(f"      shape: {list(value.shape)}")
                    print(f"      dtype: {value.dtype}")
                    print(f"      range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}:")
                    print(f"      shape: {value.shape}")
                    print(f"      dtype: {value.dtype}")
                    print(f"      range: [{value.min():.3f}, {value.max():.3f}]")
    except Exception as e:
        print(f"  Could not load dataset: {e}")
        print("  (This is fine - dtype info shown in FEATURES section above)")

    # ========== VIDEOS ==========
    print(f"\n{'='*60}")
    print("VIDEOS")
    print("-" * 40)

    videos_path = dataset_path / "videos"
    if videos_path.exists():
        for video_dir in sorted(videos_path.iterdir()):
            if video_dir.is_dir():
                chunk_path = video_dir / "chunk-000"
                if chunk_path.exists():
                    video_files = list(chunk_path.glob("*.mp4"))
                    total_size = sum(f.stat().st_size for f in video_files)
                    print(f"  {video_dir.name}:")
                    print(f"      {len(video_files)} videos, {format_size(total_size)}")
                else:
                    print(f"  {video_dir.name}: no chunk-000 found")
    else:
        print("  WARNING: videos directory not found!")

    # ========== TASKS ==========
    print(f"\n{'='*60}")
    print("TASKS")
    print("-" * 40)

    tasks_file = dataset_path / "meta/tasks.parquet"
    if tasks_file.exists():
        try:
            tasks_df = pd.read_parquet(tasks_file)
            print(f"  Tasks defined: {len(tasks_df)}")
            if 'task' in tasks_df.columns:
                for i, task in enumerate(tasks_df['task'].unique()):
                    print(f"      {i}: {task}")
        except Exception as e:
            print(f"  ERROR reading tasks: {e}")
    else:
        print("  WARNING: tasks.parquet not found!")

    # ========== SUMMARY ==========
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print("="*60)

    checks = {
        "info.json exists": info_path.exists(),
        "stats.json exists": stats_path.exists(),
        "data parquet exists": data_file.exists(),
        "videos directory exists": videos_path.exists(),
    }

    all_good = True
    for check, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {check}")
        if not passed:
            all_good = False

    if all_good:
        print(f"\nDataset ready for training!")
    else:
        print(f"\nDataset has issues - check warnings above")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect converted LeRobot dataset")
    parser.add_argument("--repo_id", type=str, help="HuggingFace repo ID (e.g., user/dataset)")
    parser.add_argument("--dataset_path", type=str, help="Direct path to dataset directory")
    args = parser.parse_args()

    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser()
    elif args.repo_id:
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    else:
        # Default
        dataset_path = Path.home() / ".cache/huggingface/lerobot/eternalmay33/lift_cube_3cams"

    inspect_lerobot_dataset(dataset_path)


if __name__ == "__main__":
    main()
