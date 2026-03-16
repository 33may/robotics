"""
Convert leisaac HDF5 datasets to LeRobot v3.0 format.

Handles:
  - Joint position: radians → sim degrees → real degrees → [-100, 100] normalized
  - Images: raw uint8 arrays → AV1 mp4 video
  - Arbitrary number of cameras (auto-discovered from HDF5 obs keys)
  - Skips empty episodes (no actions)

Usage:
    python vbti/logic/dataset/convert_utils.py discover /path/to/dataset.hdf5
    python vbti/logic/dataset/convert_utils.py convert /path/to/dataset.hdf5 may33/my_dataset "Pick up the duck" --camera_map '{"cam_top":"top","cam_left":"left"}'
    python vbti/logic/dataset/convert_utils.py roundtrip_test
"""
import re
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


# ── Joint limit tables ──────────────────────────────────────────
# Sim: from robot USD (physics:lowerLimit / physics:upperLimit)
# Real: from servo calibration (range_min/range_max → degrees)

SIM_LIMITS_DEG = [
    (-110.0, 110.0),   # shoulder_pan
    (-100.0, 100.0),   # shoulder_lift
    (-100.0,  90.0),   # elbow_flex
    ( -95.0,  95.0),   # wrist_flex
    (-160.0, 160.0),   # wrist_roll
    ( -10.0, 100.0),   # gripper
]

REAL_LIMITS_DEG = [
    (-114.5, 125.5),   # shoulder_pan
    (-109.9, 101.0),   # shoulder_lift
    (-106.0,  89.5),   # elbow_flex
    (-103.7, 103.8),   # wrist_flex
    (-171.3, 165.2),   # wrist_roll
    (  -2.3, 110.4),   # gripper
]

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Gripper (index 5) uses RANGE_0_100 [0, 100] on real robot,
# body joints (0-4) use RANGE_M100_100 [-100, 100].
GRIPPER_IDX = 5


# ── Conversion functions ────────────────────────────────────────


def sim_to_normalized(joint_pos_rad: np.ndarray) -> np.ndarray:
    """Convert sim joint positions from radians to normalized.

    Pipeline per joint:
        sim_radians → sim_degrees → real_degrees → normalized
    Body joints (0-4): [-100, 100] (RANGE_M100_100)
    Gripper (5):       [0, 100]    (RANGE_0_100)
    """
    result = np.empty_like(joint_pos_rad)
    for i in range(joint_pos_rad.shape[-1]):
        sim_lo, sim_hi = SIM_LIMITS_DEG[i]
        real_lo, real_hi = REAL_LIMITS_DEG[i]

        # Step 1: radians → sim degrees
        sim_deg = joint_pos_rad[..., i] * 180.0 / np.pi

        # Step 2: sim degrees → real degrees (linear remap)
        t = (sim_deg - sim_lo) / (sim_hi - sim_lo)
        real_deg = t * (real_hi - real_lo) + real_lo

        # Step 3: real degrees → normalized
        frac = (real_deg - real_lo) / (real_hi - real_lo)
        if i == GRIPPER_IDX:
            result[..., i] = frac * 100          # [0, 100]
        else:
            result[..., i] = frac * 200 - 100    # [-100, 100]

    return result


def normalized_to_sim(joint_pos_norm: np.ndarray) -> np.ndarray:
    """Convert normalized back to sim radians.

    Pipeline per joint:
        normalized → real_degrees → sim_degrees → sim_radians
    Body joints (0-4): [-100, 100] (RANGE_M100_100)
    Gripper (5):       [0, 100]    (RANGE_0_100)
    """
    result = np.empty_like(joint_pos_norm)
    for i in range(joint_pos_norm.shape[-1]):
        sim_lo, sim_hi = SIM_LIMITS_DEG[i]
        real_lo, real_hi = REAL_LIMITS_DEG[i]

        # Step 1: normalized → real degrees
        if i == GRIPPER_IDX:
            frac = joint_pos_norm[..., i] / 100               # [0, 100] → [0, 1]
        else:
            frac = (joint_pos_norm[..., i] + 100) / 200       # [-100, 100] → [0, 1]
        real_deg = frac * (real_hi - real_lo) + real_lo

        # Step 2: real degrees → sim degrees
        t = (real_deg - real_lo) / (real_hi - real_lo)
        sim_deg = t * (sim_hi - sim_lo) + sim_lo

        # Step 3: sim degrees → radians
        result[..., i] = sim_deg * np.pi / 180.0

    return result


# ── Feature builders ────────────────────────────────────────────


def _video_feature(height: int = 480, width: int = 640, fps: float = 30.0) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def build_features(camera_map: dict[str, str], fps: float = 30.0) -> dict:
    """Build LeRobot features dict from a camera mapping.

    Args:
        camera_map: {hdf5_key: lerobot_name}, e.g. {"cam_top": "top", "wrist": "gripper"}
        fps: frames per second for video encoding
    """
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
    }
    for hdf5_key, lerobot_name in camera_map.items():
        features[f"observation.images.{lerobot_name}"] = _video_feature(fps=fps)
    return features


# ── Camera discovery ────────────────────────────────────────────


def discover_cameras(hdf5_path: str | Path) -> list[str]:
    """Auto-discover RGB camera keys from the first valid episode."""
    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["data"].keys():
            ep = f["data"][ep_name]
            if "actions" not in ep or "obs" not in ep:
                continue
            obs = ep["obs"]
            cameras = []
            for key in obs.keys():
                ds = obs[key]
                if hasattr(ds, "shape") and len(ds.shape) == 4 and ds.shape[-1] == 3:
                    cameras.append(key)
            return sorted(cameras)
    return []


# ── Main conversion ─────────────────────────────────────────────


def convert(
    hdf5_path: str | Path,
    repo_id: str,
    task: str,
    camera_map: dict[str, str],
    root: str | Path | None = None,
    fps: int = 30,
    robot_type: str = "so101_follower",
    min_episode_frames: int = 10,
    skip_first_n: int = 0,
    push_to_hub: bool = False,
) -> Path:
    """Convert an HDF5 leisaac dataset to LeRobot v3.0 format.

    Args:
        hdf5_path: Path to source HDF5 file.
        repo_id: HuggingFace repo ID (e.g. "may33/so101_sim_v1").
        task: Task description string.
        camera_map: Mapping from HDF5 obs keys to LeRobot camera names.
                    e.g. {"cam_top": "top", "cam_left": "left", "wrist": "gripper"}
        root: Output directory. Defaults to HF_LEROBOT_HOME/repo_id.
        fps: Dataset frame rate.
        robot_type: Robot identifier string.
        min_episode_frames: Skip episodes shorter than this.
        skip_first_n: Skip first N frames of each episode.
        push_to_hub: Push to HuggingFace Hub after conversion.

    Returns:
        Path to the created dataset.
    """
    hdf5_path = Path(hdf5_path)
    features = build_features(camera_map, fps=fps)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=root,
    )

    saved = 0
    skipped_empty = 0
    skipped_short = 0

    with h5py.File(hdf5_path, "r") as f:
        # Sort episodes numerically
        ep_names = sorted(
            f["data"].keys(),
            key=lambda n: int(re.findall(r"\d+", n)[-1]) if re.findall(r"\d+", n) else 0,
        )
        print(f"Found {len(ep_names)} episodes in {hdf5_path.name}")

        for ep_name in tqdm(ep_names, desc="Converting episodes"):
            ep = f["data"][ep_name]

            # Skip empty episodes
            if "actions" not in ep:
                skipped_empty += 1
                continue

            actions = np.array(ep["actions"])
            states = np.array(ep["obs/joint_pos"])
            n_frames = actions.shape[0]

            if n_frames < min_episode_frames:
                skipped_short += 1
                continue

            # Convert joint positions
            actions_norm = sim_to_normalized(actions)
            states_norm = sim_to_normalized(states)

            # Load all camera data for this episode
            camera_data = {}
            for hdf5_key in camera_map:
                camera_data[hdf5_key] = np.array(ep[f"obs/{hdf5_key}"])

            # Add frames
            for t in range(skip_first_n, n_frames):
                frame = {
                    "task": task,
                    "action": actions_norm[t].astype(np.float32),
                    "observation.state": states_norm[t].astype(np.float32),
                }
                for hdf5_key, lerobot_name in camera_map.items():
                    frame[f"observation.images.{lerobot_name}"] = camera_data[hdf5_key][t]

                dataset.add_frame(frame)

            dataset.save_episode()
            saved += 1

    print(f"\nConversion complete:")
    print(f"  Saved: {saved} episodes")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (short): {skipped_short}")
    print(f"  Output: {dataset.root}")

    if push_to_hub:
        dataset.push_to_hub()
        print(f"  Pushed to hub: {repo_id}")

    return dataset.root


def roundtrip_test():
    """Verify sim_to_normalized ↔ normalized_to_sim roundtrip accuracy."""
    test_rad = np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.3]])
    normed = sim_to_normalized(test_rad)
    back = normalized_to_sim(normed)
    err = np.max(np.abs(test_rad - back))
    print(f"Input (rad):    {test_rad[0]}")
    print(f"Normalized:     {normed[0]}")
    print(f"Roundtrip (rad):{back[0]}")
    print(f"Max error:      {err:.2e}")
    assert err < 1e-10, f"Roundtrip error too large: {err}"
    print("PASS")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "convert":        convert,
        "discover":       discover_cameras,
        "roundtrip_test": roundtrip_test,
    })
