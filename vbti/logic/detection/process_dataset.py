"""Run duck/cup detection on all frames of a LeRobot v3.0 dataset.

Decodes video files with ffmpeg, runs Grounding DINO detection,
interpolates skipped frames, and saves results as a parquet file.

Usage:
    python -m vbti.logic.detection.process_dataset eternalmay33/01_black_gripper_front
    python -m vbti.logic.detection.process_dataset eternalmay33/01_black_gripper_front --stride 3
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from vbti.logic.dataset import resolve_dataset_path
from vbti.logic.detection.detect import create_detector, DEFAULT_MAX_AREA, GRIPPER_MAX_AREA

DEFAULT_CAMERAS = ["left", "right", "top", "gripper"]
OBJECTS = ["duck", "cup"]
FIELDS = ["cx", "cy", "conf"]      # interpolated per object per camera
BBOX_FIELDS = ["x1", "y1", "x2", "y2"]  # raw bbox (normalized, NaN at skipped frames)


# ------------------------------------------------------------------
# Video decoding
# ------------------------------------------------------------------

class VideoReader:
    """Stream frames from a video file via ffmpeg pipe.

    Supports reading specific frame ranges without loading the entire video
    into memory.  Frames are yielded lazily — only one raw frame buffer is
    held at a time.
    """

    def __init__(self, video_path: Path, width: int, height: int):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-v", "quiet", "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._pos = 0  # current frame index (next frame to read)

    def skip(self, n: int):
        """Skip *n* frames without decoding into numpy."""
        to_skip = n * self.frame_bytes
        while to_skip > 0:
            chunk = self.proc.stdout.read(min(to_skip, 1 << 20))
            if not chunk:
                break
            to_skip -= len(chunk)
        self._pos += n

    def read_one(self) -> np.ndarray | None:
        """Read a single frame. Returns None at EOF."""
        raw = self.proc.stdout.read(self.frame_bytes)
        if len(raw) < self.frame_bytes:
            return None
        self._pos += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 3
        ).copy()

    def read_range(self, start: int, count: int, stride: int = 1):
        """Read frames [start, start+count) returning only every *stride*-th.

        Always includes the first (start) and last (start+count-1) frames.
        Yields (local_index, frame) tuples where local_index is 0-based
        within the episode.
        """
        # Skip to start if needed
        if self._pos < start:
            self.skip(start - self._pos)

        last_idx = count - 1
        for i in range(count):
            is_detect = (i % stride == 0) or (i == last_idx)
            if is_detect:
                frame = self.read_one()
                if frame is None:
                    return
                yield i, frame
            else:
                # Skip this frame (just advance the pipe)
                self.skip(1)

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
            self.proc.wait()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------
# Dataset metadata helpers
# ------------------------------------------------------------------

def load_dataset_meta(ds_path: Path):
    """Load info.json and all episode metadata.

    Returns (info_dict, episodes_df).
    """
    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)

    ep_files = sorted((ds_path / "meta" / "episodes").rglob("*.parquet"))
    frames = []
    for f in ep_files:
        frames.append(pq.read_table(f).to_pandas())
    episodes = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    return info, episodes


def get_video_path(ds_path: Path, cam_key: str, chunk_idx: int, file_idx: int) -> Path:
    """Build path to a video file."""
    return ds_path / "videos" / cam_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


# ------------------------------------------------------------------
# Detection columns
# ------------------------------------------------------------------

def detection_columns(cameras: list[str]) -> list[str]:
    """Return the list of output column names."""
    cols = ["frame_index", "episode_index"]
    for cam in cameras:
        for obj in OBJECTS:
            for field in FIELDS:
                cols.append(f"{cam}_{obj}_{field}")
    return cols


def detection_row_from_result(det: dict, cam: str) -> dict:
    """Extract flat dict of {cam}_{obj}_{field} from a single detect() result."""
    row = {}
    for obj in OBJECTS:
        d = det[obj]
        cx, cy = d["center_norm"]
        row[f"{cam}_{obj}_cx"] = cx
        row[f"{cam}_{obj}_cy"] = cy
        row[f"{cam}_{obj}_conf"] = d["confidence"]
    return row


# ------------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------------

def apply_confidence_hold(
    ep_df: pd.DataFrame, cameras: list[str], threshold: float
) -> pd.DataFrame:
    """Hold last high-confidence (cx, cy) through low-confidence sampled frames.

    For every (camera, object) pair, walk the episode in time:
      - conf is NaN (stride-skipped): leave untouched (linear-interp fills later)
      - conf >= threshold: update the last-seen (cx, cy), keep values
      - conf <  threshold: replace (cx, cy) with last-seen good values, set conf := 0

    The conf := 0 signal lets the model know the position is a "held" (stale)
    reading rather than a fresh detection.  If no good frame has been seen yet
    at the start of the episode, low-conf frames stay NaN and are filled by
    the subsequent linear interpolation.
    """
    ep_df = ep_df.copy()
    for cam in cameras:
        for obj in OBJECTS:
            cx_col = f"{cam}_{obj}_cx"
            cy_col = f"{cam}_{obj}_cy"
            conf_col = f"{cam}_{obj}_conf"

            cx = ep_df[cx_col].values.astype(float).copy()
            cy = ep_df[cy_col].values.astype(float).copy()
            conf = ep_df[conf_col].values.astype(float).copy()

            last_cx = np.nan
            last_cy = np.nan
            for i in range(len(cx)):
                if np.isnan(conf[i]):
                    continue
                if conf[i] >= threshold:
                    last_cx = cx[i]
                    last_cy = cy[i]
                else:
                    cx[i] = last_cx
                    cy[i] = last_cy
                    conf[i] = 0.0

            ep_df[cx_col] = cx
            ep_df[cy_col] = cy
            ep_df[conf_col] = conf

    return ep_df


def interpolate_episode(ep_df: pd.DataFrame, cameras: list[str]) -> pd.DataFrame:
    """Fill NaN rows (skipped frames) with linear interpolation."""
    value_cols = []
    for cam in cameras:
        for obj in OBJECTS:
            for field in FIELDS:
                value_cols.append(f"{cam}_{obj}_{field}")

    ep_df = ep_df.copy()
    ep_df[value_cols] = ep_df[value_cols].interpolate(method="linear", limit_direction="both")
    ep_df[value_cols] = ep_df[value_cols].fillna(0.0)

    return ep_df


# ------------------------------------------------------------------
# Core processing
# ------------------------------------------------------------------

def process_dataset(
    dataset_path: str,
    stride: int = 3,
    batch_size: int = 4,
    cameras: list[str] | None = None,
    per_camera_stride: dict | None = None,
    confidence_threshold: float = 0.1,
    conf_hold_threshold: float | None = None,
    device: str = "cuda",
    root: str = None,
):
    """Run detection on a LeRobot dataset and save results parquet.

    Args:
        dataset_path: Repo ID or filesystem path.
        stride: Default per-frame stride; overridden per-camera by per_camera_stride.
        batch_size: Batch size for detection.
        cameras: Camera names to process (default: left, right, top, gripper).
        per_camera_stride: Optional {cam: stride} overriding the default stride.
        confidence_threshold: Detection confidence threshold at the model level.
        conf_hold_threshold: If set, apply hold-last-good logic at this threshold
            (any sampled frame with conf < threshold re-uses the prior good cx/cy).
        device: Torch device.
        root: Override dataset root path.
    """
    if cameras is None:
        cameras = list(DEFAULT_CAMERAS)

    ds_path = resolve_dataset_path(dataset_path, root=root)
    output_path = ds_path / "detection_results.parquet"

    print(f"[detection] Dataset: {ds_path}")
    print(f"[detection] Output:  {output_path}")
    print(f"[detection] Cameras: {cameras}, stride={stride}, batch_size={batch_size}")

    # Load metadata
    info, episodes = load_dataset_meta(ds_path)
    fps = info["fps"]
    total_episodes = len(episodes)
    total_frames = episodes["length"].sum()

    # Resolve video resolution per camera
    cam_resolution = {}
    for cam in cameras:
        cam_key = f"observation.images.{cam}"
        feat = info["features"].get(cam_key)
        if feat is None:
            raise ValueError(f"Camera '{cam}' not found in dataset features. "
                             f"Available: {[k for k in info['features'] if 'images' in k]}")
        cam_resolution[cam] = (
            feat["info"]["video.width"],
            feat["info"]["video.height"],
        )

    res_str = ", ".join(f"{c}={cam_resolution[c][0]}x{cam_resolution[c][1]}" for c in cameras)
    print(f"[detection] {total_episodes} episodes, {total_frames} total frames, "
          f"{res_str} @ {fps} fps")

    # Resume support
    done_episodes = set()
    existing_rows = []
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            done_episodes = set(existing_df["episode_index"].unique())
            existing_rows.append(existing_df)
            print(f"[detection] Resuming: {len(done_episodes)} episodes already done")
        except Exception as e:
            print(f"[detection] Warning: could not read existing output, starting fresh: {e}")

    remaining_eps = episodes[~episodes["episode_index"].isin(done_episodes)]
    if remaining_eps.empty:
        print("[detection] All episodes already processed!")
        return output_path

    remaining_frames = remaining_eps["length"].sum()
    print(f"[detection] Processing {len(remaining_eps)} episodes ({remaining_frames} frames)")

    # Initialize detector (ONNX if available, falls back to PyTorch)
    detector = create_detector(
        device=device,
        confidence_threshold=confidence_threshold,
    )

    cam_key_map = {cam: f"observation.images.{cam}" for cam in cameras}

    # Pre-allocate result storage
    episode_results = {}
    for _, ep_row in remaining_eps.iterrows():
        ep_idx = int(ep_row["episode_index"])
        n_frames = int(ep_row["length"])
        episode_results[ep_idx] = {
            "frame_index": np.arange(n_frames),
            "episode_index": np.full(n_frames, ep_idx, dtype=np.int64),
        }
        for cam in cameras:
            for obj in OBJECTS:
                for field in FIELDS + BBOX_FIELDS:
                    episode_results[ep_idx][f"{cam}_{obj}_{field}"] = np.full(n_frames, np.nan)

    t_start = time.perf_counter()
    frames_detected = 0

    def _flush_batch(batch_imgs, batch_meta, cam, ep_results):
        """Run detection on accumulated batch and store results."""
        nonlocal frames_detected
        if not batch_imgs:
            return
        max_area = GRIPPER_MAX_AREA if cam == "gripper" else DEFAULT_MAX_AREA
        results = detector.detect_batch(batch_imgs, max_area=max_area)
        for (ep_idx, fi), img, det in zip(batch_meta, batch_imgs, results):
            ih, iw = img.shape[:2]
            for obj_name in OBJECTS:
                cx, cy = det[obj_name]["center_norm"]
                ep_results[ep_idx][f"{cam}_{obj_name}_cx"][fi] = cx
                ep_results[ep_idx][f"{cam}_{obj_name}_cy"][fi] = cy
                ep_results[ep_idx][f"{cam}_{obj_name}_conf"][fi] = det[obj_name]["confidence"]
                x1, y1, x2, y2 = det[obj_name]["bbox"]
                ep_results[ep_idx][f"{cam}_{obj_name}_x1"][fi] = x1 / iw
                ep_results[ep_idx][f"{cam}_{obj_name}_y1"][fi] = y1 / ih
                ep_results[ep_idx][f"{cam}_{obj_name}_x2"][fi] = x2 / iw
                ep_results[ep_idx][f"{cam}_{obj_name}_y2"][fi] = y2 / ih
        frames_detected += len(batch_imgs)

    for cam in cameras:
        cam_key = cam_key_map[cam]
        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        from_ts_col = f"videos/{cam_key}/from_timestamp"

        max_area = GRIPPER_MAX_AREA if cam == "gripper" else DEFAULT_MAX_AREA
        cam_stride = (per_camera_stride or {}).get(cam, stride)
        print(f"\n[detection] Camera: {cam} (max_area={max_area:.0%}, stride={cam_stride})")

        video_groups = remaining_eps.groupby([chunk_col, file_col])

        pbar = tqdm(
            total=remaining_frames,
            desc=f"  {cam}",
            unit="frame",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for (chunk_idx, file_idx), group_eps in video_groups:
            video_path = get_video_path(ds_path, cam_key, int(chunk_idx), int(file_idx))
            if not video_path.exists():
                print(f"  Warning: video not found: {video_path}")
                for _, ep_row in group_eps.iterrows():
                    ep_idx = int(ep_row["episode_index"])
                    n_frames = int(ep_row["length"])
                    for obj_name in OBJECTS:
                        for field in FIELDS:
                            episode_results[ep_idx][f"{cam}_{obj_name}_{field}"][:] = 0.0
                    pbar.update(n_frames)
                continue

            group_eps = group_eps.sort_values(from_ts_col)

            batch_imgs = []
            batch_meta = []
            vid_w, vid_h = cam_resolution[cam]

            with VideoReader(video_path, vid_w, vid_h) as reader:
                for _, ep_row in group_eps.iterrows():
                    ep_idx = int(ep_row["episode_index"])
                    n_ep_frames = int(ep_row["length"])
                    from_ts = float(ep_row[from_ts_col])
                    start_frame = round(from_ts * fps)

                    for local_idx, frame in reader.read_range(start_frame, n_ep_frames, cam_stride):
                        batch_imgs.append(frame)
                        batch_meta.append((ep_idx, local_idx))

                        if len(batch_imgs) >= batch_size:
                            _flush_batch(batch_imgs, batch_meta, cam, episode_results)
                            batch_imgs = []
                            batch_meta = []

                    pbar.update(n_ep_frames)

            _flush_batch(batch_imgs, batch_meta, cam, episode_results)
            batch_imgs = []
            batch_meta = []

        pbar.close()

    # Hold-then-interpolate
    if conf_hold_threshold is not None:
        print(f"\n[detection] Applying confidence hold (threshold={conf_hold_threshold:.2f})...")
    print("\n[detection] Interpolating skipped frames...")
    new_rows = []
    for ep_idx in tqdm(sorted(episode_results.keys()), desc="  interpolate", unit="ep"):
        ep_df = pd.DataFrame(episode_results[ep_idx])
        if conf_hold_threshold is not None:
            ep_df = apply_confidence_hold(ep_df, cameras, conf_hold_threshold)
        ep_df = interpolate_episode(ep_df, cameras)
        new_rows.append(ep_df)

    new_df = pd.concat(new_rows, ignore_index=True)

    # Merge with existing
    if existing_rows:
        final_df = pd.concat([*existing_rows, new_df], ignore_index=True)
        final_df = final_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    else:
        final_df = new_df

    # Cast types
    final_df["frame_index"] = final_df["frame_index"].astype(np.int64)
    final_df["episode_index"] = final_df["episode_index"].astype(np.int64)
    for col in final_df.columns:
        if col not in ("frame_index", "episode_index"):
            final_df[col] = final_df[col].astype(np.float32)

    # Save
    final_df.to_parquet(output_path, index=False)
    elapsed = time.perf_counter() - t_start
    print(f"\n[detection] Done! {len(final_df)} rows -> {output_path}")
    print(f"[detection] {elapsed:.1f}s ({frames_detected} frames detected, "
          f"{len(final_df)} rows output)")

    return output_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run duck/cup detection on a LeRobot dataset."
    )
    parser.add_argument("dataset", help="Dataset repo ID or path")
    parser.add_argument("--stride", type=int, default=30,
                        help="Default per-frame stride for detection (default: 30)")
    parser.add_argument("--gripper-stride", type=int, default=10,
                        help="Stride override for gripper camera (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Detection batch size (default: 4)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera names (default: left right top gripper)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Detection confidence threshold (default: 0.1)")
    parser.add_argument("--conf-hold-threshold", type=float, default=0.15,
                        help="Hold last-good cx/cy when detection conf < this value "
                             "(default: 0.15; pass 0 to disable)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--root", default=None,
                        help="Override dataset root path")

    args = parser.parse_args()

    per_cam_stride = {"gripper": args.gripper_stride}
    hold = args.conf_hold_threshold if args.conf_hold_threshold > 0 else None

    process_dataset(
        dataset_path=args.dataset,
        stride=args.stride,
        batch_size=args.batch_size,
        cameras=args.cameras,
        per_camera_stride=per_cam_stride,
        confidence_threshold=args.threshold,
        conf_hold_threshold=hold,
        device=args.device,
        root=args.root,
    )


if __name__ == "__main__":
    main()
