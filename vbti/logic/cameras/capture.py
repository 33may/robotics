"""Interactive camera capture CLI for prompt-stability experiment.

Usage:
    python capture.py --save_dir ./captures/cup_center
    python capture.py --save_dir ./captures/no_cup
    python capture.py --save_dir ./captures/cup_left --preset opencv

Controls:
    c     — capture snapshot from all cameras
    q     — quit and close cameras
    ESC   — quit and close cameras

Each capture saves one PNG per camera + metadata.json.
Filenames: {index:03d}_{camera}.png
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from vbti.logic.cameras.cameras import (
    CAMERA_PRESETS,
    init_cameras,
    capture_frames,
    stop_cameras,
)

CAMERA_ORDER = ["top", "left", "right", "gripper"]


def build_preview(frames, height=480, width=640):
    """2x2 grid with camera labels."""
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    positions = {"top": (0, 0), "left": (0, 1), "right": (1, 0), "gripper": (1, 1)}
    for name, (row, col) in positions.items():
        img = frames.get(name)
        if img is not None:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grid[row * height:(row + 1) * height, col * width:(col + 1) * width] = bgr
        cv2.putText(grid, name, (col * width + 10, row * height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return grid


def save_capture(frames, save_dir, index):
    """Save all camera frames + metadata for one capture."""
    for name, img in frames.items():
        path = save_dir / f"{index:03d}_{name}.png"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    meta = {
        "index": index,
        "timestamp": datetime.now().isoformat(),
        "cameras": list(frames.keys()),
        "resolution": [640, 480],
    }
    with open(save_dir / f"{index:03d}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Interactive multi-camera capture")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save captures (e.g. ./captures/cup_center)")
    parser.add_argument("--preset", type=str, default="realsense",
                        choices=list(CAMERA_PRESETS.keys()),
                        help="Camera preset to use")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Continue numbering from existing captures
    existing = list(save_dir.glob("*_meta.json"))
    capture_idx = len(existing)
    if capture_idx > 0:
        print(f"Found {capture_idx} existing captures, continuing from {capture_idx}")

    print(f"Using preset: {args.preset}")
    cam_config = CAMERA_PRESETS[args.preset]
    cameras = init_cameras(cam_config)

    found = [n for n in CAMERA_ORDER if n in cameras]
    print(f"\nReady: {found}")
    print(f"Saving to: {save_dir.resolve()}")
    print(f"\n  [c] capture  |  [q/ESC] quit")
    print(f"{'─' * 40}")

    try:
        while True:
            frames = capture_frames(cameras)
            preview = build_preview(frames)

            status = f"Captures: {capture_idx} | {save_dir.name} | [c] capture  [q] quit"
            cv2.putText(preview, status, (10, preview.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                save_capture(frames, save_dir, capture_idx)
                print(f"  [{capture_idx:03d}] Saved {len(frames)} cameras")
                capture_idx += 1
            elif key == ord("q") or key == 27:
                break

    finally:
        stop_cameras(cameras)
        cv2.destroyAllWindows()
        print(f"\nDone. {capture_idx} total captures in {save_dir.resolve()}")


if __name__ == "__main__":
    main()
