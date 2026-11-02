#!/usr/bin/env python3
"""Live view of cameras from a shared cameras.py preset.

Usage:
    python view_cameras.py              # default preset, 640x480@30
    python view_cameras.py --preset realsense_depth
    python view_cameras.py --fps 15     # custom default fps
    python view_cameras.py --width 320 --height 240  # lower res
"""
import argparse
import cv2

from cameras import CAMERA_PRESETS, DEFAULT_PRESET, build_grid_frame, capture_frames, init_cameras, stop_cameras


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(CAMERA_PRESETS))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    camera_config = CAMERA_PRESETS[args.preset]
    camera_names = list(camera_config)
    print(f"Using preset '{args.preset}': {camera_names}")
    cameras = init_cameras(camera_config, width=args.width, height=args.height, fps=args.fps)

    print("\nPress 'q' to quit.")

    try:
        while True:
            cv2.imshow(
                "Cameras",
                build_grid_frame(capture_frames(cameras), camera_names, width=args.width, height=args.height),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_cameras(cameras)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
