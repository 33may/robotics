#!/usr/bin/env python3
"""Live view of all connected RealSense cameras in a single window.

Usage:
    python view_cameras.py              # all cameras, 640x480@30
    python view_cameras.py --fps 15     # custom fps
    python view_cameras.py --width 320 --height 240  # lower res
"""
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for d in devices:
        serials.append(d.get_info(rs.camera_info.serial_number))

    if not serials:
        print("No RealSense cameras found.")
        return

    print(f"Found {len(serials)} cameras: {serials}")

    pipelines = []
    for sn in serials:
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipe = rs.pipeline()
        try:
            pipe.start(cfg)
            pipelines.append((sn, pipe))
            print(f"  {sn}: streaming {args.width}x{args.height}@{args.fps}")
        except Exception as e:
            print(f"  {sn}: FAILED — {e}")

    if not pipelines:
        print("No cameras started.")
        return

    print("\nPress 'q' to quit.")

    try:
        while True:
            frames = []
            for sn, pipe in pipelines:
                try:
                    fs = pipe.wait_for_frames(timeout_ms=1000)
                    color = fs.get_color_frame()
                    if color:
                        img = np.asanyarray(color.get_data())
                        # Add serial number label
                        cv2.putText(img, sn[-4:], (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        frames.append(img)
                except Exception:
                    frames.append(np.zeros((args.height, args.width, 3), dtype=np.uint8))

            if frames:
                # Arrange in 2xN grid
                cols = 2
                rows = (len(frames) + cols - 1) // cols
                while len(frames) < rows * cols:
                    frames.append(np.zeros_like(frames[0]))
                grid_rows = []
                for r in range(rows):
                    grid_rows.append(np.hstack(frames[r * cols:(r + 1) * cols]))
                grid = np.vstack(grid_rows)
                cv2.imshow("RealSense Cameras", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, pipe in pipelines:
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
