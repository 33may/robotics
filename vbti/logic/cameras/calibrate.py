"""Live camera calibration overlay — align physical cameras to reference images.

Usage:
    python calibrate.py --ref_dir /tmp/currect_cams/
    python calibrate.py --ref_dir /tmp/currect_cams/ --camera top
    python calibrate.py --ref_dir /tmp/currect_cams/ --preset opencv

Controls:
    n       — next camera
    p       — previous camera
    q / ESC — quit
    trackbar — adjust reference overlay opacity
"""

import argparse
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
WINDOW = "Calibrate"


def load_references(ref_dir):
    """Load reference images, auto-detecting naming convention."""
    ref_dir = Path(ref_dir)
    refs = {}
    for name in CAMERA_ORDER:
        # Try both formats: {name}.png and *_{name}.png (capture script output)
        candidates = [ref_dir / f"{name}.png"] + sorted(ref_dir.glob(f"*_{name}.png"))
        for path in candidates:
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    refs[name] = img
                    break
    return refs


def main():
    parser = argparse.ArgumentParser(description="Live camera calibration overlay")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Directory with reference images")
    parser.add_argument("--camera", type=str, default=None,
                        choices=CAMERA_ORDER,
                        help="Single camera to calibrate (default: cycle all)")
    parser.add_argument("--preset", type=str, default="realsense",
                        choices=list(CAMERA_PRESETS.keys()),
                        help="Camera preset to use")
    args = parser.parse_args()

    refs = load_references(args.ref_dir)
    if not refs:
        print(f"No reference images found in {args.ref_dir}")
        return

    print(f"Loaded references: {list(refs.keys())}")

    cam_config = CAMERA_PRESETS[args.preset]
    cameras = init_cameras(cam_config)

    available = [n for n in CAMERA_ORDER if n in cameras and n in refs]
    if args.camera:
        if args.camera not in available:
            print(f"Camera '{args.camera}' not available. Have: {available}")
            stop_cameras(cameras)
            return
        available = [args.camera]

    print(f"Calibrating: {available}")
    print(f"\n  [n] next  |  [p] prev  |  [q/ESC] quit  |  slider: opacity")
    print(f"{'─' * 50}")

    cam_idx = 0
    toggle_mode = False  # False = slider blend, True = full ref/live toggle
    show_ref = True
    last_slider = 40
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("ref %", WINDOW, 40, 100, lambda _: None)

    try:
        while True:
            name = available[cam_idx]
            frames = capture_frames(cameras)
            live = frames.get(name)
            if live is None:
                continue

            live_bgr = cv2.cvtColor(live, cv2.COLOR_RGB2BGR)
            ref = refs[name]

            # Resize ref to match live if needed
            h, w = live_bgr.shape[:2]
            if ref.shape[:2] != (h, w):
                ref = cv2.resize(ref, (w, h))

            slider_val = cv2.getTrackbarPos("ref %", WINDOW)
            if slider_val != last_slider:
                toggle_mode = False
                last_slider = slider_val

            if toggle_mode:
                blended = ref.copy() if show_ref else live_bgr.copy()
                mode = "REF" if show_ref else "LIVE"
            else:
                alpha = slider_val / 100.0
                blended = cv2.addWeighted(live_bgr, 1.0 - alpha, ref, alpha, 0)
                mode = f"blend {slider_val}%"

            # HUD
            label = f"{name} [{cam_idx + 1}/{len(available)}]  [{mode}]  SPACE=toggle"
            cv2.putText(blended, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW, blended)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not toggle_mode:
                    toggle_mode = True
                    show_ref = True
                else:
                    show_ref = not show_ref
            elif key == ord("n"):
                cam_idx = (cam_idx + 1) % len(available)
            elif key == ord("p"):
                cam_idx = (cam_idx - 1) % len(available)
            elif key == ord("q") or key == 27:
                break

    finally:
        stop_cameras(cameras)
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
