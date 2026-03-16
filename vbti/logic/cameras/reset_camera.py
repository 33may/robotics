#!/usr/bin/env python3
"""Hardware reset a RealSense camera by serial number, or all cameras.

Usage:
    python reset_camera.py                  # reset all cameras
    python reset_camera.py 128422270260     # reset specific camera
"""
import sys
import time
import pyrealsense2 as rs


def main():
    target_sn = sys.argv[1] if len(sys.argv) > 1 else None

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("No RealSense cameras found.")
        return

    for d in devices:
        sn = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        name = d.get_info(rs.camera_info.name)

        if target_sn and sn != target_sn:
            continue

        print(f"{sn} ({name}, fw={fw}) — resetting...")
        d.hardware_reset()

    wait = 5
    print(f"\nWaiting {wait}s for cameras to reconnect...")
    time.sleep(wait)

    # Verify
    ctx2 = rs.context()
    devices2 = ctx2.query_devices()
    print(f"\n{len(devices2)} camera(s) back online:")
    for d in devices2:
        sn = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        print(f"  {sn}: fw={fw}")


if __name__ == "__main__":
    main()
