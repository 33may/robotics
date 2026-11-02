#!/usr/bin/env python3
"""Check USB speed and topology for all connected RealSense cameras.

Usage: python check_usb.py
"""
import subprocess
import pyrealsense2 as rs


def get_usb_speeds():
    """Parse lsusb -t to find camera USB speeds."""
    result = subprocess.run(["lsusb", "-t"], capture_output=True, text=True)
    speeds = {}
    for line in result.stdout.splitlines():
        if "Video" in line or "uvcvideo" in line:
            # Extract speed (e.g., 5000M or 480M)
            parts = line.strip().split(",")
            speed = parts[-1].strip() if parts else "unknown"
            speeds[line.strip()] = speed
    return result.stdout


def main():
    # Check usbfs memory
    try:
        with open("/sys/module/usbcore/parameters/usbfs_memory_mb") as f:
            usbfs = f.read().strip()
    except FileNotFoundError:
        usbfs = "unknown"

    print(f"usbfs_memory_mb: {usbfs} (recommended: 1000+)\n")

    # List cameras with USB info
    ctx = rs.context()
    devices = ctx.query_devices()

    print(f"{'Serial':<16} {'Name':<24} {'FW':<12} {'USB':>5}")
    print("-" * 60)
    for d in devices:
        sn = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        fw = d.get_info(rs.camera_info.firmware_version)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        status = "OK" if float(usb) >= 3.0 else "SLOW (USB 2.0)"
        print(f"{sn:<16} {name:<24} {fw:<12} {usb:>5}  {status}")

    # Show USB topology
    print(f"\n--- USB Topology ---")
    topo = get_usb_speeds()
    for line in topo.splitlines():
        if "Bus 002" in line or "5000M" in line or "Video" in line or "Hub" in line:
            print(line)


if __name__ == "__main__":
    main()
