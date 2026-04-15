#!/usr/bin/env python3
"""Load calibration from JSON file to motor EEPROM.

Delegates to profiles.load(). Kept for backward compatibility.

Usage:
    python -m vbti.logic.servos.load_calibration --port /dev/ttyACM1 --robot_id frodeo-test
"""
from .profiles import load


def main(port: str = "/dev/ttyACM1", robot_id: str = "frodeo-test"):
    load(name=robot_id, port=port)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
