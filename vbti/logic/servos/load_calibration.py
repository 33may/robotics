#!/usr/bin/env python3
"""Load calibration from JSON file to motor EEPROM.

Use after switching PCs or when motors lose their calibration.
Reads the lerobot calibration file and writes homing offsets + limits
to the servos, so `robot.connect()` won't ask for recalibration.

Usage:
    python vbti/logic/servos/load_calibration.py --port /dev/ttyACM1 --robot_id frodeo-test
    python vbti/logic/servos/load_calibration.py --port /dev/ttyACM1  # uses default frodeo-test
"""
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


def load(port: str = "/dev/ttyACM1", robot_id: str = "frodeo-test"):
    """Load calibration from JSON to motor EEPROM.

    Args:
        port: Serial port for the robot.
        robot_id: Robot ID matching the calibration file name.
    """
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)

    if not robot.calibration:
        print(f"No calibration file found for '{robot_id}'")
        print(f"Expected at: ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{robot_id}.json")
        return

    print(f"Loading calibration '{robot_id}' to motors on {port}...")
    robot.bus.connect()

    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    # Verify by reading back
    for motor, cal in robot.calibration.items():
        print(f"  {motor:15s}  homing_offset={cal.homing_offset:5d}  "
              f"range=[{cal.range_min}, {cal.range_max}]")

    robot.bus.disconnect()
    print("Done. Motors calibrated.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(load)
