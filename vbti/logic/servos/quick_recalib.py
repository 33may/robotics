#!/usr/bin/env python3
"""Quick recalibration: read current position, set it as the new zero.

Usage:
    # Step 1: Load frodeo-test calibration
    python -m vbti.logic.servos.profiles load frodeo-test

    # Step 2: Physically move each joint to where sim's 0 should be
    # (torque off, move by hand)

    # Step 3: Read positions and compute new calibration
    python vbti/logic/servos/quick_recalib.py --port=/dev/ttyACM1 \
        --old_profile=frodeo-test --new_profile=sim_accurate

    # Step 4: Load and test
    python -m vbti.logic.servos.profiles load sim_accurate
    python vbti/logic/dataset/replay_utils.py goto --robot_id=sim_accurate \
        --wrist_roll=0 --shoulder_lift=0 --elbow_flex=0 --wrist_flex=0 --gripper=0 --hold
"""
import json
import time
from pathlib import Path

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
CACHE_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"


def main(port: str = "/dev/ttyACM1", old_profile: str = "frodeo-test", new_profile: str = "sim_accurate"):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    # Load old calibration
    old_path = CACHE_DIR / f"{old_profile}.json"
    old_data = json.loads(old_path.read_text())
    print(f"Loaded old profile: {old_profile}")

    # Build bus WITH old calibration (so we can read in degrees)
    motors = {}
    calibration = {}
    for joint in JOINT_NAMES:
        j = old_data[joint]
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=j["id"], model="sts3215", norm_mode=norm)
        calibration[joint] = MotorCalibration(
            id=j["id"], drive_mode=j["drive_mode"],
            homing_offset=j["homing_offset"],
            range_min=j["range_min"], range_max=j["range_max"],
        )

    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    bus.connect()
    bus.disable_torque()

    # Write old calibration to EEPROM so reads are correct
    with bus.torque_disabled():
        bus.write_calibration(calibration)

    print("\n=== Move each joint to where 0° should be (sim zero pose) ===")
    print("Torque is OFF — move the robot by hand.")
    print("Press ENTER when the arm is in the sim zero pose...\n")
    input()

    # Read current positions in degrees (using old calibration)
    positions = bus.sync_read("Present_Position")
    print("Current positions (old calibration degrees):")
    for joint in JOINT_NAMES:
        print(f"  {joint:18s} = {positions[joint]:>8.2f}°")

    # The offset between old 0° and where the joint is now = the correction
    # New homing_offset = old_homing_offset + (position_in_old_degrees / degrees_per_tick)
    # Because: if the joint reads +5° in old calibration, we need to shift the
    # offset so that this physical position becomes 0° instead.
    DEGREES_PER_TICK = 360.0 / 4095.0

    new_data = {}
    print(f"\nNew calibration '{new_profile}':")
    print(f"  {'Joint':18s} {'Old offset':>10} {'Correction':>10} {'New offset':>10}")
    print("  " + "-" * 52)
    for joint in JOINT_NAMES:
        old = old_data[joint]
        old_offset = old["homing_offset"]

        # How many ticks to shift: current reading in degrees / degrees_per_tick
        current_deg = positions[joint]
        correction_ticks = int(round(current_deg / DEGREES_PER_TICK))

        new_offset = old_offset + correction_ticks
        print(f"  {joint:18s} {old_offset:>10} {correction_ticks:>+10} {new_offset:>10}")

        new_data[joint] = {
            "id": old["id"],
            "drive_mode": old["drive_mode"],
            "homing_offset": new_offset,
            "range_min": old["range_min"],
            "range_max": old["range_max"],
        }

    # Save
    new_path = CACHE_DIR / f"{new_profile}.json"
    new_path.write_text(json.dumps(new_data, indent=4) + "\n")
    print(f"\nSaved to {new_path}")

    # Load to motors
    ans = input("Write to motors now? [y/n] ").strip().lower()
    if ans == "y":
        new_calib = {}
        for joint in JOINT_NAMES:
            j = new_data[joint]
            new_calib[joint] = MotorCalibration(
                id=j["id"], drive_mode=j["drive_mode"],
                homing_offset=j["homing_offset"],
                range_min=j["range_min"], range_max=j["range_max"],
            )
        with bus.torque_disabled():
            bus.write_calibration(new_calib)
        print("Written to motors.")

    bus.disconnect()
    print("Done.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
