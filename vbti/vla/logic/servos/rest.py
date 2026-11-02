#!/usr/bin/env python3
"""Move SO-101 arm to resting position.

Usage:
    python vbti/logic/servos/rest.py
    python vbti/logic/servos/rest.py --port=/dev/ttyACM1
    python vbti/logic/servos/rest.py --speed=5.0
"""

import time
import numpy as np

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Safe resting pose in degrees — arm tucked back, gripper open
REST_POSITION = {
    "shoulder_pan":  0.0,
    "shoulder_lift": -95.0,
    "elbow_flex":    95.0,
    "wrist_flex":    45.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}


def move_to_rest(robot, speed_deg_per_step: float = 3.0, fps: int = 30):
    """Move robot smoothly to REST_POSITION.

    Interpolates from current position so the arm doesn't snap.
    Blocks until complete.

    Args:
        robot: connected SO101Follower instance
        speed_deg_per_step: max degrees per joint per step
        fps: control rate during movement
    """
    obs = robot.get_observation()
    current = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
    target = np.array([REST_POSITION[n] for n in JOINT_NAMES])
    step_dt = 1.0 / fps

    print("Moving to rest position...")
    while True:
        diff = target - current
        if np.abs(diff).max() < 0.5:
            break
        step = np.clip(diff, -speed_deg_per_step, speed_deg_per_step)
        current = current + step
        action_dict = {f"{name}.pos": float(current[j]) for j, name in enumerate(JOINT_NAMES)}
        robot.send_action(action_dict)
        time.sleep(step_dt)

    print(f"  At rest: {np.round(target, 1)}")


def rest(port: str = "/dev/ttyACM0", robot_id: str | None = None,
         speed: float = 3.0, fps: int = 30):
    """Connect to robot and move to resting position.

    Args:
        port: serial port
        robot_id: robot identifier (default: active profile from registry)
        speed: max degrees per joint per step
        fps: control rate
    """
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from vbti.logic.servos.profiles import get_active_profile

    if robot_id is None:
        robot_id = get_active_profile()

    config = SOFollowerRobotConfig(port=port, id=robot_id)
    robot = SOFollower(config)
    robot.connect()

    try:
        move_to_rest(robot, speed_deg_per_step=speed, fps=fps)
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(rest)
