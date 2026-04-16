"""Replay LeRobot dataset episodes on the real SO-101 arm.

Used for validating sim-to-real conversion: replay sim-converted actions
on the real arm and visually verify the motions match the simulation.

Usage:
    python vbti/logic/dataset/replay_utils.py replay datasets/so101_v1_mix/sim/duck_cup_130eps 33 --port /dev/ttyACM0
    python vbti/logic/dataset/replay_utils.py replay datasets/so101_v1_mix/sim/duck_cup_130eps 33 --dry_run
    python vbti/logic/dataset/replay_utils.py show datasets/so101_v1_mix/sim/duck_cup_130eps 33
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd



JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_episode(dataset_path: str, episode_idx: int) -> tuple[np.ndarray, int]:
    """Load a single episode's actions from a LeRobot dataset.

    Returns:
        (actions array [n_frames, 6], fps)
    """
    import json
    from vbti.logic.dataset import resolve_dataset_path
    ds = resolve_dataset_path(dataset_path)
    info = json.load(open(ds / "meta" / "info.json"))
    fps = info["fps"]

    frames = []
    for pq in sorted((ds / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        ep = df[df["episode_index"] == episode_idx].sort_values("frame_index")
        if len(ep) > 0:
            frames.append(ep)

    if not frames:
        raise ValueError(f"Episode {episode_idx} not found in {dataset_path}")

    df = pd.concat(frames)
    actions = np.stack(df["action"].values)
    return actions, fps


def show(dataset_path: str, episode_idx: int = 0):
    """Print episode actions without connecting to robot.

    Shows first/last frames and joint ranges to preview what will be replayed.
    """
    actions, fps = _load_episode(dataset_path, episode_idx)
    print(f"\nEpisode {episode_idx}: {len(actions)} frames @ {fps} FPS ({len(actions)/fps:.1f}s)")
    print(f"\nFirst frame:  {np.round(actions[0], 2)}")
    print(f"Last frame:   {np.round(actions[-1], 2)}")
    print(f"\nJoint ranges:")
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = actions[:, i].min(), actions[:, i].max()
        print(f"  {name:15s} [{lo:7.2f}, {hi:7.2f}]")


def _go_to_start(robot, target: dict[str, float], steps: int = 100, duration: float = 3.0):
    """Slowly interpolate from current position to target over `duration` seconds.

    Reads current position, linearly interpolates in `steps` increments,
    sends each intermediate position to the robot.
    """
    current = robot.get_observation()
    current_pos = {name: current[f"{name}.pos"] for name in JOINT_NAMES}
    dt = duration / steps

    for step in range(1, steps + 1):
        t = step / steps
        interp = {}
        for name in JOINT_NAMES:
            interp[f"{name}.pos"] = current_pos[name] + t * (target[f"{name}.pos"] - current_pos[name])
        robot.send_action(interp)
        time.sleep(dt)


def _go_to_pos(robot, target: dict[str, float], steps: int = 100, duration: float = 3.0):
    """Slowly interpolate from current position to target over `duration` seconds.

    Reads current position, linearly interpolates in `steps` increments,
    sends each intermediate position to the robot.
    """
    current = robot.get_observation()
    current_pos = {name: current[f"{name}.pos"] for name in JOINT_NAMES}
    dt = duration / steps

    for step in range(1, steps + 1):
        t = step / steps
        interp = {}
        for name in JOINT_NAMES:
            interp[f"{name}.pos"] = current_pos[name] + t * (target[f"{name}.pos"] - current_pos[name])
        robot.send_action(interp)
        time.sleep(dt)

def replay(
    dataset_path: str,
    episode_idx: int = 0,
    port: str = "/dev/ttyACM0",
    robot_id: str | None = None,
    max_relative_target: float = 10.0,
    speed: float = 1.0,
    dry_run: bool = False,
    go_to_duration: float = 3.0,
    hold: bool = False,
):
    """Replay a dataset episode on the real robot.

    Args:
        dataset_path: Path to LeRobot dataset.
        episode_idx: Episode to replay.
        port: Serial port for the robot.
        max_relative_target: Max joint movement per step (safety clamp).
        speed: Playback speed multiplier (0.5 = half speed, 2.0 = double).
        dry_run: Print actions without sending to robot.
        go_to_duration: Seconds to interpolate to start position.
    """
    actions, fps = _load_episode(dataset_path, episode_idx)
    n_frames = len(actions)
    step_dt = 1.0 / (fps * speed)

    print(f"\nEpisode {episode_idx}: {n_frames} frames @ {fps} FPS (speed={speed}x → {step_dt*1000:.0f}ms/step)")
    print(f"Duration: {n_frames * step_dt:.1f}s")
    print(f"Safety: max_relative_target={max_relative_target}")

    # Build action dicts
    action_dicts = []
    for t in range(n_frames):
        action_dicts.append({f"{name}.pos": float(actions[t, i]) for i, name in enumerate(JOINT_NAMES)})

    if dry_run:
        print("\n[DRY RUN] First 5 actions:")
        for t in range(min(5, n_frames)):
            vals = "  ".join(f"{name}={actions[t,i]:7.2f}" for i, name in enumerate(JOINT_NAMES))
            print(f"  frame {t}: {vals}")
        print(f"  ... ({n_frames - 5} more frames)")
        return

    # Connect to robot
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from vbti.logic.servos.profiles import get_active_profile

    if robot_id is None:
        robot_id = get_active_profile()

    config = SOFollowerRobotConfig(
        port=port,
        id=robot_id,
        max_relative_target=max_relative_target,
    )
    robot = SOFollower(config)
    robot.connect()

    try:
        # Read current position
        obs = robot.get_observation()
        print(f"\nCurrent position:")
        for name in JOINT_NAMES:
            print(f"  {name:15s} {obs[f'{name}.pos']:7.2f}")

        # Go to start position
        print(f"\nMoving to start position over {go_to_duration}s...")
        _go_to_start(robot, action_dicts[0], duration=go_to_duration)

        input("\nPress Enter to start replay (Ctrl+C to abort)...")
        print("Replaying...")

        for t in range(n_frames):
            t0 = time.perf_counter()
            sent = robot.send_action(action_dicts[t])
            elapsed = time.perf_counter() - t0
            remaining = step_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

            # Print progress every 30 frames
            if t % 30 == 0:
                pct = 100 * t / n_frames
                print(f"  frame {t}/{n_frames} ({pct:.0f}%)", end="\r")

        print(f"\nReplay complete ({n_frames} frames)")

        if hold:
            print("Holding last frame — press 'q' or Ctrl+C to return to rest...")
            import select, sys
            while True:
                # Re-send last action to maintain position
                robot.send_action(action_dicts[-1])
                # Check for 'q' keypress (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        print("\nReleasing hold.")
                        break

    except KeyboardInterrupt:
        print("\n\nAborted!")
    finally:
        from vbti.logic.servos.rest import move_to_rest
        move_to_rest(robot)
        robot.disconnect()
        print("Robot disconnected.")


def goto(
    port: str = "/dev/ttyACM0",
    robot_id: str | None = None,
    duration: float = 3.0,
    shoulder_pan: float = 0.0,
    shoulder_lift: float = -95.0,
    elbow_flex: float = 95.0,
    wrist_flex: float = 45.0,
    wrist_roll: float = 0.0,
    gripper: float = 0.0,
    hold: bool = False,
):
    """Move robot to an arbitrary joint position.

    Usage:
        python replay_utils.py goto --wrist_roll=-70 --shoulder_pan=15
        python replay_utils.py goto --shoulder_lift=-27 --elbow_flex=-31 --duration=5
    """
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from vbti.logic.servos.profiles import get_active_profile

    if robot_id is None:
        robot_id = get_active_profile()

    target_vals = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    target = {f"{name}.pos": val for name, val in zip(JOINT_NAMES, target_vals)}

    print(f"Target position ({robot_id}):")
    for name, val in zip(JOINT_NAMES, target_vals):
        print(f"  {name:15s} {val:7.2f}")

    config = SOFollowerRobotConfig(port=port, id=robot_id)
    robot = SOFollower(config)
    robot.connect()

    try:
        _go_to_start(robot, target, duration=duration)
        print("Done.")
        if hold:
            print("Holding — press 'q' or Ctrl+C to release...")
            import select, sys
            while True:
                robot.send_action(target)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1) == 'q':
                        print("\nReleasing.")
                        break
    except KeyboardInterrupt:
        print("\nAborted!")
    finally:
        from vbti.logic.servos.rest import move_to_rest
        move_to_rest(robot)
        robot.disconnect()
        print("Robot disconnected.")




# At rest: [  0. -95.  95.  45.   0.   0.]

if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "replay": replay,
        "show":   show,
        "goto":   goto,
    })
