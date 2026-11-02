"""Replay LeRobot dataset episodes on the real SO-101 arm.

Used for validating sim-to-real conversion: replay sim-converted actions
on the real arm and visually verify the motions match the simulation.

Usage:
    python vbti/logic/dataset/replay_utils.py replay datasets/so101_v1_mix/sim/duck_cup_130eps 33 --port /dev/ttyACM0
    python vbti/logic/dataset/replay_utils.py replay datasets/so101_v1_mix/sim/duck_cup_130eps 33 --dry_run
    python vbti/logic/dataset/replay_utils.py show datasets/so101_v1_mix/sim/duck_cup_130eps 33
"""
import time

import numpy as np
import pandas as pd
import rerun as rr



JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_episode(dataset_path: str, episode_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a single episode's actions and states from a LeRobot dataset.

    Returns:
        (actions [n_frames, 6], states [n_frames, 6], fps)
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
    actions = np.stack(df["action"].to_list())
    states = np.stack(df["observation.state"].to_list()) if "observation.state" in df.columns else None
    return actions, states, fps


def show(dataset_path: str, episode_idx: int = 0):
    """Print episode actions without connecting to robot.

    Shows first/last frames and joint ranges to preview what will be replayed.
    """
    actions, _, fps = _load_episode(dataset_path, episode_idx)
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

def _plot_mpl(
    actions: np.ndarray,
    dataset_states: np.ndarray | None,
    robot_states: np.ndarray,
    save_path: str | None = None,
    joints: list[str] | None = None,
):
    import math
    import matplotlib.pyplot as plt
    from datetime import datetime

    active = joints if joints else JOINT_NAMES
    # validate names
    bad = [j for j in active if j not in JOINT_NAMES]
    if bad:
        raise ValueError(f"Unknown joints: {bad}. Valid: {JOINT_NAMES}")
    indices = [JOINT_NAMES.index(j) for j in active]

    frames = np.arange(len(actions))
    has_ds = dataset_states is not None

    # --- per-step print table (filtered joints only) ---
    hdr = f"{'joint':<16} {'action':>8}  {'dataset':>8}  {'robot':>8}"
    if has_ds:
        hdr += f"  {'Δact-rob':>8}  {'Δds-rob':>8}"
    else:
        hdr += f"  {'Δact-rob':>8}"
    sep = "-" * len(hdr)

    for t in range(len(actions)):
        print(f"\nstep {t:>4d}:  {hdr}")
        print(f"         {sep}")
        for i, name in zip(indices, active):
            a = actions[t, i]
            r = robot_states[t, i]
            d = float(dataset_states[t, i]) if has_ds else None
            if has_ds and d is not None:
                row = f"            {name:<16} {a:>8.2f}  {d:>8.2f}  {r:>8.2f}  {a-r:>+8.2f}  {d-r:>+8.2f}"
            else:
                row = f"            {name:<16} {a:>8.2f}  {'--':>8}  {r:>8.2f}  {a-r:>+8.2f}"
            print(row)

    # --- plots (filtered joints only) ---
    n = len(active)
    ncols = 2
    nrows = math.ceil(n / ncols)
    # squeeze=False guarantees axes is always 2D regardless of nrows/ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), sharex=True, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle("REAL ROBOT: action vs dataset state vs robot state", fontsize=13)

    for ax, name, i in zip(axes_flat, active, indices):
        a = actions[:, i]
        r = robot_states[:, i]
        d = dataset_states[:, i] if has_ds else None
        ax.plot(frames, a, label="action", linewidth=1.2)
        if d is not None:
            ax.plot(frames, d, label="dataset_state", linewidth=1.2, linestyle="--")
        ax.plot(frames, r, label="robot_state", linewidth=1.2, linestyle=":")
        ax.set_title(name)
        ax.set_ylabel("degrees")
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    if save_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/tmp/replay_joints_{ts}.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved: {save_path}")
    plt.show()


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
    plot: bool = True,
    mpl: bool = False,
    save_path: str | None = None,
    joints: list[str] | None = None,
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
    actions, dataset_states, fps = _load_episode(dataset_path, episode_idx)
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

        if plot:
            rr.init("vbti/replay", spawn=True)

        robot_states_log: list[list[float]] = []

        for t in range(n_frames):
            t0 = time.perf_counter()
            robot.send_action(action_dicts[t])
            obs = robot.get_observation()
            elapsed = time.perf_counter() - t0

            if mpl:
                robot_states_log.append([float(obs[f"{name}.pos"]) for name in JOINT_NAMES])

            if plot:
                rr.set_time("frame", sequence=t)
                for i, name in enumerate(JOINT_NAMES):
                    rr.log(f"joints/{name}/action", rr.Scalars(float(actions[t, i])))
                    rr.log(f"joints/{name}/robot_state", rr.Scalars(float(obs[f"{name}.pos"])))
                    if dataset_states is not None:
                        rr.log(f"joints/{name}/dataset_state", rr.Scalars(float(dataset_states[t, i])))

            remaining = step_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

            # Print progress every 30 frames
            if t % 30 == 0:
                pct = 100 * t / n_frames
                print(f"  frame {t}/{n_frames} ({pct:.0f}%)", end="\r")

        print(f"\nReplay complete ({n_frames} frames)")

        if mpl:
            _plot_mpl(actions, dataset_states, np.array(robot_states_log), save_path=save_path, joints=joints)

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
