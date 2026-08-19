"""Tier 1 — validated direct joint move (MAY-184).

The workhorse: straight joint interpolation q_now -> q_goal, exactly what the
robot's moveJ will execute, judged config-by-config by the world. No planning,
no search — either this exact motion is provably clear, or the caller escalates
to tier 2 (sampling search). Measured ~75% of viewpoint moves land here.

Usage (from repo root, robo env active):
    p inspection/motion/direct.py          # headless: plan to a pose over the table
    p inspection/motion/direct.py --demo   # meshcat: ghost-replay the chosen path
"""

import argparse
import sys
import time

import numpy as np

from inspection.cell.world import RobotCell, Q_PARK, DEFAULT_STEP
from inspection.motion.ik import UR5eIK


def direct_move(world, q_from, q_goal, step=DEFAULT_STEP):
    """Validate the straight joint move. Returns the waypoint pair or None."""
    ok, _ = world.path_valid([q_from, q_goal], step=step)
    return [np.asarray(q_from), np.asarray(q_goal)] if ok else None


def plan_to_pose(world, ik, q_from, T_flange, step=DEFAULT_STEP):
    """Tier-1 planning to a flange pose: enumerate IK branches, drop colliding
    endpoints, try direct moves nearest-branch-first.

    Returns (path, info): path is [q_from, q_goal] or None; info is a dict
    with the branch bookkeeping (how many enumerated / endpoint-free / tried).
    """
    branches = ik.branches(T_flange)
    info = {"enumerated": len(branches), "endpoint_free": 0, "tried": 0}
    if len(branches) == 0:
        return None, info
    free = [q for q in branches if not world.is_colliding(q)]
    info["endpoint_free"] = len(free)
    # nearest branch first: shortest joint-space move wins
    free.sort(key=lambda q: np.abs(q - q_from).sum())
    for q_goal in free:
        info["tried"] += 1
        path = direct_move(world, q_from, q_goal, step=step)
        if path is not None:
            return path, info
    return None, info


def look_down_pose(x, y, z, yaw=0.0):
    """Flange pose above base-frame point (x,y,z), tool face pointing DOWN
    (flange +X = -Z base), yaw spins the tool about the vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, 0] = [0.0, 0.0, -1.0]           # +X (tool face) straight down
    T[:3, 1] = [-s, c, 0.0]
    T[:3, 2] = [c, s, 0.0]
    T[:3, 3] = [x, y, z]
    return T


def sample_viewpoint(rng):
    """Random hover pose over the table: position + view direction tilted up
    to 40 deg off vertical, random roll about the view axis."""
    pos = np.array([rng.uniform(-0.12, 0.32),      # over the table, base frame
                    rng.uniform(-0.72, 0.12),
                    rng.uniform(0.10, 0.45)])
    tilt = rng.uniform(0.0, np.deg2rad(40))
    azim = rng.uniform(-np.pi, np.pi)
    d = np.array([np.sin(tilt) * np.cos(azim),      # view dir, mostly downward
                  np.sin(tilt) * np.sin(azim),
                  -np.cos(tilt)])
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    y = np.cross(d, ref); y /= np.linalg.norm(y)
    roll = rng.uniform(-np.pi, np.pi)
    y = np.cos(roll) * y + np.sin(roll) * np.cross(d, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = d, y, np.cross(d, y)
    T[:3, 3] = pos
    return T


def _mark_target(world, T, ok):
    import meshcat.geometry as g
    node = world._viz.viewer["target"]
    node.set_object(g.Sphere(0.02), g.MeshLambertMaterial(
        color=0x22CC44 if ok else 0xDD2222))
    M = np.eye(4); M[:3, 3] = T[:3, 3]
    node.set_transform(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="meshcat: sample viewpoints, execute every attempt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    world = RobotCell()
    ik = UR5eIK()

    if not args.demo:
        # headless check: one fixed target above the table center
        target = look_down_pose(0.09, -0.28, 0.30)
        t0 = time.perf_counter()
        path, info = plan_to_pose(world, ik, Q_PARK, target)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"branches: {info['enumerated']} enumerated, "
              f"{info['endpoint_free']} endpoint-free, {info['tried']} tried "
              f"({dt:.1f} ms total)")
        if path is None:
            print("tier 1: NO valid direct move — this target needs tier 2/3")
            return 1
        print(f"tier 1: direct move VALID, goal branch {np.round(path[1], 3).tolist()}")
        return 0

    # --demo: sample -> plan -> EXECUTE every attempt. Valid moves run green;
    # invalid ones freeze red at the impact config, then the ghost teleports
    # to the goal branch and the next viewpoint is sampled.
    world.init_viewer()
    rng = np.random.default_rng(args.seed)
    q_now = Q_PARK.copy()
    world.show(q_now, label="park")
    n = valid = 0
    try:
        while True:
            time.sleep(1.5)
            T = sample_viewpoint(rng)
            n += 1
            branches = ik.branches(T)
            free = [q for q in branches if not world.is_colliding(q)]
            if not free:
                print(f"[{n}] unreachable: {len(branches)} branches, "
                      f"0 collision-free endpoints — next")
                _mark_target(world, T, ok=False)
                continue
            free.sort(key=lambda q: np.abs(q - q_now).sum())
            q_goal = free[0]
            ok, _ = world.path_valid([q_now, q_goal])
            _mark_target(world, T, ok)
            print(f"[{n}] {len(branches)} branches, {len(free)} free, "
                  f"direct move {'VALID' if ok else 'INVALID — watch it fail'}")
            world.replay([q_now, q_goal])          # freezes red at impact if bad
            if not ok:
                time.sleep(1.5)
                world.show(q_goal, label="teleport to goal")
            valid += ok
            q_now = q_goal
            print(f"    tally: {valid}/{n} direct moves valid")
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
