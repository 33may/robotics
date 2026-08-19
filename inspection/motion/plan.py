"""The motion ladder — the ONE planning entry point (MAY-184).

plan_viewpoint() walks two tiers for a target flange pose:
    tier 1  validated direct joint move       (the workhorse, ~75% of moves)
    tier 2  OMPL best-of-N + smoothing + gate (the rest, or an honest refusal)
Every returned path is validator-approved end to end. None = the viewpoint is
refused — either no free IK branch, or no path good enough to execute — and the
exploration loop picks another viewpoint.

A third rung (radial retract via an outer shell) was removed 2026-08-19: it
solved 0/11 hard cases here, because the safe shell around an object sitting on
a table is mostly inside the table. See git history if it is ever needed.

Usage (from repo root, robo env active):
    p inspection/motion/plan.py           # headless: ladder stats over 40 samples
    p inspection/motion/plan.py --demo    # meshcat: orbit a fake cup, execute all
"""

import argparse
import sys
import time

import numpy as np

from inspection.cell.world import RobotCell, Q_PARK, DEFAULT_STEP
from inspection.motion.ik import UR5eIK
from inspection.motion.direct import direct_move
from inspection.motion.rrt import rrt_move


def plan_viewpoint(world, ik, q_from, T_target, seed=0, n_branches=3,
                   step=DEFAULT_STEP, tiers=(1, 2)):
    """Ladder-plan to a flange pose. Returns (path, report).

    path: list of joint waypoints (validated) or None.
    report: dict — branches enumerated/free, tier that won, detour, ms spent,
    and on refusal a `reason` distinguishing "cannot get there" from "can get
    there but only badly".
    tiers: which rungs may run — (1, 2) is the ladder; a single tier isolates
    one rung for evaluation.
    """
    t0 = time.perf_counter()
    rep = {"enumerated": 0, "endpoint_free": 0, "tier": None, "ms": 0.0,
           "detour": None, "tries": 0, "reason": None}
    branches = ik.branches(T_target)
    rep["enumerated"] = len(branches)
    free_all = [q for q in branches if not world.is_colliding(q)]
    rep["endpoint_free"] = len(free_all)
    if not free_all:
        rep["ms"] = (time.perf_counter() - t0) * 1e3
        rep["reason"] = "no collision-free IK branch"
        return None, rep
    free_all.sort(key=lambda q: np.abs(q - q_from).sum())
    # tier 1 commits to one branch at a time, so try only the nearest few;
    # tier 2 takes the WHOLE set as an OMPL goal set and picks for itself.
    free = free_all[:n_branches]

    if 1 in tiers:                                        # tier 1
        for q_goal in free:
            path = direct_move(world, q_from, q_goal, step=step)
            if path is not None:
                rep["tier"], rep["ms"] = 1, (time.perf_counter() - t0) * 1e3
                return path, rep

    if 2 in tiers:                        # tier 2 — whole branch set at once
        path, info = rrt_move(world, q_from, free_all, step=step)
        rep["tries"], rep["reason"] = info["tries"], info["reason"]
        if path is not None:
            rep["tier"], rep["detour"] = 2, info["detour"]
            rep["ms"] = (time.perf_counter() - t0) * 1e3
            return path, rep

    rep["ms"] = (time.perf_counter() - t0) * 1e3
    return None, rep


# ------------------------------------------------------------------ demo
CUP_POS = np.array([0.09, -0.28, 0.10])      # base frame: on the table, 10cm tall-ish
CUP_DIMS = [0.08, 0.08, 0.12]

# park clear of the cup column (the old Q_PARK hovers its tip IN the cup —
# with a hard cup the start config must itself be valid)
DEMO_PARK = np.array([-1.821, -0.992, 0.728, -0.305, -1.265, -0.619])


def sample_orbit_viewpoint(rng, center, r_lo=0.30, r_hi=0.45, max_tilt=65.0):
    """Look-AT pose: position on a random upper-hemisphere shell around the
    center, flange +X aimed at the center, random roll about the view axis."""
    tilt = rng.uniform(0.0, np.deg2rad(max_tilt))        # 0 = straight above
    azim = rng.uniform(-np.pi, np.pi)
    out = np.array([np.sin(tilt) * np.cos(azim),
                    np.sin(tilt) * np.sin(azim),
                    np.cos(tilt)])                        # center -> camera
    pos = center + out * rng.uniform(r_lo, r_hi)
    d = -out                                              # view direction
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    y = np.cross(d, ref); y /= np.linalg.norm(y)
    roll = rng.uniform(-np.pi, np.pi)
    y = np.cos(roll) * y + np.sin(roll) * np.cross(d, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = d, y, np.cross(d, y)
    T[:3, 3] = pos
    return T


def _mark_target(world, T, color):
    import meshcat.geometry as g
    node = world._viz.viewer["target"]
    node.set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=color))
    M = np.eye(4); M[:3, 3] = T[:3, 3]
    node.set_transform(M)


def run(demo, seed, tiers=(1, 2)):
    world = RobotCell()
    ik = UR5eIK()
    rng = np.random.default_rng(seed)
    center = CUP_POS + np.array([0.0, 0.0, CUP_DIMS[2] / 2])
    q_now = DEMO_PARK.copy()
    tally = {1: 0, 2: 0, "unreachable": 0, "refused": 0}
    TIER_COLOR = {1: 0x22CC44, 2: 0x2288DD}

    # the cup is a HARD obstacle now — it belongs in the world in every mode
    world.set_object("cup", CUP_DIMS, [*CUP_POS, 0, 0, 0], parent="base")

    if demo:
        world.init_viewer()
        world.show(q_now, label="park")

    n_samples = 0
    try:
        while True:
            if not demo and n_samples >= 40:
                break
            n_samples += 1
            T = sample_orbit_viewpoint(rng, center)
            path, rep = plan_viewpoint(world, ik, q_now, T,
                                       seed=seed or 0, tiers=tiers)
            if path is None:
                key = "unreachable" if rep["endpoint_free"] == 0 else "refused"
                tally[key] += 1
                print(f"[{n_samples}] {key}: {rep['enumerated']} branches, "
                      f"{rep['endpoint_free']} free — {rep['reason']} "
                      f"({rep['ms']:.0f} ms)")
                if demo:
                    _mark_target(world, T, 0xDD2222)
                    time.sleep(1.0)
                continue
            tally[rep["tier"]] += 1
            det = f" detour {rep['detour']:.2f}x" if rep["detour"] else ""
            print(f"[{n_samples}] tier {rep['tier']} "
                  f"({len(path)} waypoints,{det} {rep['ms']:.0f} ms) — "
                  f"t1/t2 {tally[1]}/{tally[2]}, "
                  f"unreachable {tally['unreachable']}, "
                  f"refused {tally['refused']}")
            if demo:
                _mark_target(world, T, TIER_COLOR[rep["tier"]])
                world.replay(path)
                time.sleep(1.0)
            q_now = path[-1]
    except KeyboardInterrupt:
        pass
    if not demo:
        print(f"\n40 orbit viewpoints: tier1 {tally[1]}, tier2 {tally[2]}, "
              f"unreachable {tally['unreachable']}, refused {tally['refused']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="meshcat: orbit the fake cup, execute every plan")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tier", type=int, choices=(1, 2), default=None,
                    help="isolate ONE rung: only this tier may plan")
    args = ap.parse_args()
    tiers = (args.tier,) if args.tier else (1, 2)
    return run(args.demo, args.seed, tiers)


if __name__ == "__main__":
    sys.exit(main())
