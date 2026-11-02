"""Planner benchmark — reproduce the tier-2 selection (MAY-184).

Kept in the repo because the answer is CELL-DEPENDENT: it was measured in the
calibrated workcell with a hard cup, and it will need re-running when the cell
geometry changes, when the tool changes, or to compare against the real arm.
Re-deriving it from scratch cost most of a day.

Metrics are real-robot metrics, not planner-paper metrics:
    detour  joint-L1 path length / straight-line lower bound (1.0 = ideal)
    cart    flange travel [m]
    maxj    largest single-joint total travel [rad]
    revers  summed per-joint direction reversals (hunting / jerk)
    segs    waypoints-1 (blends off, so each is a full stop)

Fair-fight rules: identical hard-case set, identical time budget, identical
post-processing, identical acceptance gate (world.path_valid, true margins).

Hard-kill guard: every planner call runs in a forked child killed at its
deadline. OMPL's timed termination condition only works if the planner POLLS
it; AORRTC was observed running 9.5 minutes on a 1 s budget, and a C++
solve() holding the GIL cannot be interrupted from Python. A killed round is
reported as "killed", never silently retried.

Usage (from repo root, robo env active):
    p inspection/motion/bench.py                # current tier 2, 15 cases
    p inspection/motion/bench.py --compare      # vs AORRTC / BIT* / SORRT*
    p inspection/motion/bench.py --cases 25
"""

import argparse
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from ompl import base as ob
from ompl import geometric as og
import ompl.util as ou

from inspection.cell.world import RobotCell, DEFAULT_STEP
from inspection.motion.ik import UR5eIK
from inspection.motion.direct import direct_move
from inspection.motion.rrt import rrt_move, _ompl_once, CHECK_RESOLUTION
from inspection.motion.smooth import path_length, detour, polish
from inspection.motion.plan import (CUP_POS, CUP_DIMS, DEMO_PARK,
                                    sample_orbit_viewpoint)

ou.setLogLevel(ou.LOG_ERROR)

CTX = mp.get_context("fork")
KILL_SLACK = 2.0


# ------------------------------------------------------------------ metrics
def metrics(ik, path, q_from, q_goal):
    P = np.asarray(path, dtype=float)
    d = np.diff(P, axis=0)
    cart = sum(np.linalg.norm(ik.fk(b)[:3, 3] - ik.fk(a)[:3, 3])
               for a, b in zip(P, P[1:]))
    rev = 0
    for j in range(P.shape[1]):
        s = np.sign(d[:, j])
        s = s[s != 0]
        rev += int((np.diff(s) != 0).sum())
    return dict(detour=detour(path, q_from, q_goal), cart=cart, revers=rev,
                maxj=float(np.abs(d).sum(axis=0).max()), segs=len(P) - 1)


def summarize(tag, rows, n, killed=0):
    if not rows:
        print(f"{tag:22s} 0/{n} solved (killed {killed})", flush=True)
        return
    g = lambda k: np.array([r[k] for r in rows])
    print(f"{tag:22s} {len(rows):2d}/{n} killed {killed} | "
          f"detour med {np.median(g('detour')):5.2f} "
          f"p90 {np.percentile(g('detour'), 90):5.2f} "
          f"max {g('detour').max():5.2f} | "
          f"cart med {np.median(g('cart')):4.2f}m | "
          f"maxj med {np.median(g('maxj')):5.2f} | "
          f"rev med {np.median(g('revers')):3.0f} | "
          f"ms med {np.median(g('ms')):5.0f}", flush=True)


# ------------------------------------------------------------------- cases
def build_cases(n_target, seed=1):
    """Hard cases: tier 1 fails, but the viewpoint HAS free IK branches.

    Deliberately excludes viewpoints with no free branch — those are geometry
    refusals, not planning problems, and including them would flatter whichever
    planner is being tested.
    """
    world, ik = RobotCell(), UR5eIK()
    world.set_object("cup", CUP_DIMS, [*CUP_POS, 0, 0, 0], parent="base")
    center = CUP_POS + np.array([0.0, 0.0, CUP_DIMS[2] / 2])
    cases, rng, q_now, tries = [], np.random.default_rng(seed), DEMO_PARK.copy(), 0
    while len(cases) < n_target and tries < 400:
        tries += 1
        T = sample_orbit_viewpoint(rng, center)
        free = [q for q in ik.branches(T) if not world.is_colliding(q)]
        if not free:
            continue
        free.sort(key=lambda q: np.abs(q - q_now).sum())
        p1 = direct_move(world, q_now, free[0])
        if p1 is not None:
            q_now = p1[-1]
            continue
        cases.append(dict(q_from=q_now.copy(), goals=free[:4], T=T))
        q_now = free[0]
    return world, ik, center, cases


# ------------------------------------------------------------- fork guard
def _child(fn, q):
    try:
        q.put(fn())
    except Exception:
        q.put(None)


def guarded(fn, deadline):
    q = CTX.Queue()
    p = CTX.Process(target=_child, args=(fn, q))
    p.start()
    p.join(deadline)
    if p.is_alive():
        p.kill()
        p.join()
        return "KILLED"
    try:
        return q.get_nowait()
    except Exception:
        return None


# ------------------------------------------------------------ alternatives
def ompl_other(world, q_from, goals, planner_name, budget):
    """Any OMPL planner against the true world — for --compare only."""
    nq = len(q_from)
    space = ob.RealVectorStateSpace(nq)
    b = ob.RealVectorBounds(nq)
    b.setLow(-np.pi)
    b.setHigh(np.pi)
    space.setBounds(b)
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(
        lambda s: not world.is_colliding(np.array([s[i] for i in range(nq)])))
    si.setStateValidityCheckingResolution(CHECK_RESOLUTION)
    si.setup()
    pdef = ob.ProblemDefinition(si)
    st = si.allocState()
    for i in range(nq):
        st[i] = float(q_from[i])
    pdef.addStartState(st)
    gs = ob.GoalStates(si)
    for g in goals:
        s = si.allocState()
        for i in range(nq):
            s[i] = float(g[i])
        gs.addState(s)
    pdef.setGoal(gs)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    planner = getattr(og, planner_name)(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    planner.solve(ob.timedPlannerTerminationCondition(budget))
    if not pdef.hasExactSolution():
        return None
    return [np.array([s[i] for i in range(nq)])
            for s in pdef.getSolutionPath().getStates()]


# --------------------------------------------------------------- runners
def bench_production(world, ik, cases):
    """The shipping tier 2, exactly as plan.py calls it."""
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        path, info = rrt_move(world, c["q_from"], c["goals"])
        ms = (time.perf_counter() - t0) * 1e3
        if path is None:
            continue
        assert world.path_valid(path)[0], "gate violation"
        m = metrics(ik, path, c["q_from"], path[-1])
        m["ms"] = ms
        rows.append(m)
    summarize("tier 2 (production)", rows, len(cases))
    return rows


def bench_alternative(world, ik, cases, name, budget):
    rows, killed = [], 0
    for c in cases:
        t0 = time.perf_counter()
        raw = guarded(
            lambda: ompl_other(world, c["q_from"], c["goals"], name, budget),
            budget + KILL_SLACK)
        if raw == "KILLED":
            killed += 1
            continue
        if raw is None:
            continue
        path = polish(world, raw, budget=0.3 * budget)
        ms = (time.perf_counter() - t0) * 1e3
        if path is None:
            continue
        m = metrics(ik, path, c["q_from"], path[-1])
        m["ms"] = ms
        rows.append(m)
    summarize(f"{name} {budget:.2f}s", rows, len(cases), killed)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=15)
    ap.add_argument("--compare", action="store_true",
                    help="also run the asymptotically-optimal planners")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="budget for the --compare planners")
    args = ap.parse_args()

    world, ik, center, cases = build_cases(args.cases)
    print(f"hard cases: {len(cases)} (tier 1 failed, >=1 free branch)\n",
          flush=True)

    bench_production(world, ik, cases)
    if args.compare:
        for name in ("AORRTC", "BITstar", "SORRTstar"):
            bench_alternative(world, ik, cases, name, args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
