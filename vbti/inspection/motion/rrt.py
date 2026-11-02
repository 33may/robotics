"""Tier 2 — sampling search with a quality gate (MAY-184).

The net under the ladder: tier 1 handles ~75% of moves with a validated
straight joint move, and only the residual reaches here.

Design, and why
---------------
1. OMPL, not pyroboplan. pyroboplan takes ONE global collision padding and
   hardcodes it at every call site. Our validator needs TWO margins (env 20 mm,
   self 5 mm -- the UR5e's own design clearances at park are 17-19 mm, so a
   20 mm blanket margin reports the parked robot in self-collision). Expressing
   two margins through one number forced a second, inflated "planning world",
   and the two worlds disagreed exactly at the boundary -- which is where our
   viewpoints live, since a config looking closely at a cup on a table hugs the
   table. That seam is what the old `_escape_hug` / bridge ladder existed to
   paper over. OMPL takes a validity CALLBACK, so the planner explores exactly
   the set we accept. One world; the seam and its workarounds are gone.

2. Best-of-N, not an asymptotically optimal planner. Benchmarked on 15 hard
   cases in the calibrated cell (see bench.py):
       RRTC best-of-N 0.2s  15/15 solved, 0 hangs, detour med 1.13
       AORRTC         0.2s   7/15 solved, 2 hangs, detour med 1.03
   AORRTC makes tighter paths but solves 40-60% of queries and HANGS on up to
   27% of them -- and its hang rate GROWS with budget. It also ignores its own
   termination condition (a 1 s budget observed running 9.5 min): OMPL's timed
   condition only works if the planner polls it, and a C++ solve() holding the
   GIL cannot be preempted from Python. Reliability beat the last few percent
   of path quality.

3. The goal is a SET. A viewpoint has several free IK branches; handing OMPL
   all of them (ob.GoalStates) lets the search pick the cheapest to reach
   instead of committing to the nearest-by-L1 one up front.

4. Escalate on QUALITY, then refuse. The gate is what turns an unbounded risk
   ("the planner might emit something wild") into a bounded cost ("spend a bit
   more, then look from somewhere else"). A gate with no escalation behind it
   is just a refusal machine; escalation with no gate never knows when to stop.

Reproducibility note: OMPL's global RNG cannot be re-seeded once sampling has
started, so per-call seeding is impossible and runs are not bit-reproducible.
Best-of-N relies on exactly that variation.
"""

import time

import numpy as np

from ompl import base as ob
from ompl import geometric as og
import ompl.util as ou

from inspection.cell.world import DEFAULT_STEP
from inspection.motion.smooth import detour, path_length, polish

ou.setLogLevel(ou.LOG_ERROR)

# One RRT-Connect run. Measured 30-150 ms on this cell; the cap only bounds
# pathological queries -- it is not the quality knob, best-of-N is.
SINGLE_RUN = 0.05

BUDGET = 0.20            # stage A total: ~70% search, ~30% smoothing
ESCALATED = 0.60         # stage B, fires only when stage A fails the gate
GATE = 2.5               # refuse paths longer than this x the straight line

# Exploration resolution. Coarser than the validator on purpose: this bounds
# tree growth cost, while the fine-step gate in world.path_valid remains the
# safety authority.
CHECK_RESOLUTION = 0.002


def _ompl_once(world, q_from, goals, budget=SINGLE_RUN):
    """One RRT-Connect run against the TRUE world. Returns waypoints or None."""
    q_from = np.asarray(q_from, dtype=float)
    goals = [np.asarray(g, dtype=float) for g in goals]
    nq = len(q_from)

    space = ob.RealVectorStateSpace(nq)
    bounds = ob.RealVectorBounds(nq)
    # Our IK wraps branches to +-pi, and sampling wider quadruples the space
    # for no reachable gain -- but never exclude an endpoint we were handed.
    lo = min(-np.pi, float(min(q_from.min(), min(g.min() for g in goals))))
    hi = max(np.pi, float(max(q_from.max(), max(g.max() for g in goals))))
    bounds.setLow(lo)
    bounds.setHigh(hi)
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    # THE point of using OMPL: the planner asks our validator, with its real
    # split margins, instead of us approximating it with inflated geometry.
    si.setStateValidityChecker(
        lambda state: not world.is_colliding(
            np.array([state[i] for i in range(nq)])))
    si.setStateValidityCheckingResolution(CHECK_RESOLUTION)
    si.setup()

    pdef = ob.ProblemDefinition(si)
    start = si.allocState()
    for i in range(nq):
        start[i] = float(q_from[i])
    pdef.addStartState(start)

    goal_set = ob.GoalStates(si)
    for g in goals:
        s = si.allocState()
        for i in range(nq):
            s[i] = float(g[i])
        goal_set.addState(s)
    pdef.setGoal(goal_set)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    planner = og.RRTConnect(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    planner.solve(ob.timedPlannerTerminationCondition(budget))
    if not pdef.hasExactSolution():
        return None
    return [np.array([st[i] for i in range(nq)])
            for st in pdef.getSolutionPath().getStates()]


def best_of_n(world, q_from, goals, budget):
    """Replan until the budget expires; keep the shortest raw path.

    Selection is on RAW length because that is available before the expensive
    smoothing pass -- we smooth only the winner.
    Returns (path, n_tries).
    """
    best, tries, t0 = None, 0, time.perf_counter()
    while True:
        raw = _ompl_once(world, q_from, goals)
        tries += 1
        if raw is not None and (best is None
                                or path_length(raw) < path_length(best)):
            best = raw
        if time.perf_counter() - t0 >= budget:
            break
    return best, tries


def rrt_move(world, q_from, goals, budget=BUDGET, escalated=ESCALATED,
             gate=GATE, step=DEFAULT_STEP):
    """Plan q_from -> any of `goals`, with a path-quality gate.

    Stage A: best-of-N within `budget`, smooth, gate.
    Stage B: same with `escalated` budget (fires only if A failed the gate).
    Then refuse.

    Returns (path, info). `path` is None on refusal; `info` carries the
    diagnosis so the exploration loop can tell "cannot get there" from
    "can get there but only badly".
    """
    goals = [np.asarray(g, dtype=float) for g in goals]
    info = {"stage": None, "tries": 0, "detour": None, "ms": 0.0,
            "reason": None}
    t0 = time.perf_counter()

    for stage, b in (("A", budget), ("B", escalated)):
        raw, tries = best_of_n(world, q_from, goals, 0.7 * b)
        info["tries"] += tries
        if raw is None:
            info["reason"] = "no path found"
            continue
        path = polish(world, raw, budget=0.3 * b, step=step)
        if path is None:
            info["reason"] = "path failed the true-margin gate"
            continue
        d = detour(path, q_from, path[-1])
        if d > gate:
            info["reason"] = f"detour {d:.2f}x over gate {gate}x"
            continue
        info.update(stage=stage, detour=d,
                    ms=(time.perf_counter() - t0) * 1e3, reason=None)
        return path, info

    info["ms"] = (time.perf_counter() - t0) * 1e3
    return None, info
