"""Path post-processing — planner-agnostic (MAY-184).

This module, not the planner, is what fixes path quality. Measured on 15 hard
cases in the calibrated cell: bolting `partial_shortcut` onto the SAME planner
moved median detour 1.84 -> 1.15 and the worst case 8.97 -> 3.50.

Why the classic shortcut is not enough
--------------------------------------
The standard algorithm (and pyroboplan's `shortcut_path`) samples two points on
the path and tries a straight line between them in ALL joints at once. If one
joint wrapped the long way around, every all-joint chord across that wrap
collides, so every attempt is rejected and the wrap survives -- however many
iterations you run. We measured exactly that pathology: a single joint
travelling 427 deg (worst 667 deg) to move the camera 30 cm.

`partial_shortcut` (Geraerts & Overmars 2007) straightens ONE joint at a time,
which unwinds the wrap incrementally. It is the cure for the failure mode, and
it applies to any sampling planner's output.

Everything here validates against the world's TRUE split margins (env 20 mm,
self 5 mm). Post-processing may never widen the accepted set.
"""

import time

import numpy as np

from inspection.cell.world import DEFAULT_STEP

# Fed to the greedy pruner. Each candidate hop costs a full validated sweep, so
# an unbounded waypoint list makes pruning cost more than planning.
PRUNE_MAX = 12


def path_length(path):
    """Joint-space L1 length -- the cost we select and gate on."""
    return float(np.abs(np.diff(np.asarray(path, dtype=float), axis=0)).sum())


def detour(path, q_from, q_goal):
    """Path length / straight-line lower bound. 1.0 = the ideal direct move.

    This is the gate metric: it is scale-free, so one threshold works for both
    a small wrist reorientation and a big base swing.
    """
    lb = np.abs(np.asarray(q_goal, dtype=float)
                - np.asarray(q_from, dtype=float)).sum()
    return path_length(path) / max(lb, 1e-6)


def partial_shortcut(world, path, budget=0.06, step=DEFAULT_STEP, rng=None):
    """Straighten one joint at a time until the budget runs out.

    Operates on a densely resampled copy (4x the validator step -- fine enough
    to shape the path, coarse enough that each accepted edit is cheap). Every
    candidate edit is accepted only if it is BOTH shorter and valid under the
    true margins, so the result can never be worse or less safe than the input.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    P = [np.asarray(q, dtype=float).copy()
         for q in world.discretize(path, step * 4)]
    n_dof = len(P[0])
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget:
        n = len(P)
        if n < 3:
            break
        i, j = sorted(rng.integers(0, n, 2))
        if j - i < 2:
            continue
        k = int(rng.integers(0, n_dof))          # the ONE joint being fixed
        seg = [P[t].copy() for t in range(i, j + 1)]
        a, b = P[i][k], P[j][k]
        m = len(seg) - 1
        for t in range(m + 1):
            seg[t][k] = a + (b - a) * t / m
        if path_length(seg) >= path_length(P[i:j + 1]) - 1e-9:
            continue                             # no gain, skip the check
        if world.path_valid(seg, step=step)[0]:
            P[i:j + 1] = seg
    return P


def prune(world, path, step=DEFAULT_STEP, max_waypoints=PRUNE_MAX):
    """Collapse a dense path to the fewest waypoints that still validate.

    From each waypoint, binary-search the FARTHEST later waypoint reachable by
    a straight validated move. Binary search assumes reachability is monotone
    along the path, which is not guaranteed -- so this is a heuristic for
    choosing waypoints, never for safety. The caller re-validates the result.

    Returns None if even the original consecutive hop fails the true margins
    (i.e. the planner's path does not survive our gate).
    """
    P = [np.asarray(q, dtype=float) for q in path]
    if len(P) > max_waypoints:
        idx = np.unique(np.linspace(0, len(P) - 1, max_waypoints).astype(int))
        P = [P[i] for i in idx]

    out = [P[0]]
    i = 0
    while i < len(P) - 1:
        lo, hi, best = i + 1, len(P) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if world.path_valid([P[i], P[mid]], step=step)[0]:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best is None:
            if not world.path_valid([P[i], P[i + 1]], step=step)[0]:
                return None
            best = i + 1
        out.append(P[best])
        i = best
    return out


def polish(world, path, budget=0.06, step=DEFAULT_STEP, rng=None):
    """partial_shortcut -> prune -> re-validate. The whole post-processing.

    Returns a validator-approved waypoint path, or None if the input cannot be
    made to survive the true margins.
    """
    if path is None or len(path) < 2:
        return None
    p = partial_shortcut(world, path, budget=budget, step=step, rng=rng)
    p = prune(world, p, step=step)
    if p is None or not world.path_valid(p, step=step)[0]:
        return None
    return p
