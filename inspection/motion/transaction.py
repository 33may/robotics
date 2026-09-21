"""The three-leg move transaction (design: 2026-09-17-ray-views-design.md,
§Move transaction; Stage B task 2).

A move between two ray addresses is never one plan. It is three legs:

    1. slide-out  — radially outward on the CURRENT ray, joint-space straight
                    lines between consecutive samples, roll-locked
    2. hop        — the two-tier ladder (`plan_viewpoint`) through open space;
                    THE ONLY LEG WHERE THE WRIST MAY FLIP ROLL
    3. slide-in   — radially inward on the TARGET ray to its closest safe
                    sample (the view pose), roll-locked, one IK branch

Why hop-then-slide: the long move stays in open space where planning is easy,
and the delicate approach is short, straight and radial — it cannot laterally
sweep the object.

The search is PINNED and LINEAR (design): the entry on the target ray is
pinned to the farthest sample its slide corridor reaches, and the current
ray's out-depths are walked current-depth-first, one plan each — ~18 plans,
not the ~18x18 (~10 s) an out x entry product would cost. Only when that whole
walk refuses does the entry widen inward, and only `MAX_ENTRY_TRIES` times.

It is also TWO-PASS (controller ruling 2026-09-17). Linear was not cheap
enough on the refusing side: when an entry pose has free IK branches but no
path, every attempt escalates to tier 2 and pays ~620 ms before refusing —
a measured 20.5 s refusal, against the design's ~2 s worst case. So the same
candidate order is walked twice: pass 1 probes each candidate with TIER 1
ONLY (a validated direct move, milliseconds), and only if that whole sweep
refuses does pass 2 spend TIER 2 ONLY (OMPL) on the first `MAX_LADDER_TRIES`
candidates. A refusal now costs ~18 probes + 3 searches, and the common case
costs ONE direct move.

`None` is the honest answer: the target address is transition-inaccessible and
the caller marks it blocked. No fallback substitution.

Every leg is validated against the world passed in, and that world travels
with the legs (`Legs.world`) so execution can re-validate against the world
the legs were planned under — the retreat precedent (`rig.move(path,
world=...)`).
"""
from collections import namedtuple

import numpy as np

from inspection.cell.world import DEFAULT_STEP
from inspection.motion.plan import plan_viewpoint

#: Where the arm is now: the ray it sits on, the depth/roll of that sample,
#: and the ray's full valid sample list (the slide-out corridor). `None`
#: instead of a RayState = the first move after a survey or a retreat.
#:
#: `ray` is a READER-FACING SEAM, not a planner input: `plan_transaction`
#: reads only `depth`/`roll`/`samples` (the corridor and the lock), and the
#: samples already carry their own poses. It is carried so a caller, a test
#: or a record can say WHICH ray a state describes without reconstructing it
#: from a camera pose — keep it filled, but do not assume the search
#: consults it.
RayState = namedtuple("RayState", ["ray", "depth", "roll", "samples"])

#: The planned transaction. `out` may be empty (no slide-out needed / no
#: current ray); `slide_in` always ends on the target view pose. `world` is
#: the world the three legs were validated against.
Legs = namedtuple("Legs", ["out", "hop", "slide_in", "world"])

#: How many entries may be tried in total (the pinned one included) before the
#: address is called transition-inaccessible. Bounds the widening fallback at
#: MAX_ENTRY_TRIES x n_out_depths plans (~54 x 30 ms worst case) instead of
#: the quadratic product the pinning exists to avoid.
MAX_ENTRY_TRIES = 3

#: How many candidates may spend tier 2 (OMPL) once the tier-1 sweep has
#: refused every one of them. Tier 2 costs ~600 ms per refusal here, so this
#: number is the refusal's price tag: 3 searches on top of the sweep, ~2 s
#: worst case (controller ruling 2026-09-17).
MAX_LADDER_TRIES = 3

#: Largest per-joint step a slide may take between two consecutive ray samples
#: [rad]. A slide is one IK branch by contract — no +-pi re-expression, no
#: branch change mid-approach. Measured on the Stage A box world: a genuine
#: RAY_STEP_M (2 cm) slide step moves every joint by <= 0.19 rad, while the
#: nearest free branch at a corridor break jumped 3.09 rad (a wrist flip that
#: was collision-free and would still have executed as an ugly mid-slide
#: spin). pi/4 sits in the empty gap between the two populations.
SLIDE_BRANCH_MAX = np.pi / 4


def _slide_step(world, ik, T_flange, q_prev, step):
    """One validated joint segment from `q_prev` to the pose `T_flange`.

    Branch identity along a ray is not labelled anywhere, so it is derived:
    seed the IK with the previous waypoint (`q_near` re-expresses every
    solution in the revolution nearest it), take the nearest free branch that
    stays inside SLIDE_BRANCH_MAX, and accept it only if the straight joint
    interpolation is clear at the world's own collision resolution. None = the
    corridor ends here.
    """
    q_prev = np.asarray(q_prev, dtype=float)
    free = [q for q in ik.branches(T_flange, q_near=q_prev)
            if not world.is_colliding(q)]
    free.sort(key=lambda q: np.abs(q - q_prev).sum())
    for q in free:
        if np.abs(q - q_prev).max() > SLIDE_BRANCH_MAX:
            continue                      # another branch, not this slide
        if world.path_valid([q_prev, q], step=step)[0]:
            return np.asarray(q, dtype=float)
    return None


def _slide(world, ik, samples, q_start, step):
    """Waypoints for a slide through `samples` in order, starting from
    `q_start` (which is NOT part of the result). Stops at the first sample the
    corridor cannot reach — a slide is contiguous or it is not a slide."""
    out, q = [], np.asarray(q_start, dtype=float)
    for s in samples:
        q_next = _slide_step(world, ik, s.T_flange, q, step)
        if q_next is None:
            break
        out.append(q_next)
        q = q_next
    return out


def _roll_locked(samples, roll):
    """The leading run of `samples` that keeps `roll` — the safety filter on
    the slide's roll lock.

    `sample_ray` emits uniform-roll lists (task 2b), so today this passes the
    whole list through. It stays because the invariant belongs to the SLIDE,
    not to the sampler: a list that changes roll partway out would otherwise
    put a wrist flip inside a radial approach, and that must not become
    possible again through a sampler change nobody re-read this file for."""
    run = []
    for s in samples:
        if abs(s.roll - roll) > 1e-9:
            break
        run.append(s)
    return run


def _out_samples(current):
    """The current ray's out-depths, current-depth-first: samples at or beyond
    the depth the camera sits at, roll-locked to the pose it is in."""
    beyond = [s for s in current.samples if s.depth >= current.depth - 1e-9]
    return _roll_locked(beyond, current.roll)


def plan_transaction(world, ik, t_flange_cam, current, target, q_now, seed=0,
                     planner=plan_viewpoint, max_entries=MAX_ENTRY_TRIES,
                     max_ladder=MAX_LADDER_TRIES, step=DEFAULT_STEP):
    """Plan the three legs from `q_now` to `target`'s view pose, or None.

    world:   the LIVE planner world at plan time (ruling) — every leg is
             validated against it and it rides home in `Legs.world`.
    current: `RayState` the arm sits on, or None (first move after a survey or
             a retreat — then there is no slide-out and the hop starts at
             `q_now`).
    target:  a `RayResolution` from `resolve_cell` / `resolve_face`; its
             `samples` list (element 0 = closest = the view pose) is the
             slide-in corridor.
    planner: the hop planner, `plan_viewpoint(world, ik, q_from, T, seed=,
             tiers=)` by default — injectable so the search can be exercised
             without paying for real planning. It MUST honour `tiers` and
             `step`: the search asks for `(1,)` while it sweeps and `(2,)`
             only for the few candidates it is willing to spend OMPL on.

    `t_flange_cam` is accepted for symmetry with the resolvers' call sites;
    the samples already carry their flange poses, so it is not needed here.
    """
    if max_entries < 1:
        # Argument validation, before any work: a caller who asked for zero
        # entries gets told so, not a corridor probe's worth of IK first and
        # then the same error (a stub planner's call count would have
        # recorded a search that was never allowed to happen).
        raise ValueError("max_entries must be >= 1")
    if not target.samples:
        return None
    q_now = np.asarray(q_now, dtype=float)

    # --- the slide-in corridor, and with it the entry candidates ----------
    # Probe how far outward a valid roll-locked slide reaches on the target
    # ray: that farthest sample IS the pinned entry.
    corridor = _roll_locked(target.samples, target.roll)
    reach = 1 + len(_slide(world, ik, corridor[1:],
                           np.asarray(target.q, dtype=float), step))
    # Widening stops one sample short of the view pose: `e == 0` means the
    # hop lands ON the address, which is the lateral arrival hop-then-slide
    # exists to prevent (ruled, task 2b). A corridor that is genuinely ONE
    # sample long keeps it as the honest last resort — refusing there would
    # mark cells transition-inaccessible that Stage A proved reachable.
    wide = list(range(reach - 1, 0, -1)) or [0]      # reach 1: the last resort
    entries = wide[:max_entries]

    # --- the slide-out corridor, walked once ------------------------------
    out_chain = []
    if current is not None:
        out = _out_samples(current)
        if out and abs(out[0].depth - current.depth) <= 1e-9:
            out = out[1:]                 # the arm is already at that sample
        out_chain = _slide(world, ik, out, q_now, step)

    # --- the candidate order: every out-depth against one entry, then widen
    candidates = [(e, j) for e in entries for j in range(len(out_chain) + 1)]

    def attempt(e, j, tiers):
        """One hop-and-slide-in try, or None."""
        q_out = out_chain[j - 1] if j else q_now
        path, _rep = planner(world, ik, q_out, corridor[e].T_flange,
                             seed=seed, step=step, tiers=tiers)
        if path is None:
            return None
        # The hop picks its own branch at the entry pose, so the slide-in is
        # derived from where the hop actually LANDS, not from the probe
        # above — that is what keeps it a single-branch approach.
        slide = [np.asarray(path[-1], dtype=float)]
        inward = _slide(world, ik, list(reversed(corridor[:e])), slide[0],
                        step)
        if len(inward) != e:
            return None                   # arrival branch cannot slide in
        slide.extend(inward)
        return Legs(list(out_chain[:j]),
                    [np.asarray(q, dtype=float) for q in path], slide, world)

    # --- pass 1: sweep every candidate on tier 1; pass 2: the capped ladder
    # Pass 2 asks for tier 2 ALONE. Tier 1 is deterministic and pass 1 already
    # ran it on every candidate, so offering it again would replay an
    # identical refusal — and for a candidate whose pass-1 plan was thrown out
    # at the landing-branch check below, an identical SUCCESS that is thrown
    # out again, burning a ladder slot without ever reaching OMPL. The rungs
    # are independent: tier 2 is not seeded by tier 1 (plan.py).
    for tiers, pool in (((1,), candidates),
                        ((2,), candidates[:max_ladder])):
        for e, j in pool:
            legs = attempt(e, j, tiers)
            if legs is not None:
                return legs
    return None
