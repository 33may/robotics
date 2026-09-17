"""The three-leg move transaction (design: §Move transaction, Stage B task 2).

Fixtures are the Stage A box world (copied from test_rays.py) — real IK, real
collision world, real ray samples. The HOP is the only leg that needs a
planner, so it is stubbed: a counting stand-in proves the search is linear and
entry-pinned without paying OMPL, and one smoke test runs the real ladder.
"""
from collections import namedtuple

import numpy as np

from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.view.faceview import face_cam_pose
from inspection.view.rays import cell_ray, resolve_cell
from inspection.view.viewsphere import ROLLS, T_FLANGE_CAM

CENTER = (0.05, -0.38, 0.0375)
AABB = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))


def _box_world():
    w = RobotCell()
    w.set_object("object", [0.15, 0.15, 0.075],
                 [0.05, -0.38, 0.0375, 0, 0, 0.17])
    return w, UR5eIK()


def _cell(w, ik, h, v):
    res = resolve_cell(w, ik, T_FLANGE_CAM, CENTER, AABB, h, v,
                       fy=647.0, height_px=720)
    assert res is not None, f"fixture cell {(h, v)} must be accessible"
    return res


def _current(w, ik, h=3, v=2):
    """The arm parked on cell {3,2}'s ray at its closest sample: 11 valid
    out-depths, all roll 0 — the slide-out corridor the search walks."""
    from inspection.motion.transaction import RayState
    res = _cell(w, ik, h, v)
    return (RayState(cell_ray(CENTER, h, v), res.depth, res.roll, res.samples),
            np.asarray(res.q, dtype=float))


def _ray_pose(ik, q, ray):
    """(depth, roll) of config `q` read back off `ray`, or AssertionError if
    the config is not a roll-locked pose ON the ray — this is what makes a
    slide leg radial rather than merely collision-free."""
    T_cam = ik.fk(q) @ T_FLANGE_CAM
    d = float(np.linalg.norm(T_cam[:3, 3] - np.asarray(ray.anchor)))
    for roll in ROLLS:
        T = face_cam_pose(np.asarray(ray.anchor),
                          np.asarray(ray.direction), d, float(roll))
        if np.allclose(T, T_cam, atol=1e-6):
            return d, float(roll)
    raise AssertionError("waypoint is not a roll-locked pose on the ray")


class _Keepout:
    """The real world with a wall across ONE ray: a straight segment that
    reaches beyond `d_max` on `ray` is refused exactly as a keep-out at that
    depth would refuse it. Endpoint collision checks are untouched, so only
    the segment-wise validation can catch it."""

    def __init__(self, world, ik, ray, d_max):
        self._w, self._ik, self._ray, self._d = world, ik, ray, d_max

    def __getattr__(self, name):
        return getattr(self._w, name)

    def path_valid(self, path, **kw):
        for q in path:
            T = self._ik.fk(q) @ T_FLANGE_CAM
            off = T[:3, 3] - np.asarray(self._ray.anchor)
            d = float(np.linalg.norm(off))
            if d > self._d + 1e-9 and np.allclose(off / d, self._ray.direction,
                                                  atol=1e-6):
                return False, ("keepout", d)
        return self._w.path_valid(path, **kw)


_Call = namedtuple("_Call", ["q_from", "T", "tiers", "step"])


class _StubPlanner:
    """Counting stand-in for plan_viewpoint. Records every call — start
    config, entry pose and the RUNGS it was allowed to use — and answers with
    a two-waypoint move to the nearest collision-free branch;
    `fail(stub, q_from, T)` refuses instead."""

    def __init__(self, fail=None, worst=False):
        self.calls = []
        self._fail = fail or (lambda stub, q_from, T: False)
        self._worst = worst          # land on the UGLIEST branch on purpose

    def __call__(self, world, ik, q_from, T_target, seed=0, **kw):
        q_from = np.asarray(q_from, dtype=float)
        T = np.asarray(T_target, dtype=float)
        self.calls.append(_Call(q_from, T,
                                tuple(kw.get("tiers", (1, 2))),
                                kw.get("step")))
        if self._fail(self, q_from, T):
            return None, {"reason": "stub refusal"}
        free = [q for q in ik.branches(T, q_near=q_from)
                if not world.is_colliding(q)]
        if not free:
            return None, {"reason": "no collision-free IK branch"}
        free.sort(key=lambda q: np.abs(q - q_from).sum(), reverse=self._worst)
        return [q_from, np.asarray(free[0], dtype=float)], {"tier": 1}

    @property
    def entries(self):
        """Distinct entry poses asked for, first-use order."""
        out = []
        for c in self.calls:
            if not any(np.allclose(c.T, s) for s in out):
                out.append(c.T)
        return out

    @property
    def probes(self):
        """Pass-1 calls: the cheap tier-1-only hop probes."""
        return [c for c in self.calls if c.tiers == (1,)]

    @property
    def ladders(self):
        """Pass-2 calls: any call offered tier 2 at all. Deliberately looser
        than what pass 2 is allowed to ask for, so a regression to
        `tiers=(1, 2)` still lands here and trips the equality assertions."""
        return [c for c in self.calls if 2 in c.tiers]


def test_pinned_entry_search_is_linear_and_current_depth_first():
    """Design §transaction: entry pinned to the target ray's farthest valid
    sample, out-depths walked current-depth-first, ONE plan per out-depth —
    never out x entry quadratic."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: len(s.calls) <= 3)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None
    assert len(stub.calls) == 4, "one plan per out-depth, no re-tries"
    assert len(stub.calls) <= len(current.samples)     # <= n_out_depths
    assert len(stub.entries) == 1, "entry stays pinned while out-depths remain"
    assert len(legs.out) == 3, "three failures = three steps outward"
    # each attempt hops from exactly the pose the slide-out walked to
    assert np.allclose(stub.calls[0].q_from, q_now)
    for k in range(1, 4):
        assert np.allclose(stub.calls[k].q_from, legs.out[k - 1])
    # the pinned entry is the FARTHEST sample of the target's slide corridor
    d_entry, _ = _ray_pose(ik, legs.slide_in[0], cell_ray(CENTER, 6, 2))
    assert d_entry > target.depth + 1e-9
    assert np.allclose(stub.calls[0].T, ik.fk(legs.slide_in[0]))


def test_no_out_depth_plans_is_none():
    """No out-depth from which the transition plans -> honestly inaccessible.
    The refusal is the expensive path, so it is the one that must stay cheap:
    the whole search sweeps on tier-1 probes and only MAX_LADDER_TRIES
    attempts are ever allowed to spend tier 2 (controller ruling 2026-09-17 —
    a refusal measured 20.5 s when every attempt escalated)."""
    from inspection.motion.transaction import (MAX_ENTRY_TRIES,
                                               MAX_LADDER_TRIES,
                                               plan_transaction)
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: True)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is None
    assert 1 < len(stub.entries) <= MAX_ENTRY_TRIES, "widen, but bounded"
    assert len(stub.probes) + len(stub.ladders) == len(stub.calls)
    assert len(stub.probes) <= MAX_ENTRY_TRIES * len(current.samples)
    assert len(stub.ladders) == MAX_LADDER_TRIES, "pass 2 is capped"
    assert all(c.tiers == (2,) for c in stub.ladders), \
        "pass 2 asks for tier 2 ALONE — tier 1 is deterministic and already " \
        "ran on every candidate, so replaying it burns a ladder slot"
    assert all(c.tiers == (1,) for c in stub.calls[:len(stub.probes)]), \
        "pass 2 never starts before the cheap sweep is exhausted"


def test_only_tier_two_hop_resolves_in_pass_two():
    """A hop no direct move can make is still a transaction: the cheap sweep
    exhausts, then the full ladder plans it."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: s.calls[-1].tiers == (1,))

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None
    assert len(stub.probes) > 1, "the cheap sweep ran first, and whole"
    assert all(c.tiers == (1,) for c in stub.calls[:-1])
    assert len(stub.ladders) == 1, "the first full-ladder attempt wins"
    assert stub.ladders[0].tiers == (2,), "pass 2 spends tier 2 alone"


def test_step_reaches_the_hop_as_well_as_the_slides():
    """One collision resolution for the whole transaction: a caller tightening
    `step` must not get slides validated at one resolution and a hop planned
    at another (world.path_valid asserts step x lever < padding)."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: len(s.calls) <= 1)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub, step=0.008)

    assert legs is not None and len(stub.calls) > 1
    assert all(c.step == 0.008 for c in stub.calls)


def test_pass_one_success_never_spends_tier_two():
    """The common case pays for ONE direct-move probe — tier 2 is never even
    offered the query."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner()

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None
    assert stub.ladders == [], "no tier-2 query on a pass-1 success"
    assert len(stub.calls) == len(stub.probes) == 1


def test_entry_widens_only_after_the_out_depths_exhaust():
    """Widening is the fallback, not the first move: every out-depth is tried
    against the pinned entry before any closer entry is considered."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    # refuse the pinned entry forever; the next entry inward plans fine
    stub = _StubPlanner(
        fail=lambda s, q, T: np.allclose(T, s.calls[0].T))

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None
    pinned = stub.entries[0]
    n_pinned = sum(1 for c in stub.calls if np.allclose(c.T, pinned))
    assert n_pinned >= 2, "the whole out-depth walk runs on the pinned entry"
    first_widened = next(i for i, c in enumerate(stub.calls)
                         if not np.allclose(c.T, pinned))
    assert first_widened == n_pinned, "no interleaving: pinned, then wider"
    ray = cell_ray(CENTER, 6, 2)
    d_pinned, _ = _ray_pose(ik, target.samples[-1].q, ray)
    d_entry, _ = _ray_pose(ik, legs.slide_in[0], ray)
    assert d_entry < d_pinned, "the winning entry is closer than pinned"


def test_roll_flip_lives_in_the_hop_only():
    """Rolls are locked inside a slide; the wrist may only flip in open space
    between out[-1] and slide_in[0] (design: the hop absorbs roll flips)."""
    from inspection.motion.transaction import _roll_locked, plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 9, 0)                 # roll pi at its closest sample
    assert abs(target.roll - current.roll) > 1e-9, "fixture must flip the roll"
    assert len(_roll_locked(target.samples, target.roll)) \
        == len(target.samples) - 1, "fixture's corridor ends at a roll change"
    stub = _StubPlanner(fail=lambda s, q, T: len(s.calls) <= 2)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None and legs.out and len(legs.slide_in) > 1
    out_ray, in_ray = current.ray, cell_ray(CENTER, 9, 0)
    for q in [q_now] + list(legs.out):
        assert _ray_pose(ik, q, out_ray)[1] == current.roll
    for q in legs.slide_in:
        assert _ray_pose(ik, q, in_ray)[1] == target.roll


def test_slide_never_changes_branch():
    """A slide is ONE IK branch: no +-pi re-expression, no wrist flip mid-
    approach. Cell {0,2}'s corridor is exactly where an unguarded nearest-
    branch walk takes a 3.09 rad jump that is collision-free and still wrong —
    walked outward, as the slide-out walks it."""
    from inspection.motion.transaction import (SLIDE_BRANCH_MAX,
                                               plan_transaction)
    w, ik = _box_world()
    current, q_now = _current(w, ik, h=0, v=2)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: len(s.calls) <= 5)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is not None
    chain = [q_now] + list(legs.out)
    for a, b in zip(chain, chain[1:]):
        assert np.abs(b - a).max() <= SLIDE_BRANCH_MAX, np.abs(b - a).max()
    for a, b in zip(legs.slide_in, legs.slide_in[1:]):
        assert np.abs(b - a).max() <= SLIDE_BRANCH_MAX, np.abs(b - a).max()
    # the corridor ENDS at the branch change: it does not walk through it
    assert len(legs.out) < len(current.samples) - 1


def test_slide_corridor_ends_where_a_segment_is_refused():
    """A slide leg is only as long as the world allows: every interpolated
    segment is validated, so a keep-out at 0.30 m pins the entry to the
    farthest sample still reachable, not to the farthest sample in the list."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    ray = cell_ray(CENTER, 6, 2)
    assert target.samples[-1].depth > 0.30 + 1e-9, "fixture must reach past it"
    blocked = _Keepout(w, ik, ray, d_max=0.30)

    legs = plan_transaction(blocked, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=_StubPlanner())

    assert legs is not None
    depths = [_ray_pose(ik, q, ray)[0] for q in legs.slide_in]
    assert max(depths) <= 0.30 + 1e-9, depths
    assert np.isclose(max(depths), 0.30), "entry pins to farthest reachable"
    assert legs.world is blocked, "the legs carry the world that ruled"


def test_slide_out_stops_at_a_roll_change():
    """Rolls are locked inside a slide, so a sample list that flips roll one
    step out leaves exactly ONE out-depth: the one the camera is already at."""
    from inspection.motion.transaction import _out_samples, plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik, h=3, v=1)     # roll pi, then roll 0 out
    assert abs(current.samples[1].roll - current.roll) > 1e-9
    assert len(_out_samples(current)) == 1
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner(fail=lambda s, q, T: True)

    assert plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub) is None
    assert stub.probes, "the cheap sweep must have run"
    for T in stub.entries:
        n = sum(1 for c in stub.probes if np.allclose(c.T, T))
        assert n == 1, "only the current depth is a roll-locked out-depth"


def test_hop_that_cannot_slide_in_is_refused():
    """The slide-in is derived from the branch the hop LANDS on, and it must
    reach the view pose. A hop that arrives on a branch no slide can follow
    inward is not a transaction — it is an honest refusal, never a truncated
    approach that stops short of the address."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 3, 2)
    stub = _StubPlanner(worst=True)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=stub)

    assert legs is None
    assert len(stub.calls) > 1, "every out-depth and entry was actually tried"


def test_no_current_ray_means_no_slide_out():
    """First move after survey / after retreat: legs 2-3 only."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    _, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)
    stub = _StubPlanner()

    legs = plan_transaction(w, ik, T_FLANGE_CAM, None, target, q_now,
                            planner=stub)

    assert legs is not None
    assert legs.out == []
    assert len(stub.calls) == 1
    assert np.allclose(stub.calls[0].q_from, q_now)
    assert np.allclose(legs.hop[0], q_now)


def test_slide_in_reverses_to_exactly_the_forward_list():
    """The retreat retraces recorded waypoints in reverse, so the reversed
    slide-in must be the ray's own corridor walked outward: waypoint k sits on
    sample k, ascending depth, and the whole reversal is still valid. The
    sample poses are the independent reconstruction — the leg is never
    compared against itself."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=_StubPlanner())

    assert legs is not None and len(legs.slide_in) > 1
    rev = list(reversed(legs.slide_in))
    for q, s in zip(rev, target.samples[:len(rev)]):
        assert np.allclose(ik.fk(q), s.T_flange, atol=1e-6)
    ok, info = w.path_valid(rev)
    assert ok, info


def test_slide_in_is_collision_free_and_radial():
    """Every slide-in configuration is free in the PLAN world, the leg is
    segment-wise valid, and it walks the ray inward to the target sample."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now,
                            planner=_StubPlanner())

    assert legs is not None
    assert legs.world is w, "the legs carry the world they were planned under"
    assert np.array_equal(legs.hop[-1], legs.slide_in[0]), \
        "the legs join: the slide-in starts on the branch the hop landed on"
    for q in legs.slide_in:
        assert not w.is_colliding(q)
    ok, info = w.path_valid(legs.slide_in)
    assert ok, info
    ray = cell_ray(CENTER, 6, 2)
    depths = [_ray_pose(ik, q, ray)[0] for q in legs.slide_in]
    assert depths == sorted(depths, reverse=True), "inward, monotonically"
    assert np.isclose(depths[-1], target.depth)
    assert np.allclose(ik.fk(legs.slide_in[-1]), target.T_flange, atol=1e-6)
    ok, info = w.path_valid([q_now] + list(legs.out))
    assert ok, info


def test_real_ladder_plans_a_transaction():
    """Smoke: the default planner is the two-tier ladder, and the three legs
    concatenate into one validated path."""
    from inspection.motion.transaction import plan_transaction
    w, ik = _box_world()
    current, q_now = _current(w, ik)
    target = _cell(w, ik, 6, 2)

    legs = plan_transaction(w, ik, T_FLANGE_CAM, current, target, q_now)

    assert legs is not None
    path = [q_now] + list(legs.out) + list(legs.hop) + list(legs.slide_in)
    ok, info = w.path_valid(path)
    assert ok, info
    assert np.allclose(ik.fk(path[-1]), target.T_flange, atol=1e-6)


def main():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
    print(f"OK test_transaction ({n} tests)")


if __name__ == "__main__":
    main()
