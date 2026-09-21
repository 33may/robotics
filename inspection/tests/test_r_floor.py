#!/usr/bin/env python3
"""The viewsphere resolution path Stage B stopped calling but did not delete.

`derive_r_floor`'s pick rule — the count profiles are the measured sweeps
(block_report --rs, 2026-09-15) that motivated the per-run floor — and
`plan_to_cell`, whose only exercise died with its machine call site.

Both stay importable AND tested until the Stage D merge (ray-views Stage B
plan, Global Constraints): `run/machine.py` no longer calls either, so
without a direct test here a rewrite could break them silently and the
whole-branch review would have nothing to fail on.

Run: p inspection/tests/test_r_floor.py"""
from inspection.view.viewsphere import derive_r_floor


def counter(profile):
    """count_at stub from {radius: reachable_count}, tracking probes."""
    probed = []

    def count_at(r):
        probed.append(round(r, 4))
        return profile[round(r, 4)]

    return count_at, probed


def test_want_above_static_floor_is_untouched():
    count_at, probed = counter({})
    assert derive_r_floor(None, None, None, 0.31, count_at=count_at) == 0.31
    assert probed == []          # no probing when the floor is not in the way


def test_box_profile_floor_drops_to_020():
    # bench box 1509: flat 23/36 down to 0.20, cliff at 0.18
    count_at, _ = counter({0.24: 23, 0.22: 23, 0.20: 23, 0.19: 19})
    assert derive_r_floor(None, None, None, 0.19, count_at=count_at) == 0.20


def test_cup_profile_keeps_static_floor():
    # cup 2408-cup1: 27/36 at 0.24, cliff straight to 18 at 0.22
    count_at, probed = counter({0.24: 27, 0.22: 18})
    assert derive_r_floor(None, None, None, 0.20, count_at=count_at) == 0.24
    assert probed == [0.24, 0.22]   # stops at the first losing radius


def test_want_exactly_on_a_flat_profile_is_granted():
    count_at, _ = counter({0.24: 23, 0.22: 23, 0.20: 23})
    assert derive_r_floor(None, None, None, 0.20, count_at=count_at) == 0.20


# --- plan_to_cell: the shell planner the machine stopped calling -------------

CENTER = (0.05, -0.38, 0.05)


def _shell_world(buried=False):
    """A box on the table and the shell around it, or a world in which every
    approach to every cell is swallowed."""
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.viewsphere import ViewSphere
    w = RobotCell()
    if buried:
        w.set_object("object", [1.6, 1.6, 1.2],
                     [CENTER[0], CENTER[1], 0.60, 0, 0, 0])
    else:
        w.set_object("object", [0.15, 0.15, 0.10], list(CENTER) + [0, 0, 0])
    return w, UR5eIK(), ViewSphere(CENTER, r=0.30)


def test_plan_to_cell_returns_a_path_and_the_roll_that_earned_it():
    """The contract three callers used to read: `(path, report, roll)`, with
    the roll saying WHICH of the two the flange pose was reachable at — the
    tool hull is a cross with an off-axis camera slab, so half the shell is
    only reachable flipped."""
    import numpy as np
    from inspection.motion.plan import DEMO_PARK
    from inspection.view.viewsphere import ROLLS
    w, ik, sphere = _shell_world()

    path, rep, roll = sphere.plan_to_cell(w, ik, DEMO_PARK.copy(), 9, 0)

    assert path is not None and len(path) >= 2
    assert roll in ROLLS
    assert rep["tier"] in (1, 2)
    # The path really ends on that cell's flange pose at that roll, and it is
    # a validated path in this world — not merely a pair of configurations.
    assert np.allclose(ik.fk(path[-1]), sphere.flange_pose(9, 0, roll),
                       atol=1e-6)
    assert np.allclose(path[0], DEMO_PARK, atol=1e-9)
    ok, info = w.path_valid(path)
    assert ok, info


def test_plan_to_cell_refuses_when_every_roll_collides():
    """No roll with a collision-free endpoint -> `(None, rep, None)`, and
    `rep` is None because the planner was never reached: the roll walk
    rejected both poses on IK/collision alone. A caller reading `rep` for a
    reason has to handle that, which is why it is asserted rather than
    left to be discovered live."""
    from inspection.motion.plan import DEMO_PARK
    w, ik, sphere = _shell_world(buried=True)

    path, rep, roll = sphere.plan_to_cell(w, ik, DEMO_PARK.copy(), 3, 1)

    assert (path, rep, roll) == (None, None, None)


def main():
    test_want_above_static_floor_is_untouched()
    test_box_profile_floor_drops_to_020()
    test_cup_profile_keeps_static_floor()
    test_want_exactly_on_a_flat_profile_is_granted()
    test_plan_to_cell_returns_a_path_and_the_roll_that_earned_it()
    test_plan_to_cell_refuses_when_every_roll_collides()
    print("OK test_r_floor")


if __name__ == "__main__":
    main()
