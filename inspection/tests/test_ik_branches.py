#!/usr/bin/env python3
"""branches(q_near=...) must re-express IK solutions in the revolution the arm
actually lives in. 2026-08-31: wrist_2 wound to -5.05 rad made the survey
"unplannable" — every +-pi-wrapped goal branch was a 380 deg unwind away,
through the tool_cable_loop/forearm collision belt, while the 2pi-alias of the
same pose was a straight 20 deg move. Run: p inspection/tests/test_ik_branches.py"""
import numpy as np

from inspection.motion.ik import UR5eIK

# today's live scenario, frozen as a regression fixture
Q_WOUND = np.array([-1.127, -1.316, -1.944, -1.075, -5.048, 0.710])
Q_SURVEY = np.array([-1.234, -0.579, -2.022, -1.667, 1.580, 0.271])

# ur5e model limits (pinned against robot_descriptions in the world test below)
LIM = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])


def test_default_stays_wrapped():
    ik = UR5eIK()
    Q = ik.branches(ik.fk(Q_SURVEY))
    assert len(Q) > 0
    assert np.all(np.abs(Q) <= np.pi + 1e-9), "no q_near -> canonical +-pi"


def test_alias_lands_in_the_arms_revolution():
    ik = UR5eIK()
    T = ik.fk(Q_SURVEY)
    Q = ik.branches(T, q_near=Q_WOUND)
    assert len(Q) > 0
    # every branch still solves the SAME pose (aliasing is FK-invariant)
    for q in Q:
        assert np.allclose(ik.fk(q), T, atol=1e-6), "alias changed the pose"
    # each joint is the nearest LEGAL representative of its angle: within pi
    # of the arm, unless the closer alias would cross a joint limit (elbow
    # +-pi can never alias; a full analytic branch flip is genuinely far)
    lim = np.broadcast_to(LIM, Q.shape)
    far = np.abs(Q - Q_WOUND) > np.pi + 1e-9
    alias = Q + 2 * np.pi * np.round((Q_WOUND - Q) / (2 * np.pi))
    assert np.all(np.abs(alias[far]) > lim[far] + 1e-9), (
        f"joint stayed far although a legal alias exists:\n{np.round(Q, 2)}")
    # the wound wrist specifically must be re-expressed next to the arm
    assert np.any(np.abs(Q[:, 4] - Q_WOUND[4]) <= np.pi + 1e-9)
    # limits always respected
    assert np.all(np.abs(Q) <= LIM + 1e-9)


def test_alias_never_crosses_a_hard_stop():
    ik = UR5eIK()
    # arm parked 0.1 rad from the wrist_3 -2pi stop: a solution wrapped near
    # +pi has NO legal alias in that revolution and must stay wrapped
    q_edge = Q_WOUND.copy()
    q_edge[5] = -2 * np.pi + 0.1
    Q = ik.branches(ik.fk(Q_SURVEY), q_near=q_edge)
    assert np.all(Q >= -LIM - 1e-9) and np.all(Q <= LIM + 1e-9), (
        f"alias crossed a joint limit:\n{np.round(Q, 2)}")


def test_wound_wrist_survey_plans_tier1():
    """The full regression: today's wound start -> survey pose must plan."""
    from inspection.cell.world import RobotCell
    from inspection.motion.plan import plan_viewpoint

    world, ik = RobotCell(), UR5eIK()
    # pin the LIM table to the real model, so the constant cannot rot
    lo, hi = world.model.lowerPositionLimit, world.model.upperPositionLimit
    assert np.allclose(-LIM, lo) and np.allclose(LIM, hi)

    assert not world.is_colliding(Q_WOUND), "fixture start must be valid"
    assert not world.is_colliding(Q_SURVEY), "fixture goal must be valid"

    path, rep = plan_viewpoint(world, ik, Q_WOUND, ik.fk(Q_SURVEY), seed=0)
    assert path is not None, f"survey still unplannable: {rep}"
    assert rep["tier"] == 1, (
        f"expected the straight 20 deg move on tier 1, got {rep}")


if __name__ == "__main__":
    test_default_stays_wrapped()
    test_alias_lands_in_the_arms_revolution()
    test_alias_never_crosses_a_hard_stop()
    test_wound_wrist_survey_plans_tier1()
    print("ok")
