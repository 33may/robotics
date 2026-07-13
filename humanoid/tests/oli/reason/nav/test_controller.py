"""TDD for the pure-pursuit path follower (nav/controller.py).

Differential-drive with a 120° front cone: it never strafes or reverses. Within ±`front_cone` of
facing → drive forward + turn (forward speed eases as heading error grows); outside the cone →
rotate in place until aligned, then move. Stop on empty path / at goal. Pure: runs in `brain`.
"""

import math

import pytest

from humanoid.logic.oli.reason.nav import RobotPose
from humanoid.logic.oli.reason.nav.controller import PurePursuit

pytestmark = pytest.mark.brain


def _pose(x=0.0, y=0.0, yaw=0.0):
    return RobotPose(stamp_ns=0, x=x, y=y, yaw=yaw)


def test_empty_path_is_zero():
    assert PurePursuit().command(_pose(), []) == (0.0, 0.0, 0.0)


def test_at_goal_is_zero():
    pp = PurePursuit(goal_tol=0.15)
    assert pp.command(_pose(x=0.0, y=0.0), [(0.05, 0.0)]) == (0.0, 0.0, 0.0)


def test_goal_ahead_drives_forward():
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (5.0, 0.0)])
    assert v_x == pytest.approx(1.0)             # full forward, aligned
    assert v_y == pytest.approx(0.0, abs=1e-9)   # never strafes
    assert w_z == pytest.approx(0.0, abs=1e-9)


def test_never_strafes_or_reverses():
    """v_y is always 0 and v_x never negative — differential-drive, no lateral/back motion."""
    pp = PurePursuit(max_lin=1.0)
    for goal in [(0.0, 5.0), (-5.0, 0.0), (3.0, -3.0), (-2.0, 4.0)]:
        v_x, v_y, _ = pp.command(_pose(yaw=0.0), [(0.0, 0.0), goal])
        assert v_y == 0.0 and v_x >= 0.0


def test_goal_beyond_front_cone_rotates_in_place():
    # goal 90° to the left of facing (yaw=0) → outside the ±60° cone → pivot, no translation
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (0.0, 5.0)])
    assert v_x == 0.0 and v_y == 0.0
    assert w_z > 0.0                             # turns CCW toward +y


def test_goal_behind_rotates_in_place():
    pp = PurePursuit(max_lin=1.0, max_wz=1.5, k_yaw=1.5)
    v_x, v_y, w_z = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (-5.0, 0.0)])
    assert v_x == 0.0 and v_y == 0.0
    assert abs(w_z) == pytest.approx(1.5)        # saturated turn toward the target behind


def test_within_cone_moves_and_rotates():
    # goal ~30° to the left (inside the ±60° cone) → forward AND turning at once
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (4.33, 2.5)])  # atan2(2.5,4.33)=30°
    assert v_x > 0.0                             # moves
    assert w_z > 0.0                             # and turns toward the goal
    assert v_y == 0.0


def test_forward_speed_eases_as_heading_error_grows():
    pp = PurePursuit(max_lin=1.0)
    v_ahead, _, _ = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (5.0, 0.0)])      # 0°
    v_off, _, _ = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (4.33, 2.5)])       # 30°
    assert v_ahead > v_off > 0.0                 # more aligned = faster


def test_speed_capped_when_far():
    pp = PurePursuit(max_lin=0.7)
    v_x, _, _ = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (10.0, 0.0)])
    assert v_x == pytest.approx(0.7)


def test_speed_eases_near_goal():
    pp = PurePursuit(max_lin=1.0, goal_tol=0.1, lookahead=0.2)
    v_x, _, _ = pp.command(_pose(yaw=0.0), [(0.5, 0.0)])  # 0.5 m out, below max, aligned
    assert v_x == pytest.approx(0.5)


def test_aligned_when_facing_travel_drives_forward():
    # Robot faces +y (yaw=pi/2); goal in world +y → aligned → forward, no turn, no strafe.
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=math.pi / 2), [(0.0, 0.0), (0.0, 5.0)])
    assert v_x == pytest.approx(1.0)
    assert v_y == pytest.approx(0.0, abs=1e-9)
    assert w_z == pytest.approx(0.0, abs=1e-9)
