"""TDD for the pure-pursuit path follower (nav/controller.py).

Checks the body-frame twist it produces: stop on empty path / at goal, drive forward when the
goal is ahead, strafe+turn when the goal is to the side, cap at max speed when far, and ease
below max speed near the goal. Pure: runs in `brain`.
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
    assert v_x == pytest.approx(1.0)        # full forward
    assert v_y == pytest.approx(0.0, abs=1e-9)
    assert w_z == pytest.approx(0.0, abs=1e-9)  # already aligned


def test_goal_to_left_strafes_and_turns():
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (0.0, 5.0)])
    assert v_y == pytest.approx(1.0)   # +y is left in the body frame (yaw=0)
    assert v_x == pytest.approx(0.0, abs=1e-9)
    assert w_z > 0.0                   # turns CCW to face +y travel


def test_speed_capped_when_far():
    pp = PurePursuit(max_lin=0.7)
    v_x, _, _ = pp.command(_pose(yaw=0.0), [(0.0, 0.0), (10.0, 0.0)])
    assert v_x == pytest.approx(0.7)


def test_speed_eases_near_goal():
    pp = PurePursuit(max_lin=1.0, goal_tol=0.1, lookahead=0.2)
    v_x, _, _ = pp.command(_pose(yaw=0.0), [(0.5, 0.0)])  # 0.5 m out, below max
    assert 0.0 < v_x < 1.0
    assert v_x == pytest.approx(0.5)


def test_body_frame_rotation_when_facing_plus_y():
    # Robot faces +y (yaw=pi/2); goal straight ahead in world +y → forward in body frame.
    pp = PurePursuit(max_lin=1.0)
    v_x, v_y, w_z = pp.command(_pose(yaw=math.pi / 2), [(0.0, 0.0), (0.0, 5.0)])
    assert v_x == pytest.approx(1.0)          # travel is straight ahead in body frame
    assert v_y == pytest.approx(0.0, abs=1e-9)
    assert w_z == pytest.approx(0.0, abs=1e-9)  # already facing travel
