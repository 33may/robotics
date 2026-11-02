"""TDD for the arm-gated navigation reason (nav/arm.py).

Proves the gate: disarmed → the operator's joystick drives; armed → Nav drives toward the goal and
ignores the stick; armed with no goal holds still (safe). Pure: runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation
from humanoid.logic.oli.reason.localization import GroundTruthLocalizer, RobotPose
from humanoid.logic.oli.reason.mapping import OccupancyGrid, StaticMapping
from humanoid.logic.oli.reason.nav import ArmedNav, GoalCoordinate, Nav
from humanoid.logic.oli.reason.teleoperation.joystick.source import FixedJoystick
from humanoid.logic.oli.reason.teleoperation.joystick.teleop import Teleop

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _nav_at(x, y, yaw=0.0):
    loc = GroundTruthLocalizer(pose_reader=lambda: RobotPose(stamp_ns=0, x=x, y=y, yaw=yaw))
    return Nav(StaticMapping.from_grid(OccupancyGrid(np.zeros((10, 10), dtype=bool), 1.0)), loc)


def test_disarmed_operator_drives_via_joystick():
    arm = ArmedNav(Teleop(mode=Mode.WALK), _nav_at(0.5, 0.5))     # nav has no goal
    joy = FixedJoystick(v_x=0.3).poll()                           # push the stick forward
    pin = arm.to_policy_in(_obs(), joy)
    assert pin.intent.v_x > 0.0                                   # motion is the joystick's


def test_armed_nav_drives_toward_goal_ignoring_stick():
    nav = _nav_at(0.5, 0.5, yaw=0.0)
    nav.set_goal(GoalCoordinate(8.5, 0.5))                        # goal straight ahead (+x)
    arm = ArmedNav(Teleop(mode=Mode.WALK), nav)
    arm.set_armed(True)
    pin = arm.to_policy_in(_obs(), FixedJoystick(v_x=0.0).poll())  # stick centered → motion is Nav's
    assert pin.intent.v_x > 0.5


def test_armed_without_goal_holds_and_ignores_stick():
    arm = ArmedNav(Teleop(mode=Mode.WALK), _nav_at(0.5, 0.5))
    arm.set_armed(True)
    pin = arm.to_policy_in(_obs(), FixedJoystick(v_x=0.9).poll())  # stick shoved, but armed + no goal
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_arm_toggle_and_mode_delegates_to_teleop():
    tele = Teleop(mode=Mode.WALK)
    arm = ArmedNav(tele, _nav_at(0.5, 0.5))
    assert not arm.armed
    arm.set_armed(True)
    assert arm.armed
    arm.set_mode(Mode.STAND)
    assert arm.mode is Mode.STAND and tele.mode is Mode.STAND
