"""TDD for the Nav reason module (nav/nav.py) — the localize→plan→follow→intent composition.

Verifies it behaves as a drop-in Reason module: emits `PolicyIn` wrapping the given Observation,
drives toward a set goal, and holds zero-velocity when there is no goal / no pose / no path. Pure:
runs in `brain`.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Mode, Observation, PolicyIn
from humanoid.logic.oli.reason.nav import GroundTruthLocalizer, OccupancyGrid, RobotPose
from humanoid.logic.oli.reason.nav.nav import Nav

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _empty_map(n=10, res=1.0):
    return OccupancyGrid(np.zeros((n, n), dtype=bool), res)


def _loc_at(x, y, yaw=0.0):
    return GroundTruthLocalizer(pose_reader=lambda: RobotPose(stamp_ns=0, x=x, y=y, yaw=yaw))


def test_no_goal_holds_zero_velocity():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    pin = nav.to_policy_in(_obs())
    assert isinstance(pin, PolicyIn)
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)
    assert pin.intent.mode == Mode.WALK


def test_policy_in_wraps_the_given_observation():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    obs = _obs(stamp_ns=42)
    assert nav.to_policy_in(obs).observation is obs


def test_no_pose_yet_holds():
    nav = Nav(_empty_map(), GroundTruthLocalizer(pose_reader=lambda: None))
    nav.set_goal(8.5, 0.5)
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_drives_toward_goal_ahead():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5, yaw=0.0))
    nav.set_goal(8.5, 0.5)  # straight along +x
    pin = nav.to_policy_in(_obs())
    assert pin.intent.v_x > 0.5
    assert pin.intent.mode == Mode.WALK


def test_stops_when_at_goal():
    nav = Nav(_empty_map(), _loc_at(8.5, 0.5))
    nav.set_goal(8.5, 0.5)
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_unreachable_goal_holds():
    arr = np.zeros((10, 10), dtype=bool)
    arr[0, 8] = True  # goal cell occupied → no path
    nav = Nav(OccupancyGrid(arr, 1.0), _loc_at(0.5, 0.5))
    nav.set_goal(8.5, 0.5)
    pin = nav.to_policy_in(_obs())
    assert (pin.intent.v_x, pin.intent.v_y, pin.intent.w_z) == (0.0, 0.0, 0.0)


def test_clear_goal_returns_to_hold():
    nav = Nav(_empty_map(), _loc_at(0.5, 0.5))
    nav.set_goal(8.5, 0.5)
    assert nav.to_policy_in(_obs()).intent.v_x > 0.0
    nav.clear_goal()
    assert nav.to_policy_in(_obs()).intent.v_x == 0.0
