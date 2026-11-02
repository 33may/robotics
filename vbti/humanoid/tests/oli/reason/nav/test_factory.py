"""TDD for the Nav factory (nav/factory.py) — one tuned Nav recipe, every host.

The planner/controller knobs (footprint, clearance, speed caps) were tuned live in the glide
demo and used to live in dev_app's brain_link; the factory moves them INTO nav so every brain
host (`brain_main --service`, dev_app, later the real robot stack) builds the identical Nav.
Functional tests only — no private-attribute peeking. `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Observation
from humanoid.logic.oli.reason.localization import GroundTruthLocalizer, RobotPose
from humanoid.logic.oli.reason.mapping import OccupancyGrid, StaticMapping
from humanoid.logic.oli.reason.nav import GoalCoordinate, Nav
from humanoid.logic.oli.reason.nav.factory import build_nav

pytestmark = pytest.mark.brain


def _obs(stamp_ns=1):
    return Observation(
        stamp_ns=stamp_ns, q=np.zeros(31), dq=np.zeros(31), tau=np.zeros(31),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _mapping(n=40, res=0.5):
    return StaticMapping.from_grid(OccupancyGrid(np.zeros((n, n), dtype=bool), res))


def _loc_at(x, y, yaw=0.0):
    return GroundTruthLocalizer(pose_reader=lambda: RobotPose(stamp_ns=0, x=x, y=y, yaw=yaw))


def _vx_toward_goal(speed_scale):
    nav = build_nav(_mapping(), _loc_at(2.0, 2.0), speed_scale=speed_scale)
    nav.set_goal(GoalCoordinate(15.0, 2.0))  # long straight run along +x → cap-limited
    return nav.to_policy_in(_obs()).intent.v_x


def test_factory_builds_a_driving_nav():
    nav = build_nav(_mapping(), _loc_at(2.0, 2.0))
    assert isinstance(nav, Nav)
    nav.set_goal(GoalCoordinate(15.0, 2.0))
    assert nav.to_policy_in(_obs()).intent.v_x > 0.5


def test_speed_scale_predivides_the_caps():
    # GlideAction multiplies by speed_scale downstream, so the factory pre-divides the
    # controller caps: the PRODUCT (what Oli actually does) is scale-invariant.
    v1 = _vx_toward_goal(speed_scale=1.0)
    v2 = _vx_toward_goal(speed_scale=2.0)
    assert v2 == pytest.approx(v1 / 2.0, rel=1e-6)
