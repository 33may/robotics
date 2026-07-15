"""Tests for the coverage-drive WaypointFollower (MAY-173 locdev T2).

Pure module (numpy + stdlib): GT pose (x, y, yaw) → body-frame twist (vx, vy, wz)
chasing a committed waypoint list. It is the swappable MOTION SOURCE of the
data-collection flow — the recorder never knows what produces the twist, so the
walk-motion mimic can replace/augment this later without touching the recorder.
Runs in the `brain` env (no isaacsim).
"""

import math

import numpy as np
import pytest

from humanoid.logic.simulation.mapping.waypoint_follower import WaypointFollower

pytestmark = pytest.mark.brain


def test_drives_forward_toward_waypoint_ahead():
    f = WaypointFollower([(5.0, 0.0)], speed=0.8)
    vx, vy, wz = f.command(0.0, 0.0, 0.0)
    assert vx == pytest.approx(0.8, abs=1e-6)  # aligned → full speed
    assert vy == 0.0
    assert wz == pytest.approx(0.0, abs=1e-6)
    assert not f.done


def test_turns_left_toward_lateral_waypoint():
    f = WaypointFollower([(0.0, 5.0)], speed=0.8)
    vx, _, wz = f.command(0.0, 0.0, 0.0)  # target 90° to the left
    assert wz > 0.0
    assert vx < 0.2  # large heading error → creep, don't charge


def test_yaw_error_wraps_across_pi():
    # Robot at yaw ≈ +π, waypoint straight behind its nose direction (i.e. ahead
    # in world -X): error must wrap to ~0, not spin the long way around.
    f = WaypointFollower([(-5.0, 0.0)])
    _, _, wz = f.command(0.0, 0.0, math.pi - 1e-3)
    assert abs(wz) < 0.1


def test_wz_is_clipped():
    f = WaypointFollower([(0.0, 5.0)], yaw_gain=10.0, max_wz=1.0)
    _, _, wz = f.command(0.0, 0.0, 0.0)
    assert wz == pytest.approx(1.0)


def test_arrival_advances_to_next_waypoint():
    f = WaypointFollower([(1.0, 0.0), (1.0, 5.0)], arrive_radius=0.5)
    f.command(0.9, 0.0, 0.0)  # within radius of wp0 → advance
    assert f.index == 1
    _, _, wz = f.command(0.9, 0.0, 0.0)
    assert wz > 0.0  # now steering toward (1, 5)


def test_done_after_last_waypoint_and_commands_zero():
    f = WaypointFollower([(1.0, 0.0)], arrive_radius=0.5)
    f.command(0.9, 0.0, 0.0)
    assert f.done
    assert f.command(0.9, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_loop_wraps_instead_of_finishing():
    f = WaypointFollower([(1.0, 0.0), (1.0, 1.0)], arrive_radius=0.5, loop=True)
    f.command(0.9, 0.0, 0.0)   # arrive wp0 → wp1
    f.command(0.9, 0.9, 0.0)   # arrive wp1 → wraps to wp0
    assert f.index == 0
    assert not f.done


def test_module_is_pure():
    import sys

    import humanoid.logic.simulation.mapping.waypoint_follower  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
