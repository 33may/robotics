"""TDD for the GlideModel — the glide "fake physics" (MAY-172).

Turns a commanded body-frame velocity into an integrated world-frame base pose, with a
per-axis acceleration limit so a joystick step ramps instead of teleporting (smooth
camera motion for SLAM). Pure kinematic integration — no isaacsim, runs in the `brain`
env. This is the exact seam the MuJoCo-fitted model (or the real walk) later replaces.
"""

import math

import pytest

from humanoid.logic.oli.glide import GlideModel

pytestmark = pytest.mark.brain


def test_at_rest_zero_command_does_not_move():
    m = GlideModel(lin_accel=2.0, yaw_accel=4.0)
    for _ in range(10):
        m.step(0.0, 0.0, 0.0, dt=0.01)
    assert (m.x, m.y, m.yaw) == (0.0, 0.0, 0.0)
    assert (m.vx, m.vy, m.wz) == (0.0, 0.0, 0.0)


def test_forward_velocity_ramps_at_accel_limit():
    m = GlideModel(lin_accel=2.0, yaw_accel=4.0)
    m.step(1.0, 0.0, 0.0, dt=0.1)   # may rise by at most 2.0 * 0.1 = 0.2
    assert m.vx == pytest.approx(0.2)
    m.step(1.0, 0.0, 0.0, dt=0.1)
    assert m.vx == pytest.approx(0.4)


def test_velocity_clamps_to_command_without_overshoot():
    m = GlideModel(lin_accel=100.0, yaw_accel=100.0)
    m.step(0.3, 0.0, 0.0, dt=0.1)   # accel huge → must land exactly on the command
    assert m.vx == pytest.approx(0.3)


def test_yaw_rate_ramps_at_yaw_accel_limit():
    m = GlideModel(lin_accel=2.0, yaw_accel=4.0)
    m.step(0.0, 0.0, 1.0, dt=0.1)   # rise by at most 4.0 * 0.1 = 0.4
    assert m.wz == pytest.approx(0.4)


def test_forward_motion_integrates_world_x_when_facing_x():
    m = GlideModel(lin_accel=1000.0, yaw_accel=1000.0)  # reach the command in one step
    m.step(0.5, 0.0, 0.0, dt=0.1)
    assert m.vx == pytest.approx(0.5)
    assert m.x == pytest.approx(0.05)   # 0.5 m/s × 0.1 s
    assert m.y == pytest.approx(0.0)
    assert m.yaw == pytest.approx(0.0)


def test_body_frame_forward_maps_to_world_y_when_yawed_90deg():
    m = GlideModel(lin_accel=1000.0, yaw_accel=1000.0, yaw=math.pi / 2)
    m.step(0.5, 0.0, 0.0, dt=0.1)       # body-forward at yaw=90° = +y in world
    assert m.x == pytest.approx(0.0, abs=1e-6)
    assert m.y == pytest.approx(0.05)


def test_velocity_decelerates_to_zero_without_undershoot():
    m = GlideModel(lin_accel=2.0, yaw_accel=4.0)
    for _ in range(100):
        m.step(0.5, 0.0, 0.0, dt=0.1)   # ramp up to and hold 0.5
    assert m.vx == pytest.approx(0.5)
    m.step(0.0, 0.0, 0.0, dt=0.1)       # decel by 0.2 → 0.3
    assert m.vx == pytest.approx(0.3)
    for _ in range(100):
        m.step(0.0, 0.0, 0.0, dt=0.1)
    assert m.vx == pytest.approx(0.0)   # lands exactly at 0, no negative undershoot
