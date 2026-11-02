"""TDD for StandPolicy (analytic ramp) and PolicyRunner (mode dispatch).

StandPolicy linearly interpolates from the captured spawn pose to `stand_pos` over a
fixed wall/sim duration (rate-independent, paced by the Observation stamp). PolicyRunner
selects WalkPolicy/StandPolicy by `Intent.mode` and re-seeds the entered policy on a
switch (so walk enters with a fresh 5-deep history from the current stance). Pure:
runs in the `brain` env (StandPolicy has no ONNX; PolicyRunner loads the walk ONNX).
"""

import numpy as np
import pytest

from humanoid.logic.oli import Intent, Mode, Observation, PolicyIn
from humanoid.logic.oli.action.policy_runner import PolicyRunner, StandPolicy

pytestmark = pytest.mark.brain

N = 31
SEC = 1_000_000_000


def _obs(stamp, q=None):
    q = np.zeros(N, dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32)
    return Observation(
        stamp_ns=stamp, q=q, dq=np.zeros(N), tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


@pytest.fixture(scope="module")
def stand():
    return StandPolicy()


def test_stand_params_loaded(stand):
    assert stand.stand_pos.shape == (N,)
    assert stand.stand_pos[3] == pytest.approx(0.30)   # left_knee
    assert stand.stand_kp[3] == pytest.approx(660)
    assert stand.stand_kd[3] == pytest.approx(8)


def test_stand_ramps_from_spawn_to_stand_pose(stand):
    init = np.full(N, -0.5, dtype=np.float32)  # arbitrary spawn pose
    stand.reset()
    out0 = stand.step(PolicyIn(observation=_obs(0, init), intent=Intent(mode=Mode.STAND)))
    np.testing.assert_allclose(out0.q_des, init, atol=1e-4)            # t=0 → spawn pose
    out_end = stand.step(PolicyIn(observation=_obs(2 * SEC, init), intent=Intent(mode=Mode.STAND)))
    np.testing.assert_allclose(out_end.q_des, stand.stand_pos, atol=1e-4)  # t=ramp → stand


def test_stand_midpoint_is_halfway(stand):
    init = np.full(N, -0.5, dtype=np.float32)
    stand.reset()
    stand.step(PolicyIn(observation=_obs(0, init), intent=Intent(mode=Mode.STAND)))
    mid = stand.step(PolicyIn(observation=_obs(SEC, init), intent=Intent(mode=Mode.STAND)))
    np.testing.assert_allclose(mid.q_des, 0.5 * (init + stand.stand_pos), atol=1e-3)


def test_stand_attaches_stand_gains(stand):
    stand.reset()
    out = stand.step(PolicyIn(observation=_obs(0), intent=Intent(mode=Mode.STAND)))
    np.testing.assert_allclose(out.kp, stand.stand_kp, atol=1e-3)
    np.testing.assert_allclose(out.kd, stand.stand_kd, atol=1e-3)


def test_runner_dispatches_by_mode():
    runner = PolicyRunner()
    out_stand = runner.step(PolicyIn(observation=_obs(1), intent=Intent(mode=Mode.STAND)))
    np.testing.assert_allclose(out_stand.kp, runner.stand.stand_kp, atol=1e-3)
    out_walk = runner.step(PolicyIn(
        observation=_obs(2, runner.walk.default_angle), intent=Intent(mode=Mode.WALK)))
    np.testing.assert_allclose(out_walk.kp, runner.walk.kp, atol=1e-3)


def test_runner_reseeds_walk_history_on_reentry():
    runner = PolicyRunner()
    for t in range(1, 4):  # prime walk: history accumulates distinct frames
        runner.step(PolicyIn(
            observation=_obs(t, runner.walk.default_angle + 0.01 * t),
            intent=Intent(mode=Mode.WALK)))
    runner.step(PolicyIn(observation=_obs(10), intent=Intent(mode=Mode.STAND)))  # interlude
    runner.step(PolicyIn(
        observation=_obs(11, runner.walk.default_angle), intent=Intent(mode=Mode.WALK)))
    h = runner.walk._history  # reset-on-reentry → first post-switch obs replicated ×5
    np.testing.assert_allclose(h[0:102], h[408:510], atol=1e-6)
