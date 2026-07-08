"""TDD for the World's spawn-pose path (option A: World owns the init condition).

Two pure pieces, both runnable in the `brain` env (no isaacsim):
  - `SimComm.pr_to_isaac_vector` — permute a PR-ordered vector to Isaac DOF order,
    the same PR→Isaac transform `apply()` uses. The World needs it to place a
    PR-space home pose onto the Isaac articulation (D4: SimComm owns permutation).
  - `HOME_POSE_PR` (sim_world_main) — the crouch Oli spawns into. It MUST equal the
    walk policy's `default_angle`, because the policy's whole observation is
    (q − default_angle); spawning anywhere else hands the policy a non-zero pose.
"""

import numpy as np
import pytest
import yaml

from humanoid.logic.oli import NUM_JOINTS, PR_ORDER
from humanoid.logic.oli.action.policy_runner import DEFAULT_WALK_PARAM
from humanoid.logic.simulation.isaacsim.sim_comm import SimComm
from humanoid.logic.simulation.isaacsim.sim_world_main import HOME_POSE_PR

pytestmark = pytest.mark.brain

N = NUM_JOINTS


class _NamesOnlyBody:
    """SimComm only touches `dof_names` at construction. Isaac order = reversed PR,
    so the permutation is non-trivial: isaac index of PR joint pr is (N-1-pr)."""

    dof_names = list(reversed(PR_ORDER))


def test_pr_to_isaac_vector_permutes_into_isaac_order():
    sc = SimComm(_NamesOnlyBody(), socket_path="/tmp/oli-test-unused.sock")
    pr_vec = np.arange(N, dtype=np.float32)  # PR order: v[pr] = pr
    isaac_vec = sc.pr_to_isaac_vector(pr_vec)
    # reversed Isaac order → isaac[i] holds PR joint (N-1-i) → value N-1-i
    expected = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
    np.testing.assert_allclose(isaac_vec, expected, atol=1e-6)
    assert isaac_vec.shape == (N,)


def test_home_pose_pr_matches_walk_default_angle():
    home = np.asarray(HOME_POSE_PR, dtype=np.float32)
    assert home.shape == (N,), "HOME_POSE_PR must list exactly 31 joints"
    with open(DEFAULT_WALK_PARAM) as f:
        default_angle = np.asarray(
            yaml.safe_load(f)["HumanoidRobotCfg"]["control"]["default_angle"],
            dtype=np.float32,
        )
    # default_angle differs from PR order only at the two head slots (15,16), both
    # 0.0, so an elementwise compare is exact. This guards against drift.
    np.testing.assert_allclose(home, default_angle, atol=1e-6)
