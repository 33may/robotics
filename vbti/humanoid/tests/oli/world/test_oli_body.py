"""Isaac-env smoke test for the slimmed Oli body (the BodyLike contract).

No mocks — boots a real headless Isaac, materializes Oli, and checks the interface
SimComm depends on: dof_names (31), read_joints_isaac / read_imu shapes, and that
apply_isaac holds the pose without diverging across a few hundred steps. Marked
`isaac` → skipped in the brain env, run via `conda run -n isaac pytest`.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.isaac

N = 31


@pytest.fixture(scope="module")
def isaac_world():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World  # noqa: E402 (must follow SimulationApp)
    world = World(stage_units_in_meters=1.0)
    yield world
    app.close()


def test_oli_body_contract(isaac_world):
    from humanoid.logic.simulation.isaacsim.oli import NUM_JOINTS, Oli

    oli = Oli(isaac_world)
    assert len(oli.dof_names) == NUM_JOINTS

    q, dq, tau = oli.read_joints_isaac()
    assert q.shape == (NUM_JOINTS,) and dq.shape == (NUM_JOINTS,) and tau.shape == (NUM_JOINTS,)

    acc, gyro, quat = oli.read_imu()
    assert acc.shape == (3,) and gyro.shape == (3,) and quat.shape == (4,)

    # set_joint_state writes the physics state (used to spawn the home crouch)
    target = np.full(NUM_JOINTS, 0.05, dtype=np.float32)  # within every joint limit
    oli.set_joint_state(target)
    q2, dq2, _ = oli.read_joints_isaac()
    np.testing.assert_allclose(q2, target, atol=1e-3)
    assert np.abs(dq2).max() < 1e-3, "set_joint_state must zero velocities"


def test_oli_apply_holds_pose_without_diverging(isaac_world):
    from humanoid.logic.simulation.isaacsim.oli import NUM_JOINTS, Oli

    oli = Oli(isaac_world, prim_path="/World/Oli2")
    q0, _, _ = oli.read_joints_isaac()
    # hold the spawn pose with moderate gains via the implicit drive
    oli.apply_isaac(
        q_des=q0, dq_des=np.zeros(N), tau_ff=np.zeros(N),
        kp=np.full(N, 100.0), kd=np.full(N, 5.0),
    )
    for _ in range(200):
        isaac_world.step(render=False)
    q1, dq1, _ = oli.read_joints_isaac()
    assert np.isfinite(q1).all(), "joint positions diverged to NaN/inf"
    assert np.abs(dq1).max() < 50.0, "velocities exploded (PD unstable)"
