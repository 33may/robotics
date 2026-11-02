"""imu_probe.py — does Oli's Isaac IMU track the body's true orientation?

Suspected root cause of the walk fall: the IMU-derived `projected_gravity` (which way is
down, body frame) is wrong, so the policy is blind to its tilt and walks itself over.

Two tests, both comparing the IMU against `get_world_pose` (the body's TRUE orientation):
  1. teleport — set the base to known tilts (gravity off) and read back.
  2. dynamics — let the robot collapse/tip under gravity (real motion, like the walk loop)
     and watch whether the IMU quat follows the true base quat.

Runs in the `isaac` env:
    conda run -n isaac python humanoid/logic/simulation/isaacsim/imu_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from humanoid.logic.simulation.isaacsim.oli import NUM_JOINTS, Oli  # noqa: E402

_G = np.array([0.0, 0.0, -1.0])


def _wxyz(rx, ry, rz):
    q = R.from_euler("xyz", [rx, ry, rz]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def proj_grav(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return R.from_quat([q[1], q[2], q[3], q[0]]).inv().as_matrix() @ _G


def quat_angle_deg(a, b):
    """Angle between two wxyz quaternions, degrees (0 = identical orientation)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


ORIENT = {
    "pitch+0.3": _wxyz(0.0, 0.3, 0.0),
    "roll+0.3":  _wxyz(0.3, 0.0, 0.0),
    "mix":       _wxyz(0.2, -0.25, 0.4),
}

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 1000.0, rendering_dt=1.0 / 50.0)
world.scene.add_default_ground_plane()
oli = Oli(world, pin_root=False, spawn_pose=(0.0, 0.0, 3.0))
phys = world.get_physics_context()

print("\n=== TELEPORT: set base tilt (gravity off), IMU quat vs TRUE quat ===", flush=True)
phys.set_gravity(0.0)
for name, q in ORIENT.items():
    oli.set_base_pose((0.0, 0.0, 3.0), q)
    for _ in range(10):
        world.step(render=False)
    true_q = oli.base_world_quat_wxyz()
    _, gyro, imu_q = oli.read_imu()
    print(f"[{name:9s}] set={q.round(3)} TRUE(get_world_pose)={true_q.round(3)} "
          f"IMU={imu_q.round(3)}  IMU-vs-TRUE={quat_angle_deg(imu_q, true_q):.1f}deg",
          flush=True)

print("\n=== GYRO FRAME: is get_angular_velocity world or body? ===", flush=True)
# Yaw the base 90° about world-z, then spin about WORLD +y. In body frame (after +90° yaw)
# a world +y vector reads as body +x. So: raw≈[0,1,0] → WORLD frame (read_imu rotation is
# correct, gyro→[1,0,0]). raw≈[1,0,0] → already BODY frame → read_imu DOUBLE-rotates (BUG).
yaw90 = _wxyz(0.0, 0.0, np.pi / 2)
oli.set_base_pose((0.0, 0.0, 3.0), yaw90)
oli.set_base_velocity((0, 0, 0), (0.0, 1.0, 0.0))  # world-frame angular velocity about +y
for _ in range(3):
    world.step(render=False)
raw = np.asarray(oli._art.get_angular_velocity(), dtype=np.float32).reshape(-1)[:3]
_, gyro, quat = oli.read_imu()
print(f"set world_w=[0,1,0] @ yaw90:  get_angular_velocity(raw)={raw.round(3)}  "
      f"read_imu gyro(rotated)={np.asarray(gyro).round(3)}", flush=True)
if np.allclose(raw, [0, 1, 0], atol=0.2):
    print("  => get_angular_velocity is WORLD frame; read_imu rotation is CORRECT "
          f"(gyro should be ~[1,0,0]; got {np.asarray(gyro).round(2)})", flush=True)
elif np.allclose(raw, [1, 0, 0], atol=0.2):
    print("  => get_angular_velocity is BODY frame; read_imu DOUBLE-ROTATES — BUG, "
          "should NOT rotate", flush=True)
else:
    print(f"  => inconclusive raw={raw.round(3)}", flush=True)

app.close()
