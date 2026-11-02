"""imu_probe.py — MuJoCo reference IMU readings (the path that WALKS).

Counterpart to `logic/simulation/isaacsim/imu_probe.py`. Sets the HU_D04_01 MuJoCo model
to the SAME known base orientations / spins and reads the `Body_Quat` / `Body_Gyro`
sensors the deploy uses, comparing against the analytic ground truth. Confirms the
deploy's IMU convention and gives the reference to align Isaac against. Runs in `limx`
(mujoco 3.2.3):

    conda run -n limx python humanoid/logic/simulation/mujoco/imu_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

_HUMANOID = Path(__file__).resolve().parents[3]  # .../humanoid
_MJCF = (_HUMANOID / "vendor" / "humanoid-mujoco-sim" / "humanoid-description"
         / "HU_D04_description" / "xml" / "HU_D04_01.xml")
_G = np.array([0.0, 0.0, -1.0])


def _wxyz(rx, ry, rz):
    q = R.from_euler("xyz", [rx, ry, rz]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # -> wxyz


def proj_grav(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64)
    xyzw = [q[1], q[2], q[3], q[0]]
    return R.from_quat(xyzw).inv().as_matrix() @ _G


ORIENT = {
    "upright":   _wxyz(0.0, 0.0, 0.0),
    "pitch+0.3": _wxyz(0.0, 0.3, 0.0),
    "pitch-0.3": _wxyz(0.0, -0.3, 0.0),
    "roll+0.3":  _wxyz(0.3, 0.0, 0.0),
    "yaw+0.5":   _wxyz(0.0, 0.0, 0.5),
    "mix":       _wxyz(0.2, -0.25, 0.4),
}
SPIN = {
    "wx+1.0": np.array([1.0, 0.0, 0.0]),
    "wy+1.0": np.array([0.0, 1.0, 0.0]),
    "wz+1.0": np.array([0.0, 0.0, 1.0]),
}

model = mujoco.MjModel.from_xml_path(str(_MJCF))
data = mujoco.MjData(model)


def sensor(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr, dim = model.sensor_adr[sid], model.sensor_dim[sid]
    return np.array(data.sensordata[adr:adr + dim])


results = {"orientation": {}, "gyro": {}}

print("=== ORIENTATION → projected_gravity (MuJoCo Body_Quat) ===", flush=True)
for name, q in ORIENT.items():
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = q  # free-joint quaternion (MuJoCo wxyz)
    mujoco.mj_forward(model, data)
    bq = sensor("Body_Quat")
    pg_imu, pg_true = proj_grav(bq), proj_grav(q)
    match = bool(np.allclose(pg_imu, pg_true, atol=2e-2))
    results["orientation"][name] = {
        "set_quat_wxyz": q.round(4).tolist(),
        "imu_quat_wxyz": bq.round(4).tolist(),
        "pg_from_imu": pg_imu.round(4).tolist(),
        "pg_analytic": pg_true.round(4).tolist(),
        "match": match,
    }
    print(f"[{name:9s}] set_q={q.round(3)} imu_q={bq.round(3)} "
          f"pg_imu={pg_imu.round(3)} pg_true={pg_true.round(3)} "
          f"{'OK' if match else 'MISMATCH <<<'}", flush=True)

print("\n=== ANGULAR VELOCITY → Body_Gyro ===", flush=True)
for name, w in SPIN.items():
    mujoco.mj_resetData(model, data)
    data.qvel[3:6] = w  # free-joint angular velocity (MuJoCo: local frame)
    mujoco.mj_forward(model, data)
    gyro = sensor("Body_Gyro")
    match = bool(np.allclose(gyro, w, atol=0.15))
    results["gyro"][name] = {
        "set_w": w.round(3).tolist(), "imu_gyro": gyro.round(3).tolist(), "match": match,
    }
    print(f"[{name}] set_w={w.round(2)} imu_gyro={gyro.round(3)} "
          f"{'OK' if match else 'MISMATCH <<<'}", flush=True)

print("\nJSON " + json.dumps(results), flush=True)
