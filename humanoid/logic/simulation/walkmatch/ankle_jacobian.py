"""ankle_jacobian.py — measure the achilles-linkage Jacobian + effective PR-space ankle
stiffness in MuJoCo (the real closed loop). A walkmatch system-ID tool (limx env, py3.8).

WHY THIS EXISTS (MAY-147, 2026-07-01): the walk policy trains/deploys against the PARALLEL
achilles ankle (two motors A/B per foot, PD in MOTOR space at kp=93.65). Our Isaac model has
a single SERIAL pitch/roll ankle. To emulate the real ankle we need the effective JOINT-space
stiffness the dual-motor linkage produces:

    q = (ankle_pitch, ankle_roll);   m = (A, B) motor angles;   m = h(q)
    G = d(pitch,roll)/d(A,B)      (measured here by perturbing each motor)
    J = G^-1 = d(A,B)/d(pitch,roll)
    tau_q = J^T tau_m = J^T [ kp (m_des - m) + kd (mdot_des - mdot) ]
          ~= J^T diag(kp) J (q_des - q) + J^T diag(kd) J (qdot_des - qdot)
    => effective joint stiffness  K = kp * (J^T J)   (2x2, symmetric)
       effective joint damping    D = kd * (J^T J)

The OFF-DIAGONAL of K is the pitch<->roll coupling a scalar --ankle-kp-scale cannot express.
This tool quantifies it. RESULT (see NOTEBOOK F11/F12): coupling ~= 0 across the whole ankle
range; K is diagonal ~x2.30 pitch / x2.04 roll. So the "coupled 2x2 PD" collapses to the
existing diagonal --ankle-kp-scale/--ankle-roll-scale flags — no coupled-matrix code is warranted.

Run:  conda run -n limx python logic/simulation/walkmatch/ankle_jacobian.py [--sweep]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco

_XML = (Path(__file__).resolve().parents[3] / "vendor" / "humanoid-mujoco-sim" /
        "humanoid-description" / "HU_D04_description" / "xml" / "HU_D04_01.xml")

KP_ANKLE, KD_ANKLE = 93.65, 11.92  # per-motor deploy gains (walk_param.yaml ankle)


def _load():
    m = mujoco.MjModel.from_xml_path(str(_XML))
    d = mujoco.MjData(m)
    m.opt.timestep = 0.001
    m.opt.gravity[:] = 0.0                                   # isolate the linkage kinematics
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return m, d


def _measure_G(m, d, side: str, tA: float, tB: float, dl: float = 0.10):
    """Local G = d(pitch,roll)/d(A,B) at motor operating point (tA,tB), by central differences.
    Pins the floating base each substep so only the ankle loop moves. Returns (G, ankle_pose)."""
    qa = lambda n: m.joint(n).qposadr[0]
    va = lambda n: m.joint(n).dofadr[0]
    An, Bn = f"{side}_A_achilles_joint", f"{side}_B_achilles_joint"
    APn, ARn = f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"
    actA, actB = m.actuator(f"ankle_A_{side}").id, m.actuator(f"ankle_B_{side}").id
    bq = m.joint("floating_base_joint").qposadr[0]

    mujoco.mj_resetData(m, d)
    d.qpos[bq:bq + 7] = [0, 0, 1.1, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    base = d.qpos[bq:bq + 7].copy()

    def settle(a, b, n=2500):
        for _ in range(n):
            qA, qB = d.qpos[qa(An)], d.qpos[qa(Bn)]
            vA, vB = d.qvel[va(An)], d.qvel[va(Bn)]
            d.ctrl[actA] = np.clip(80 * (a - qA) - 3 * vA, -42, 42)
            d.ctrl[actB] = np.clip(80 * (b - qB) - 3 * vB, -42, 42)
            mujoco.mj_step(m, d)
            d.qpos[bq:bq + 7] = base
            d.qvel[bq:bq + 6] = 0

    read = lambda: np.array([d.qpos[qa(APn)], d.qpos[qa(ARn)]])
    settle(tA, tB); pose = read()
    settle(tA + dl, tB); pAp = read(); settle(tA - dl, tB); pAm = read()
    settle(tA, tB + dl); pBp = read(); settle(tA, tB - dl); pBm = read()
    colA, colB = (pAp - pAm) / (2 * dl), (pBp - pBm) / (2 * dl)
    return np.column_stack([colA, colB]), pose


def effective_stiffness(G):
    """K/kp and D/kd = J^T J  (J = G^-1). Returns the 2x2 J^T J matrix."""
    J = np.linalg.inv(G)
    return J.T @ J


def _report_point(m, d, side, flag):
    A0 = float(d.qpos[m.joint(f"{side}_A_achilles_joint").qposadr[0]])
    B0 = float(d.qpos[m.joint(f"{side}_B_achilles_joint").qposadr[0]])
    G, pose = _measure_G(m, d, side, A0, B0)
    JTJ = effective_stiffness(G)
    print(f"\n===== {side} ankle ({flag}) @ home pose (pitch={pose[0]:+.3f}, roll={pose[1]:+.3f}) =====")
    print(f"K_pitch = kp*{JTJ[0,0]:.3f} = {KP_ANKLE*JTJ[0,0]:.1f}  (x{JTJ[0,0]:.2f})")
    print(f"K_roll  = kp*{JTJ[1,1]:.3f} = {KP_ANKLE*JTJ[1,1]:.1f}  (x{JTJ[1,1]:.2f})")
    print(f"K_coupling (off-diag) = kp*{JTJ[0,1]:+.4f} = {KP_ANKLE*JTJ[0,1]:+.2f}  "
          f"(coupling frac {JTJ[0,1]/JTJ[0,0]:+.3f})")


def _report_sweep(m, d, side="left"):
    print(f"\n===== {side} ankle: coupling across the operating range =====")
    print("pitch   roll  | pitch_scale roll_scale coupling")
    for tA in np.linspace(-0.5, 0.5, 5):
        for tB in np.linspace(-0.5, 0.5, 5):
            G, pose = _measure_G(m, d, side, tA, tB)
            if abs(pose[0]) > 0.6 or abs(pose[1]) > 0.43:   # skip singular joint-limit poses
                continue
            JTJ = effective_stiffness(G)
            print(f"{pose[0]:+.3f} {pose[1]:+.3f} |   {JTJ[0,0]:6.3f}    {JTJ[1,1]:6.3f}   {JTJ[0,1]:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="also sweep coupling over the ankle range")
    args = ap.parse_args()
    m, d = _load()
    _report_point(m, d, "left", "left_flag=+1")
    _report_point(m, d, "right", "left_flag=-1")
    if args.sweep:
        _report_sweep(m, d, "left")


if __name__ == "__main__":
    main()
