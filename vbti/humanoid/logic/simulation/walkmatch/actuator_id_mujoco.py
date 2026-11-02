"""actuator_id_mujoco.py — MuJoCo side of the sim-to-sim actuator step-response ID.

Runs in the `limx` env (mujoco 3.2.3). Loads the HU_D04_01 MJCF, but configures it to be
STRUCTURALLY IDENTICAL to the Isaac serial model so the actuator comparison is clean:

  * achilles A/B EQUALITY constraints DISABLED → the serial ankle is a normal joint (this is
    the IsaacLab *training* topology, not the parallel deploy topology),
  * CONTACT + GRAVITY disabled, base PINNED in the air → isolate the PD step response,
  * all 31 serial PR joints driven by EXPLICIT torque PD via `qfrc_applied` (τ=kp·err+kd·erṙ),
    exactly what the deploy/legged_gym does and what Isaac's `apply_torque_isaac` does.

MuJoCo's integrator is the trusted reference: its step response is the gold standard Isaac
must reproduce. Same `walkmatch.spec` step protocol as the Isaac harness.

    conda run -n limx python logic/simulation/walkmatch/actuator_id_mujoco.py \
        --target left_knee_joint --amp 0.4 --out /tmp/actid_mujoco_knee.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from humanoid.logic.simulation.walkmatch import spec  # noqa: E402

XML = (_REPO / "humanoid" / "vendor" / "humanoid-mujoco-sim" / "humanoid-description"
       / "HU_D04_description" / "xml" / "HU_D04_01.xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="left_knee_joint")
    ap.add_argument("--amp", type=float, default=0.4)
    ap.add_argument("--out", default="/tmp/actid_mujoco.json")
    ap.add_argument("--gravity", action="store_true", help="keep gravity ON (default OFF)")
    ap.add_argument("--keep-loop", action="store_true",
                    help="keep the achilles equality loop (default: disabled → serial)")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    m.opt.timestep = spec.DT
    if not args.gravity:
        m.opt.gravity[:] = 0.0
    # Disable contacts always; disable the achilles equality loop unless asked to keep it.
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if not args.keep_loop:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)

    # Joint address tables (PR order) — qpos slot + dof (qvel) slot per serial PR joint.
    qadr = np.array([m.joint(n).qposadr[0] for n in spec.PR_ORDER], dtype=np.int64)
    vadr = np.array([m.joint(n).dofadr[0] for n in spec.PR_ORDER], dtype=np.int64)
    target_pr = spec.PR_ORDER.index(args.target)
    q0 = float(spec.DEFAULT[target_pr])
    q_final = q0 + args.amp

    # Initial condition: base high + upright (pinned), all serial PR joints at default.
    mujoco.mj_resetData(m, d)
    base_q = m.joint("floating_base_joint").qposadr[0]
    base_v = m.joint("floating_base_joint").dofadr[0]
    d.qpos[base_q:base_q + 7] = [0, 0, 1.1, 1, 0, 0, 0]
    d.qpos[qadr] = spec.DEFAULT
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    base_pose = d.qpos[base_q:base_q + 7].copy()

    t_list, q_list, dq_list, qd_list = [], [], [], []
    kp, kd = spec.KP_LEG, spec.KD_LEG  # drive only the 8 leg joints (ankle excluded)
    for tick in range(spec.N_STEPS):
        q_des = spec.q_des_at(tick, target_pr, args.amp)        # PR order
        q = d.qpos[qadr]
        dq = d.qvel[vadr]
        tau = kp * (q_des - q) + kd * (0.0 - dq)                # explicit PD, PR order
        d.qfrc_applied[vadr] = tau
        mujoco.mj_step(m, d)
        # Re-pin the base (its free joint drifts under leg reaction forces).
        d.qpos[base_q:base_q + 7] = base_pose
        d.qvel[base_v:base_v + 6] = 0.0
        t_list.append(tick * spec.DT)
        q_list.append(float(d.qpos[qadr[target_pr]]))
        dq_list.append(float(d.qvel[vadr[target_pr]]))
        qd_list.append(float(q_des[target_pr]))

    t0 = spec.N_HOLD
    metrics = spec.step_metrics(
        np.array(t_list[t0:]) - spec.T_STEP, np.array(q_list[t0:]), q0, q_final)
    print(f"[actid] mujoco {args.target:16s} "
          f"onset={metrics.get('onset_lag_ms', float('nan')):6.1f}ms "
          f"rise={metrics.get('rise_time_ms', float('nan')):6.1f}ms "
          f"overshoot={metrics.get('overshoot_pct', float('nan')):5.1f}% "
          f"ss_err={metrics.get('ss_err_rad', float('nan')):+.4f}rad", flush=True)

    out = {
        "meta": {"sim": "mujoco", "target": args.target, "amp": args.amp,
                 "q0": q0, "q_final": q_final, "dt": spec.DT, "t_step": spec.T_STEP,
                 "gravity": args.gravity, "loop": args.keep_loop},
        "configs": {"mujoco": {"t": t_list, "q": q_list, "dq": dq_list,
                               "qdes": qd_list, "metrics": metrics}},
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"[actid] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
