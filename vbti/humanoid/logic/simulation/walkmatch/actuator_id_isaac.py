"""actuator_id_isaac.py — Isaac side of the sim-to-sim actuator step-response ID.

Runs in the `isaac` env. Boots Isaac ONCE, spawns Oli with the base PINNED in the air and
GRAVITY OFF (isolating a single joint's PD step response from contact + balance + gravity
load), then sweeps actuator configs {implicit,explicit} x {armature off,on}. For each
config it holds all 31 joints at the policy default with the policy's kp/kd, steps the
TARGET joint by `--amp` at t=100 ms, and records the target joint's q(t)/dq(t).

Output: JSON {meta, configs:{name:{t,q,dq,qdes}}} → compare.py overlays vs the MuJoCo run.

    conda run -n isaac python logic/simulation/walkmatch/actuator_id_isaac.py \
        --target left_knee_joint --amp 0.4 --out /tmp/actid_isaac_knee.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from humanoid.logic.simulation.walkmatch import spec  # noqa: E402  (pure numpy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="left_knee_joint",
                    help="PR joint name to step (default left_knee_joint)")
    ap.add_argument("--amp", type=float, default=0.4, help="step amplitude (rad)")
    ap.add_argument("--out", default="/tmp/actid_isaac.json")
    ap.add_argument("--configs",
                    default="implicit_arm0,implicit_arm1,explicit_arm0,explicit_arm1",
                    help="comma list of <implicit|explicit>_<arm0|arm1>")
    ap.add_argument("--gravity", action="store_true", help="keep gravity ON (default OFF)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np  # noqa: E402
    from isaacsim.core.api import World  # noqa: E402
    from humanoid.logic.simulation.isaacsim.oli import Oli, NUM_JOINTS  # noqa: E402

    world = World(stage_units_in_meters=1.0, physics_dt=spec.DT, rendering_dt=1.0 / 50.0)
    if not args.gravity:
        try:
            world.get_physics_context().set_gravity(0.0)
            print("[actid] gravity OFF", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[actid] WARN could not zero gravity: {e}", flush=True)

    # Pinned base, high spawn so the legs hang free (no ground contact even with gravity).
    oli = Oli(world, pin_root=True, spawn_pose=(0.0, 0.0, 1.1))
    dof = oli.dof_names
    if args.target not in spec.PR_ORDER:
        raise SystemExit(f"unknown target {args.target!r}")

    # PR -> Isaac permutation: isaac_vec = pr_vec[to_isaac]
    to_isaac = np.array([spec.PR_ORDER.index(n) for n in dof], dtype=np.int64)
    kp_is = spec.KP_LEG[to_isaac]  # drive only the 8 leg joints (ankle excluded — unstable)
    kd_is = spec.KD_LEG[to_isaac]
    def_is = spec.DEFAULT[to_isaac]
    arm_is = spec.ARMATURE[to_isaac]
    target_is = dof.index(args.target)
    target_pr = spec.PR_ORDER.index(args.target)
    q0 = float(spec.DEFAULT[target_pr])
    q_final = q0 + args.amp

    zeros = np.zeros(NUM_JOINTS, dtype=np.float32)

    def run_config(name: str) -> dict:
        mode, arm = name.split("_")
        explicit = mode == "explicit"
        # Clean IC every config: default pose, zero velocity.
        oli.set_joint_state(def_is, zeros)
        oli.set_armature(arm_is if arm == "arm1" else zeros)
        if explicit:
            oli.set_effort_mode()  # zero drive gains; cache effort clip
        t_list, q_list, dq_list, qd_list = [], [], [], []
        for tick in range(spec.N_STEPS):
            q_des_pr = spec.q_des_at(tick, target_pr, args.amp)
            q_des_is = q_des_pr[to_isaac]
            if explicit:
                oli.set_command_isaac(q_des_is, zeros, zeros, kp_is, kd_is)
                oli.apply_torque_isaac()
                world.step(render=False)
            else:
                oli.apply_isaac(q_des_is, zeros, zeros, kp_is, kd_is)
                world.step(render=False)
            q, dq, _ = oli.read_joints_isaac()
            t_list.append(tick * spec.DT)
            q_list.append(float(q[target_is]))
            dq_list.append(float(dq[target_is]))
            qd_list.append(float(q_des_is[target_is]))
        # Metrics from step onset.
        t0 = spec.N_HOLD
        m = spec.step_metrics(
            np.array(t_list[t0:]) - spec.T_STEP, np.array(q_list[t0:]), q0, q_final)
        print(f"[actid] {name:16s} onset={m.get('onset_lag_ms', float('nan')):6.1f}ms "
              f"rise={m.get('rise_time_ms', float('nan')):6.1f}ms "
              f"overshoot={m.get('overshoot_pct', float('nan')):5.1f}% "
              f"ss_err={m.get('ss_err_rad', float('nan')):+.4f}rad", flush=True)
        return {"t": t_list, "q": q_list, "dq": dq_list, "qdes": qd_list, "metrics": m}

    results = {}
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        results[name] = run_config(name)

    out = {
        "meta": {"sim": "isaac", "target": args.target, "amp": args.amp,
                 "q0": q0, "q_final": q_final, "dt": spec.DT,
                 "t_step": spec.T_STEP, "gravity": args.gravity},
        "configs": results,
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"[actid] wrote {args.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
