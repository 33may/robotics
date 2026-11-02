"""sim_world_main.py — the SIM World process: Isaac + Oli + SimComm server.

Runs in the `isaac` env. Boots Isaac, spawns Oli FREE-BASE over a ground plane, serves
the brain (SimComm is the server), and runs the authoritative loop: publish Observation
→ drain the latest PolicyOut → apply it → step physics. Freeze-until-first-cmd (D9)
removes the free-fall window during brain/ONNX bring-up; a stale-cmd watchdog damps to
a fail-safe. The brain (separate process, `brain` env) connects in and steers.

    # terminal 1 (World):
    conda run -n isaac python humanoid/logic/simulation/isaacsim/sim_world_main.py
    # terminal 2 (brain):
    conda run -n brain python humanoid/logic/oli/brain_main.py --walk-after 3 --vx 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.contracts import PR_ORDER  # noqa: E402  (pure: numpy only)

# Per-joint rotor inertia (armature) the walk policy was tuned against — the authoritative
# HU_D04_01 MJCF values (vendor/.../HU_D04_description/xml/HU_D04_01.xml). Isaac's USD ships
# armature=0 everywhere; injecting these mirrors IsaacLab's ImplicitActuatorCfg.armature.
# The serial ankle pitch/roll inherit the achilles A/B rotor inertia (0.1845504). Keyed by
# joint NAME (robust to the head/wrist order quirks); a drift guard pins it to the MJCF.
_ARM_HIP_KNEE = 0.14125         # hip pitch/roll/yaw, knee
_ARM_ANKLE_WAIST = 0.1845504    # ankle pitch/roll (≈ achilles A/B), waist yaw/roll/pitch
_ARM_SHOULDER_ELBOW = 0.0886706  # shoulder pitch/roll/yaw, elbow
_ARM_HEAD_WRIST = 0.0153218     # head yaw/pitch, wrist yaw/pitch/roll


def _armature_for(joint: str) -> float:
    if "ankle" in joint or "waist" in joint:
        return _ARM_ANKLE_WAIST
    if "hip" in joint or "knee" in joint:
        return _ARM_HIP_KNEE
    if "shoulder" in joint or "elbow" in joint:
        return _ARM_SHOULDER_ELBOW
    if "head" in joint or "wrist" in joint:
        return _ARM_HEAD_WRIST
    raise KeyError(f"no armature group for joint {joint!r}")


ARMATURE_PR = [_armature_for(n) for n in PR_ORDER]

# Sim init-condition (design D-option-A): the World spawns Oli into this PR-ordered
# crouch — which IS the walk policy's `default_angle` (== stand_pos). The policy's
# observation is (q − default_angle), so spawning here hands it a zero-pose start;
# spawning straight-legged (zeros) drops a non-stance pose it can't recover from.
# A drift guard (tests/oli/world/test_spawn_pose.py) pins this to walk_param.yaml.
HOME_POSE_PR = [
    -0.15, -0.00, -0.05, 0.30, -0.16, 0.0,   # left  leg  (hip p/r/y, knee, ank p/r)
    -0.15,  0.00,  0.05, 0.30, -0.16, 0.0,   # right leg
     0.0,   0.0,   0.0,                        # waist yaw/roll/pitch
     0.0,   0.0,                               # head  yaw/pitch
     0.1,   0.1,  -0.2, -0.2, 0.0, 0.0, 0.0,  # left  arm (sh p/r/y, elbow, wr y/p/r)
     0.1,  -0.1,   0.2, -0.2, 0.0, 0.0, 0.0,  # right arm
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/oli-world.sock")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render-every", type=int, default=16,
                    help="render 1 in N physics ticks (keeps the viewport cheap)")
    ap.add_argument("--cameras", action="store_true",
                    help="attach Oli's baked D435i cameras and stream RGBD on a separate "
                         "frame channel (off by default so the walk path pays no render cost)")
    ap.add_argument("--camera-socket", default="/tmp/oli-world-frames.sock",
                    help="AF_UNIX SOCK_STREAM path for the camera frame channel (separate "
                         "from --socket)")
    ap.add_argument("--camera-res", type=int, nargs=2, default=[1280, 720],
                    metavar=("W", "H"),
                    help="camera resolution W H (D435i native 1280×720)")
    ap.add_argument("--watchdog-ms", type=float, default=500.0)
    ap.add_argument("--pace", choices=["lockstep", "free"], default="lockstep",
                    help="lockstep = step exactly --decimation ticks per brain cmd "
                         "(zero sensor→actuator latency); free = step continuously")
    ap.add_argument("--decimation", type=int, default=10,
                    help="physics ticks per policy command (lockstep); must match the "
                         "policy's trained decimation (walk = 10)")
    ap.add_argument("--spawn-height", type=float, default=1.1,
                    help="base z at spawn; 1.1 is the tuned height where Oli stands "
                         "still under the walk policy (lower collapses at walk handoff)")
    ap.add_argument("--settle-steps", type=int, default=200,
                    help="pre-serve steps holding the crouch so feet plant + IMU populates")
    ap.add_argument("--settle-kp", type=float, default=150.0)
    ap.add_argument("--settle-kd", type=float, default=5.0)
    ap.add_argument("--control", choices=["explicit", "implicit"], default="implicit",
                    help="implicit = PhysX drive (smoother, default); explicit = legged_gym "
                         "per-substep torque PD — faithful to TRON1 but jitters with our "
                         "armature=0 joints (underdamped)")
    ap.add_argument("--armature", choices=["on", "off"], default="off",
                    help="inject MJCF rotor inertia; off (default) = USD/training default 0 "
                         "(TRON1 trains armature=0)")
    ap.add_argument("--ankle-effort", type=float, default=42.0,
                    help="cap serial ankle max effort to the TRAINING URDF value (42 N·m); "
                         "0 = leave USD default (deploy ~80 over-torques → whip)")
    ap.add_argument("--ankle-vel", type=float, default=13.6,
                    help="cap serial ankle max velocity to the training URDF value "
                         "(13.6 rad/s); 0 = leave USD default")
    ap.add_argument("--ankle-kp-scale", type=float, default=1.0,
                    help="multiply the ankle pitch/roll kp+kd by this factor. The deploy "
                         "ankle gains (kp 93.65) are PER-MOTOR (achilles A/B), so the "
                         "joint-space ankle stiffness the policy trained against is ~g^2 "
                         "higher; our serial ankle applies them directly → too soft, sags "
                         "into plantarflexion under load → body tips forward. ~8 ≈ g^2.")
    ap.add_argument("--ankle-roll-scale", type=float, default=1.0,
                    help="scale ankle ROLL kp+kd separately (lateral axis, achilles A−B "
                         "differential mode — different mechanical advantage than pitch)")
    ap.add_argument("--waist-kp-scale", type=float, default=1.0,
                    help="scale waist roll+pitch kp+kd (also achilles A/B driven → too soft "
                         "at the joint; torso flop drives lateral drift)")
    ap.add_argument("--ankle-effort-from-scale", action="store_true",
                    help="also scale the ankle effort cap with --ankle-kp-scale (keeps the "
                         "stiffer drive from clipping); off = use --ankle-effort verbatim")
    ap.add_argument("--ankle-parallel", action="store_true",
                    help="FAITHFUL dual-motor achilles emulation on the serial ankle (the "
                         "'free A/B + software kinematic constraint' path). Forces explicit "
                         "control; drives each ankle with motor-space PD (per-motor ±42 clip) "
                         "mapped through the linkage Jacobian Jᵀ, + reflected rotor inertia as "
                         "ankle armature. Recovers ×2.3 pitch / ×2.0 roll joint stiffness from "
                         "the raw per-motor gains — so leave --ankle-kp-scale at 1.0.")
    ap.add_argument("--solver-vel-iters", type=int, default=0,
                    help="PhysX velocity solver iters (0 = keep USD value; IsaacLab=4)")
    ap.add_argument("--solver-pos-iters", type=int, default=0,
                    help="PhysX position solver iters (0 = keep USD value, already 32; "
                         "do NOT lower to 4 — softer contacts destabilize walking)")
    ap.add_argument("--ground-friction", type=float, default=1.0,
                    help="ground plane static=dynamic friction (IsaacLab=1.0; "
                         "Isaac stock default=0.5)")
    ap.add_argument("--ground-restitution", type=float, default=0.0,
                    help="ground plane restitution (IsaacLab=0.0; Isaac stock default=0.8 "
                         "— a bouncy floor that wrecks footstrikes)")
    ap.add_argument("--pin-root", action="store_true",
                    help="pin the base (debug: isolated joint control, no walking)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wall seconds to run the loop (0 = until the window closes)")
    args = ap.parse_args()

    # SimulationApp MUST be created before any other isaacsim import.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})

    import numpy as np  # noqa: E402
    from isaacsim.core.api import World  # noqa: E402

    from humanoid.logic.simulation.isaacsim.oli import (  # noqa: E402
        NUM_JOINTS, Oli, _quat_rotate_inverse,
    )
    from humanoid.logic.simulation.isaacsim import ankle_parallel as ap_mod  # noqa: E402
    from humanoid.logic.simulation.isaacsim.sim_comm import SimComm, SimCommError  # noqa: E402
    from humanoid.logic.oli.comm.camera_publisher import CameraPublisher  # noqa: E402

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 1000.0, rendering_dt=1.0 / 50.0)
    # Contact model matters: Isaac's stock ground is friction 0.5 + restitution 0.8 (a
    # trampoline). The walk policy was trained on IsaacLab's 1.0 / 0.0 — match it so the
    # feet grip and don't bounce. Overridable for A/B (--ground-friction/-restitution).
    world.scene.add_default_ground_plane(
        static_friction=args.ground_friction,
        dynamic_friction=args.ground_friction,
        restitution=args.ground_restitution,
    )
    print(f"[world] ground: friction={args.ground_friction} "
          f"restitution={args.ground_restitution}", flush=True)
    oli = Oli(world, pin_root=args.pin_root, spawn_pose=(0.0, 0.0, args.spawn_height),
              cameras=args.cameras, camera_resolution=tuple(args.camera_res))

    simcomm = SimComm(oli, socket_path=args.socket)

    # Camera frame channel (opt-in): a SEPARATE SOCK_STREAM server that ships RGBD on the
    # render sub-tick, never blocking the control loop (latest-wins per camera). Off by
    # default so the walk path pays no render cost.
    pub = None
    if args.cameras:
        pub = CameraPublisher(oli, socket_path=args.camera_socket, every=1)
        print(f"[world] cameras ON ({args.camera_res[0]}×{args.camera_res[1]}) → "
              f"frame channel {args.camera_socket} (streams: {oli.camera_names})", flush=True)

    # ── Faithful dual-motor achilles ankle (--ankle-parallel) ────────────────────────
    # Emulate the real two-motor achilles on the serial ankle in software (no rods → no PhysX
    # loop blowup): each ankle driven by motor-space PD (per-motor ±42 clip) through the linkage
    # Jacobian, + reflected rotor inertia as ankle armature. Forces explicit control (the law is
    # per-substep torque). Recovers ×2.3/×2.0 joint stiffness from the raw per-motor gains.
    _parallel_arm_pr = None
    if args.ankle_parallel:
        args.control = "explicit"
        dn = oli.dof_names
        ankles = [
            (dn.index("left_ankle_pitch_joint"), dn.index("left_ankle_roll_joint"), ap_mod.J_LEFT),
            (dn.index("right_ankle_pitch_joint"), dn.index("right_ankle_roll_joint"), ap_mod.J_RIGHT),
        ]
        oli.configure_parallel_ankle(ankles)
        Iq = ap_mod.reflected_inertia(ap_mod.J_LEFT)  # [pitch, roll] ≈ [0.424, 0.376]
        _parallel_arm_pr = list(ARMATURE_PR)
        for i in (4, 10):   # PR ankle PITCH indices
            _parallel_arm_pr[i] = float(Iq[0])
        for i in (5, 11):   # PR ankle ROLL indices
            _parallel_arm_pr[i] = float(Iq[1])
        if args.ankle_kp_scale != 1.0 or args.ankle_roll_scale != 1.0:
            print("[world] NOTE: --ankle-parallel recovers stiffness via the Jacobian; "
                  "forcing --ankle-kp-scale/--ankle-roll-scale to 1.0", flush=True)
            args.ankle_kp_scale = args.ankle_roll_scale = 1.0
        if args.ankle_effort and args.ankle_effort < 100.0:
            args.ankle_effort = 100.0  # PhysX headroom for the faithful ~90 N·m ankle torque
        print(f"[world] ankle-parallel ON: dual-motor achilles emulation (reflected inertia "
              f"pitch={Iq[0]:.3f} roll={Iq[1]:.3f}); control forced explicit", flush=True)

    # ── Optional foot-contact trace (env-gated; OLI_FOOT_TRACE=/path.jsonl) ──────────
    # Logs the 6 contact-sphere world positions + base each policy command, to tell
    # slip (stance foot x drifts during stance) from launch (both feet z>0 = a hop)
    # from roll (heel/tip z exchange) during the dynamic step. Inert unless the env is set.
    _foot_view = None
    _foot_trace = None
    _foot_left = [800]
    if os.environ.get("OLI_FOOT_TRACE"):
        from isaacsim.core.prims import RigidPrim  # noqa: E402
        _foot_paths = [
            "/World/Oli/contact_foot_heel_L", "/World/Oli/contact_foot_center_L",
            "/World/Oli/contact_foot_tip_L", "/World/Oli/contact_foot_heel_R",
            "/World/Oli/contact_foot_center_R", "/World/Oli/contact_foot_tip_R"]
        _foot_view = RigidPrim(_foot_paths)
        _foot_view.initialize()
        _foot_trace = open(os.environ["OLI_FOOT_TRACE"], "w")
        print(f"[world] foot trace → {os.environ['OLI_FOOT_TRACE']}", flush=True)

    # Physics-fidelity injection (World-side): the walk policy was tuned in IsaacLab
    # against real rotor inertia, but our USD ships armature=0 — a stiff drive on a
    # massless rotor buzzes into divergence regardless of loop timing. Inject the MJCF
    # armature (and optionally bump the PhysX solver iters) BEFORE the crouch settle so
    # every subsequent step sees the corrected dynamics. One variable at a time: armature
    # is on by default; solver-iter overrides are opt-in.
    # Armature: MJCF rotor inertia only when --armature on. (Tested reflecting the achilles
    # rotor inertia ×g² onto the stiffened ankle for "correct" ζ — it DESTABILIZED the
    # implicit drive and fell at spawn. The implicit PhysX drive holds the ×8 ankle kp fine
    # at armature 0, so leave armature off by default.)
    if _parallel_arm_pr is not None:
        arm_isaac = simcomm.pr_to_isaac_vector(np.asarray(_parallel_arm_pr, dtype=np.float32))
        oli.set_armature(arm_isaac)
        print("[world] armature: MJCF legs + reflected achilles inertia on the serial ankle "
              "(--ankle-parallel)", flush=True)
    elif args.armature == "on":
        arm_isaac = simcomm.pr_to_isaac_vector(np.asarray(ARMATURE_PR, dtype=np.float32))
        oli.set_armature(arm_isaac)
        print(f"[world] armature injected (hip/knee {_ARM_HIP_KNEE}, ankle/waist "
              f"{_ARM_ANKLE_WAIST}, arm {_ARM_SHOULDER_ELBOW}, head/wrist "
              f"{_ARM_HEAD_WRIST})", flush=True)
    else:
        print("[world] armature OFF — USD default (0 on all joints)", flush=True)
    if args.solver_vel_iters or args.solver_pos_iters:
        oli.set_solver_iteration_counts(
            position=args.solver_pos_iters or None,
            velocity=args.solver_vel_iters or None)
        print(f"[world] solver iters → pos={args.solver_pos_iters or 'keep'} "
              f"vel={args.solver_vel_iters or 'keep'}", flush=True)
    if args.ankle_effort or args.ankle_vel:
        oli.set_actuation_limits(
            ["left_ankle_pitch_joint", "left_ankle_roll_joint",
             "right_ankle_pitch_joint", "right_ankle_roll_joint"],
            effort=args.ankle_effort or None, velocity=args.ankle_vel or None)
        print(f"[world] ankle limits → effort={args.ankle_effort or 'keep'} "
              f"velocity={args.ankle_vel or 'keep'} (training URDF values)", flush=True)
    if args.ankle_kp_scale != 1.0:
        print(f"[world] ankle kp+kd scaled ×{args.ankle_kp_scale} (achilles linkage "
              f"joint-space stiffness recovery)", flush=True)

    # Spawn into the policy's home crouch (PR→Isaac via SimComm), then settle so the
    # feet find the floor and the IMU populates before the brain takes over (D-opt-A).
    home_isaac = simcomm.pr_to_isaac_vector(np.asarray(HOME_POSE_PR, dtype=np.float32))
    oli.set_joint_state(home_isaac)
    if args.settle_steps > 0:
        z = np.zeros(NUM_JOINTS, dtype=np.float32)
        kp = np.full(NUM_JOINTS, args.settle_kp, dtype=np.float32)
        kd = np.full(NUM_JOINTS, args.settle_kd, dtype=np.float32)
        for _ in range(args.settle_steps):
            oli.apply_isaac(home_isaac, z, z, kp, kd)
            world.step(render=False)
        oli.set_joint_state(home_isaac)  # re-assert pose, zero residual velocity
        ps = oli.base_world_position()
        print(f"[world] settled crouch: base=({ps[0]:+.3f},{ps[1]:+.3f},{ps[2]:+.3f})",
              flush=True)

    # Control law: explicit reproduces legged_gym/TRON1 — drive gains zeroed (EFFORT mode)
    # and τ = kp(q_des−q)+kd(dq_des−dq)+tau_ff recomputed every physics substep from the
    # latest q/dq, pushed as pure joint effort. The settle above used the implicit drive;
    # zero it now so PhysX adds no PD of its own under explicit control.
    if args.control == "explicit":
        eff = oli.set_effort_mode()
        lim = ("no clip!" if eff is None
               else f"τ clip {float(np.min(eff)):.0f}..{float(np.max(eff)):.0f} N·m")
        print(f"[world] control: explicit per-substep torque PD (effort mode); {lim}",
              flush=True)
    else:
        print("[world] control: implicit PhysX drive", flush=True)

    # Warm the camera render products: Isaac's annotators need a few render ticks to
    # populate before get_rgba/depth return valid buffers. The publisher tolerates a
    # not-ready camera, but warming here makes RGBD flow from the first loop tick.
    if pub is not None:
        for _ in range(8):
            world.step(render=True)
        print("[world] cameras warmed (8 render ticks)", flush=True)

    print(f"[world] serving on {args.socket}; waiting for brain to connect...", flush=True)
    simcomm.serve()
    print("[world] brain connected. freeze-until-first-cmd.", flush=True)

    watchdog_s = args.watchdog_ms / 1000.0
    tick = 0
    n_cmds = 0  # distinct policy commands applied → ticks/n_cmds = effective decimation
    zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
    loop_start = time.monotonic()

    _grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def _log_state() -> None:
        p = oli.base_world_position()
        _, gyro, quat = oli.read_imu()
        pg = _quat_rotate_inverse(quat, _grav)  # projected gravity the policy sees
        dec = tick / max(1, n_cmds)
        print(f"[world] t={tick / 1000.0:5.2f}s base=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
              f"pg=({pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}) "
              f"gyro=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f}) "
              f"cmds={n_cmds} t/c={dec:.1f}", flush=True)

    def _write_foot_trace() -> None:
        """Per-command foot-contact sample (slip/launch/roll diagnosis). Env-gated."""
        if _foot_trace is None or _foot_left[0] <= 0:
            return
        _foot_left[0] -= 1
        pos, _ = _foot_view.get_world_poses()
        pos = np.asarray(pos).reshape(-1, 3)  # (6,3): heel/center/tip L then R
        b = oli.base_world_position()
        _foot_trace.write(json.dumps({
            "t": tick,
            "base": [round(float(x), 4) for x in b[:3]],
            "Lx": [round(float(pos[i, 0]), 4) for i in (0, 1, 2)],
            "Lz": [round(float(pos[i, 2]), 4) for i in (0, 1, 2)],
            "Rx": [round(float(pos[i, 0]), 4) for i in (3, 4, 5)],
            "Rz": [round(float(pos[i, 2]), 4) for i in (3, 4, 5)],
        }) + "\n")
        _foot_trace.flush()

    def _timed_out() -> bool:
        return bool(args.duration) and (time.monotonic() - loop_start) > args.duration

    explicit = args.control == "explicit"
    damp_kd = np.full(NUM_JOINTS, 5.0, dtype=np.float32)

    # PR indices of the achilles-driven joints whose deploy gains are PER-MOTOR (A/B) and so
    # ~g^2 too soft at the JOINT when applied directly to our serial model. Each group scales
    # independently (different mechanical advantage). The leg joints are direct-drive → their
    # gains are already joint-space and must NOT be scaled (they track MuJoCo exactly).
    _PARALLEL_GROUPS = [
        ("ankle_pitch", (4, 10), "ankle_kp_scale"),   # A+B additive (fore-aft)
        ("ankle_roll", (5, 11), "ankle_roll_scale"),  # A−B differential (lateral)
        ("waist_roll", (13,), "waist_kp_scale"),       # waist A/B (lateral torso)
        ("waist_pitch", (14,), "waist_kp_scale"),      # waist A/B (fore-aft torso)
    ]

    def _scale_ankle_gains(cmd):
        """Amplify the achilles-driven joints' kp+kd to recover the JOINT-space stiffness the
        policy trained against (deploy gains are per-motor A/B). PolicyOut is frozen → write
        via object.__setattr__ on the World's own copy."""
        scales = {g[2]: getattr(args, g[2]) for g in _PARALLEL_GROUPS}
        if all(s == 1.0 for s in scales.values()):
            return cmd
        kp = np.asarray(cmd.kp, dtype=np.float32).copy()
        kd = np.asarray(cmd.kd, dtype=np.float32).copy()
        for _grp, idxs, attr in _PARALLEL_GROUPS:
            s = getattr(args, attr)  # noqa: F841 (kept for readability of the scale source)
            for i in idxs:
                kp[i] *= s
                kd[i] *= s
        object.__setattr__(cmd, "kp", kp)
        object.__setattr__(cmd, "kd", kd)
        return cmd

    def _apply_command(cmd) -> None:
        """Hand a fresh PolicyOut to the actuator. Explicit stores the PD target (torque is
        recomputed per substep); implicit pushes the PhysX drive once (it holds it)."""
        cmd = _scale_ankle_gains(cmd)
        if explicit:
            simcomm.set_command(cmd)
        else:
            simcomm.apply(cmd)

    def _actuate_and_step(render: bool) -> None:
        """One physics substep. Under explicit control, recompute τ from the LATEST q/dq
        and push it as effort BEFORE stepping (legged_gym _compute_torques per substep).
        On render steps (only then do camera pixels refresh) ship RGBD on the frame
        channel — the ~render-rate sub-tick decoupled from the 1 kHz control step (§4.3)."""
        if explicit:
            oli.apply_torque_isaac()
        world.step(render=render)
        if render and pub is not None:
            pub.publish(tick, tick * 1_000_000)

    def _set_damp_command() -> None:
        """Fail-safe (brain stale): hold the current pose with light damping only."""
        q_now, _, _ = oli.read_joints_isaac()
        if explicit:
            oli.set_command_isaac(q_now, zeros, zeros, zeros, damp_kd)
        else:
            oli.apply_isaac(q_now, zeros, zeros, zeros, damp_kd)

    try:
        if args.pace == "lockstep":
            # Pace World ← brain (zero latency): publish the frozen obs, then ping-pong —
            # wait for the brain's command for THAT obs, apply it, step exactly
            # `decimation` substeps (explicit: τ recomputed each substep; implicit: drive
            # holds the target), publish the next. Stamp = tick·1 ms as an exact integer so
            # the brain's stamp gate never underflows. Matches the trained decimation.
            simcomm.publish(tick * 1_000_000)
            while app.is_running() and not _timed_out():
                cmd = simcomm.receive_blocking(timeout=watchdog_s)
                if cmd is None:  # brain alive but silent → damp one tick, resync
                    _set_damp_command()
                    _actuate_and_step(render=False)
                    tick += 1
                    simcomm.publish(tick * 1_000_000)
                    print("[world] watchdog: brain stale → damping", flush=True)
                    continue
                _apply_command(cmd)
                n_cmds += 1
                for _ in range(args.decimation):
                    _actuate_and_step(render=(tick % args.render_every == 0))
                    tick += 1
                simcomm.publish(tick * 1_000_000)
                _write_foot_trace()
                if n_cmds % 25 == 0:
                    _log_state()
        else:
            # Free-run (legacy): step continuously, latest-wins held command. Races ahead
            # of the brain headless → sensor→actuator latency → unstable (kept for A/B).
            held = None
            last_cmd_wall = None
            while app.is_running() and not _timed_out():
                simcomm.publish(int(world.current_time * 1e9))
                cmd = simcomm.receive_latest()
                if cmd is not None:
                    held = cmd
                    last_cmd_wall = time.monotonic()
                    n_cmds += 1
                    _apply_command(cmd)
                if held is None:
                    app.update()  # UI alive, but do NOT step (freeze-until-first-cmd)
                    continue
                if (last_cmd_wall is not None
                        and (time.monotonic() - last_cmd_wall) > watchdog_s):
                    _set_damp_command()
                _actuate_and_step(render=(tick % args.render_every == 0))
                if tick % 500 == 0:
                    _log_state()
                tick += 1
        if _timed_out():
            print("[world] duration reached; exiting.", flush=True)
    except KeyboardInterrupt:
        print("\n[world] stopping.", flush=True)
    except SimCommError as e:
        print(f"[world] brain disconnected: {e}; stopping.", flush=True)
    finally:
        if pub is not None:
            pub.close()
        simcomm.close()
        app.close()


if __name__ == "__main__":
    main()
