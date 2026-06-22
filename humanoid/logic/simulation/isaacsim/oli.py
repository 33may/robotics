"""
oli.py — `Oli`, the reusable Isaac Sim component for the HU_D04_01 humanoid.

`Oli` owns everything Oli-specific inside an Isaac world: USD loading, root
pinning, articulation init, an IMU sensor at `base_link`, the PR↔Isaac joint
permutation, and the PD-with-feedforward actuator law. Host apps construct one
`Oli` against their own `World` and call `oli.tick()` once per physics step.

Two modes:
  - `Oli(world, bridge=some_bridge)` — `tick()` reads state, sends it over the
    bridge, drains incoming cmds, applies the PD law. (MAY-147 deploy workflow.)
  - `Oli(world, bridge=None)` — `tick()` only applies whatever `apply_cmd(...)`
    last set. No IPC, no sidecar. (RL eval / ONNX / teleop / kinematics-only.)

The bridge is duck-typed: any object exposing `handshake(dof_names)`,
`send_state_imu(...)`, and `poll_cmd()` works. This keeps `Oli` decoupled from
the IPC implementation so a future in-process SDK adapter can drop in unchanged.

This module MUST run in the `isaac` conda env (CPython 3.11 + isaacsim 5.0).
It MUST NOT import `limxsdk` (ABI-incompatible) or anything from `bridge/`.

References:
  - design.md D5 (tick cadence), D7 (permutation), D8 (IMU), D14 (Oli API)
  - spec.md "Oli is a reusable Isaac component", "Oli.tick() applies PD law"
  - joint audit: openspec/changes/may-147-isaac-limx-sdk-bridge/_research/joint_name_audit.md
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Union

import numpy as np

# Accept python sequences OR numpy arrays for all array-valued args.
FloatArray = Union[Sequence[float], np.ndarray]

# ── PR canonical joint order (HU_D04_01) — MAY-145 § 11 ──────────────────────
# The single source of truth for the wire ordering. The permutation between this
# and Isaac's DOF order is computed at runtime in __init__ (never hard-coded), so
# a USD re-import that reshuffles Isaac's DOF order is handled automatically.
PR_ORDER: List[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
NUM_JOINTS: int = 31
assert len(PR_ORDER) == NUM_JOINTS

DEFAULT_USD = Path(
    "/home/may33/projects/ml_portfolio/robotics/humanoid/assets/oli/usd/HU_D04_01.usd"
)

# Variant → USD filename suffix
_VARIANT_SUFFIX = {"bare": "", "gripper": "_with_gripper", "hand": "_with_hand"}


# ── Duck-typed bridge interface (D14 reversibility) ─────────────────────────

class BridgeLike(Protocol):
    """Structural interface `Oli` needs from a bridge — IPC or in-process."""

    def handshake(self, dof_names: Sequence[str]) -> None: ...

    def send_state_imu(
        self, seq: int, stamp_ns: int,
        q: Sequence[float], dq: Sequence[float], tau: Sequence[float],
        acc: Sequence[float], gyro: Sequence[float], quat_wxyz: Sequence[float],
    ) -> None: ...

    def poll_cmd(self) -> Optional[Dict]: ...


# ── Cached cmd state ────────────────────────────────────────────────────────

class _CachedCmd:
    """Latched PR-space command. Zero-initialized (cold start = no torque)."""

    def __init__(self) -> None:
        z = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.q_d = z.copy()
        self.dq_d = z.copy()
        self.tau_ff = z.copy()
        self.kp = z.copy()
        self.kd = z.copy()
        self.mode = np.zeros(NUM_JOINTS, dtype=np.int32)
        self.parallel_solve_required = np.ones(NUM_JOINTS, dtype=np.int32)


# ── Oli ─────────────────────────────────────────────────────────────────────

class Oli:
    """Reusable HU_D04_01 component inside an Isaac `World`.

    Construct after the `World` exists but the caller still owns stepping and
    rendering. `Oli.tick()` performs one physics-tick's worth of state read +
    cmd apply but never calls `world.step()` or renders — that's the host's job.
    """

    def __init__(
        self,
        world,  # isaacsim.core.api.World
        *,
        prim_path: str = "/World/Oli",
        usd_path: Optional[Path] = None,
        spawn_pose: Sequence[float] = (0.0, 0.0, 1.05),
        pin_root: bool = True,
        variant: Literal["bare", "gripper", "hand"] = "bare",
        bridge: Optional[BridgeLike] = None,
        imu_offset_pitch: float = 0.0,
        imu_offset_roll: float = 0.0,
    ) -> None:
        # Imports are deferred to __init__ so the module is importable for
        # static analysis / unit imports without a live Kit runtime.
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

        self._world = world
        self._prim_path = prim_path
        self._bridge = bridge
        self._seq = 0

        if usd_path is None:
            suffix = _VARIANT_SUFFIX[variant]
            usd_path = (
                DEFAULT_USD if suffix == ""
                else DEFAULT_USD.with_name(f"HU_D04_01{suffix}.usd")
            )
        self._usd_path = Path(usd_path)
        if not self._usd_path.exists():
            raise FileNotFoundError(f"Oli USD not found: {self._usd_path}")

        # ── Reference the USD into the stage ────────────────────────────────
        add_reference_to_stage(usd_path=str(self._usd_path), prim_path=prim_path)
        stage = world.stage

        # Lift the base BEFORE reset — fixRootLink pins at the pose held at reset.
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        translate_op = (
            ops["xformOp:translate"] if "xformOp:translate" in ops
            else xform.AddTranslateOp()
        )
        translate_op.Set(Gf.Vec3d(*[float(v) for v in spawn_pose]))

        # ── Find the articulation root ──────────────────────────────────────
        root_prim = stage.GetPrimAtPath(prim_path)
        art_root = None
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = prim
                break
        if art_root is None:
            raise RuntimeError(f"No ArticulationRootAPI under {prim_path}")
        self._base_link_path = str(art_root.GetPath())  # base_link

        if pin_root:
            PhysxSchema.PhysxArticulationAPI.Apply(art_root)
            attr = art_root.GetAttribute("physxArticulation:fixRootLink")
            if not attr or not attr.IsValid():
                attr = art_root.CreateAttribute(
                    "physxArticulation:fixRootLink", Sdf.ValueTypeNames.Bool
                )
            attr.Set(True)

        # ── Materialize the articulation ────────────────────────────────────
        world.reset()
        self._art = SingleArticulation(prim_path=prim_path, name="oli")
        self._art.initialize()

        if self._art.num_dof != NUM_JOINTS:
            raise RuntimeError(
                f"expected {NUM_JOINTS} DOFs, got {self._art.num_dof} "
                f"— USD has extra joints not in PR canonical"
            )

        # ── PD realized via PhysX implicit drive (design.md D5, Option B) ───
        # We do NOT compute torque ourselves and call set_joint_efforts for the
        # PD term — that applies the cmd's Kd as an explicit one-step-lagged
        # force, which rings unstably for MuJoCo/real-robot-tuned gains. Instead
        # we push Kp/Kd into the joint drives and command position+velocity
        # targets, so PhysX integrates τ = Kp(q_d−q) + Kd(dq_d−dq) implicitly
        # (stable, identical formula, same as the real motor controller). The
        # feedforward τ_ff is added separately via set_joint_efforts (additive).
        #
        # Drive gains start at zero; they're written on the first cmd and only
        # re-written when Kp/Kd change (the walk/stand controllers send constant
        # gains, so this is rare). `_drive_gains_isaac` caches the last-written
        # gains (Isaac order) so we can detect changes cheaply.
        self._drive_gains_isaac = None  # (kps_isaac, kds_isaac) or None
        self._art._articulation_view.set_gains(
            kps=np.zeros((1, NUM_JOINTS), dtype=np.float32),
            kds=np.zeros((1, NUM_JOINTS), dtype=np.float32),
        )

        # ── Build permutation tables (D7) ───────────────────────────────────
        isaac_names = list(self._art.dof_names)
        pr_set, isaac_set = set(PR_ORDER), set(isaac_names)
        if pr_set != isaac_set:
            raise RuntimeError(
                f"joint-name set mismatch. In Isaac not PR: "
                f"{sorted(isaac_set - pr_set)}; in PR not Isaac: "
                f"{sorted(pr_set - isaac_set)}"
            )
        # pr_to_isaac[pr_idx] = isaac_idx of the same joint
        self._pr_to_isaac = np.array(
            [isaac_names.index(n) for n in PR_ORDER], dtype=np.int64
        )
        # isaac_to_pr[isaac_idx] = pr_idx of the same joint (inverse)
        self._isaac_to_pr = np.empty(NUM_JOINTS, dtype=np.int64)
        self._isaac_to_pr[self._pr_to_isaac] = np.arange(NUM_JOINTS)

        # ── Attach an IMU sensor at base_link (D8) ──────────────────────────
        self._imu = self._attach_imu(imu_offset_pitch, imu_offset_roll)

        # ── Cached cmd + latency stats ──────────────────────────────────────
        self._cmd = _CachedCmd()
        self._tick_latencies_us: List[float] = []

        # ── Bridge handshake (if attached) ──────────────────────────────────
        if self._bridge is not None:
            self._bridge.handshake(isaac_names)

    # ── IMU setup ───────────────────────────────────────────────────────────

    def _attach_imu(self, offset_pitch: float, offset_roll: float):
        from isaacsim.sensors.physics import IMUSensor
        from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

        imu_path = f"{self._base_link_path}/oli_imu"
        orientation = None
        if offset_pitch != 0.0 or offset_roll != 0.0:
            orientation = euler_angles_to_quats(
                np.array([offset_roll, offset_pitch, 0.0]), degrees=False
            )
        imu = IMUSensor(
            prim_path=imu_path,
            name="oli_imu",
            translation=np.zeros(3),
            orientation=orientation,
        )
        imu.initialize()
        return imu

    # ── Public properties ───────────────────────────────────────────────────

    @property
    def dof_names(self) -> List[str]:
        """Isaac DOF order (not PR order)."""
        return list(self._art.dof_names)

    @property
    def num_dof(self) -> int:
        return int(self._art.num_dof)

    @property
    def base_link_path(self) -> str:
        return self._base_link_path

    # ── Reading articulation + IMU (PR-ordered out) ─────────────────────────

    def _read_q_dq_tau_pr(self):
        q_isaac = np.asarray(self._art.get_joint_positions(), dtype=np.float32)
        dq_isaac = np.asarray(self._art.get_joint_velocities(), dtype=np.float32)
        tau_isaac = np.asarray(
            self._art.get_measured_joint_efforts(), dtype=np.float32
        )
        # Permute Isaac → PR
        return (
            q_isaac[self._pr_to_isaac],
            dq_isaac[self._pr_to_isaac],
            tau_isaac[self._pr_to_isaac],
        )

    def _read_imu(self):
        frame = self._imu.get_current_frame()
        acc = np.asarray(frame["lin_acc"], dtype=np.float32).reshape(-1)[:3]
        gyro = np.asarray(frame["ang_vel"], dtype=np.float32).reshape(-1)[:3]
        quat_wxyz = np.asarray(frame["orientation"], dtype=np.float32).reshape(-1)[:4]
        return acc, gyro, quat_wxyz

    def read_state(self) -> Dict:
        """RobotState-shaped dict in PR order: {stamp, q, dq, tau, motor_names}."""
        q, dq, tau = self._read_q_dq_tau_pr()
        return {
            "stamp": time.time_ns(),
            "q": q,
            "dq": dq,
            "tau": tau,
            "motor_names": list(PR_ORDER),
        }

    def read_imu(self) -> Dict:
        """ImuData-shaped dict: {stamp, acc, gyro, quat_wxyz}."""
        acc, gyro, quat_wxyz = self._read_imu()
        return {
            "stamp": time.time_ns(),
            "acc": acc,
            "gyro": gyro,
            "quat_wxyz": quat_wxyz,
        }

    # ── Command application ─────────────────────────────────────────────────

    def apply_cmd(
        self,
        q_d: FloatArray,
        dq_d: Optional[FloatArray] = None,
        tau_ff: Optional[FloatArray] = None,
        kp: Optional[FloatArray] = None,
        kd: Optional[FloatArray] = None,
    ) -> None:
        """Low-level escape hatch — set the cached cmd directly (PR order).

        Only the kwargs you pass are updated; the rest keep their previous
        values. This is what RL / ONNX / teleop call instead of a bridge.
        """
        def _set(name: str, val: Optional[FloatArray]) -> None:
            if val is None:
                return
            arr = np.asarray(val, dtype=np.float32)
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(
                    f"apply_cmd '{name}' must be shape ({NUM_JOINTS},), "
                    f"got {arr.shape}"
                )
            setattr(self._cmd, name, arr)

        _set("q_d", q_d)
        _set("dq_d", dq_d)
        _set("tau_ff", tau_ff)
        _set("kp", kp)
        _set("kd", kd)

    def _update_cmd_from_bridge_msg(self, msg: Dict) -> None:
        """Replace the cached cmd from a decoded bridge CMD message (PR order)."""
        c = self._cmd
        c.q_d = np.asarray(msg["q"], dtype=np.float32)
        c.dq_d = np.asarray(msg["dq"], dtype=np.float32)
        c.tau_ff = np.asarray(msg["tau"], dtype=np.float32)
        c.kp = np.asarray(msg["kp"], dtype=np.float32)
        c.kd = np.asarray(msg["kd"], dtype=np.float32)
        c.mode = np.asarray(msg["mode"], dtype=np.int32)
        c.parallel_solve_required = np.asarray(
            msg["parallel_solve_required"], dtype=np.int32
        )

    # ── The tick ────────────────────────────────────────────────────────────

    def _push_cmd_to_drive(self) -> None:
        """Realize the cached PR-space cmd via PhysX's implicit drive (D5/B).

        τ = Kp(q_d−q) + Kd(dq_d−dq) is integrated by PhysX from the drive gains
        + position/velocity targets (stable). τ_ff is added as explicit effort.
        Gains are only re-written when they change (cheap change-detection).
        """
        c = self._cmd
        view = self._art._articulation_view

        # Permute PR → Isaac
        kps_isaac = c.kp[self._isaac_to_pr]
        kds_isaac = c.kd[self._isaac_to_pr]
        q_d_isaac = c.q_d[self._isaac_to_pr]
        dq_d_isaac = c.dq_d[self._isaac_to_pr]
        tau_ff_isaac = c.tau_ff[self._isaac_to_pr]

        # Re-write gains only when they change (controllers send constant gains)
        if (
            self._drive_gains_isaac is None
            or not np.array_equal(self._drive_gains_isaac[0], kps_isaac)
            or not np.array_equal(self._drive_gains_isaac[1], kds_isaac)
        ):
            view.set_gains(
                kps=kps_isaac.reshape(1, -1),
                kds=kds_isaac.reshape(1, -1),
            )
            self._drive_gains_isaac = (kps_isaac.copy(), kds_isaac.copy())

        # Position + velocity targets drive the implicit PD
        view.set_joint_position_targets(q_d_isaac.reshape(1, -1))
        view.set_joint_velocity_targets(dq_d_isaac.reshape(1, -1))
        # Feedforward torque is additive on top of the implicit PD
        self._art.set_joint_efforts(tau_ff_isaac)

    def tick(self) -> None:
        """One physics-tick worth of work. Does NOT step or render the world.

        With a bridge: read state+IMU → send → drain cmd → push cmd to drive.
        Without a bridge: push the last `apply_cmd` cache to the drive.
        """
        t0 = time.perf_counter()

        if self._bridge is not None:
            q_pr, dq_pr, tau_pr = self._read_q_dq_tau_pr()
            acc, gyro, quat_wxyz = self._read_imu()
            self._bridge.send_state_imu(
                seq=self._seq,
                stamp_ns=time.time_ns(),
                q=q_pr.tolist(),
                dq=dq_pr.tolist(),
                tau=tau_pr.tolist(),
                acc=acc.tolist(),
                gyro=gyro.tolist(),
                quat_wxyz=quat_wxyz.tolist(),
            )
            self._seq += 1
            # Drain all pending cmds; keep the most recent
            latest = None
            while True:
                msg = self._bridge.poll_cmd()
                if msg is None:
                    break
                latest = msg
            if latest is not None:
                self._update_cmd_from_bridge_msg(latest)

        # Realize the cached cmd via the implicit drive (mode is recorded, not applied)
        self._push_cmd_to_drive()

        dt_us = (time.perf_counter() - t0) * 1e6
        self._tick_latencies_us.append(dt_us)
        if len(self._tick_latencies_us) > 10000:
            self._tick_latencies_us = self._tick_latencies_us[-5000:]

    # ── Stats ───────────────────────────────────────────────────────────────

    def tick_latency_stats(self) -> Dict[str, float]:
        """p50/p99 tick latency in microseconds over the rolling window."""
        if not self._tick_latencies_us:
            return {"p50_us": 0.0, "p99_us": 0.0, "n": 0}
        arr = np.asarray(self._tick_latencies_us)
        return {
            "p50_us": float(np.percentile(arr, 50)),
            "p99_us": float(np.percentile(arr, 99)),
            "n": len(arr),
        }
