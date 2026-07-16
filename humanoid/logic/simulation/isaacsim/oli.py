"""
oli.py — `Oli`, the HU_D04_01 articulation body inside an Isaac world.

`Oli` is the SIM World's body: USD loading, root pinning, articulation init, an IMU
sensor at `base_link`, and the PD-with-feedforward actuator law realized via PhysX's
implicit drive. It speaks ISAAC DOF ORDER only — the PR↔Isaac permutation and all
contract/policy knowledge live in `SimComm` (D4). The brain never sees this object.

Body interface that `SimComm` depends on (duck-typed):
    oli.dof_names                                  -> list[str]   (Isaac order)
    oli.read_joints_isaac()                        -> (q, dq, tau)   each (num_dof,)
    oli.read_imu()                                 -> (acc, gyro, quat_wxyz)
    oli.apply_isaac(q_des, dq_des, tau_ff, kp, kd)                  each (num_dof,)

This module MUST run in the `isaac` conda env (CPython 3.11 + isaacsim 5.0). It must
NOT import `limxsdk`. PD realization (memory `isaac-pd-implicit-drive`): push Kp/Kd
into the joint drives + position/velocity targets so PhysX integrates
τ = Kp(q_d−q) + Kd(dq_d−dq) implicitly (stable for real-tuned gains); τ_ff is added
additively via `set_joint_efforts`. NOT explicit-effort PD (that rings unstably).

References: design.md D4 (Comm owns permutation), D8 (IMU), D9 (freeze: the host owns
`world.step()`; Oli never steps), Oli body API.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

FloatArray = Union[Sequence[float], np.ndarray]

NUM_JOINTS: int = 31

# Shipped USD — the CORRECT Isaac model: it stands rock-solid under the walk policy.
# (Tested the URDF-imported training model HU_D04_01_rl.usd via build_rl_usd.py — it could
# not even stand: Isaac's stock URDF importer mangles the canted hip-pitch axis it warns
# about reorienting, which LimX's USD build handles correctly. So the shipped USD is the
# more faithful model; the forward-walk whip is a separate issue on it. See
# isaac_walk_physics_fidelity memory.)
DEFAULT_USD = (
    Path(__file__).resolve().parents[3] / "assets" / "oli" / "usd" / "HU_D04_01.usd"
)
_VARIANT_SUFFIX = {"bare": "", "gripper": "_with_gripper", "hand": "_with_hand"}


def pd_torque(q_des, q, dq_des, dq, kp, kd, tau_ff) -> np.ndarray:
    """Explicit PD torque, legged_gym `_compute_torques` form (control_type "P"):

        τ = kp·(q_des − q) + kd·(dq_des − dq) + tau_ff

    Pure/array-broadcast → unit-testable without isaacsim. The World calls this every
    physics substep with the LATEST q/dq (effort mode), exactly as TRON1 training does,
    instead of letting PhysX's implicit drive integrate the PD internally.
    """
    return (
        np.asarray(kp, dtype=np.float32) * (np.asarray(q_des, dtype=np.float32) - np.asarray(q, dtype=np.float32))
        + np.asarray(kd, dtype=np.float32) * (np.asarray(dq_des, dtype=np.float32) - np.asarray(dq, dtype=np.float32))
        + np.asarray(tau_ff, dtype=np.float32)
    ).astype(np.float32)


def _quat_rotate_inverse(q_wxyz, v) -> np.ndarray:
    """Rotate a WORLD-frame vector into the body frame (legged_gym `quat_rotate_inverse`),
    quaternion in wxyz. Used to express base angular velocity / gravity in the body frame,
    matching how MuJoCo's gyro/framequat and the deploy obs are defined.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, vec = q[0], q[1:4]
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(vec, v) * (2.0 * w)
    c = vec * (np.dot(vec, v) * 2.0)
    return (a - b + c).astype(np.float32)


class Oli:
    """HU_D04_01 articulation body inside an Isaac `World` (Isaac DOF order).

    Construct after the `World` exists. `Oli` reads/applies a single physics tick's
    worth of state but never calls `world.step()` or renders — the host (the World
    main loop) owns stepping, so it can freeze-until-cmd and pace rendering (D9).
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
        imu_offset_pitch: float = 0.0,
        imu_offset_roll: float = 0.0,
        cameras: bool = False,
        stereo_cameras: bool = False,
        camera_resolution: Tuple[int, int] = (1280, 720),
    ) -> None:
        # Deferred isaac imports so this module is importable for static analysis.
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

        self._world = world
        self._prim_path = prim_path

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
                f"expected {NUM_JOINTS} DOFs, got {self._art.num_dof}"
            )

        # ── PD via PhysX implicit drive (memory: isaac-pd-implicit-drive) ───
        # Gains start at zero; written on the first apply and only when they
        # change (controllers send constant gains per mode → rare re-writes).
        self._drive_gains = None  # (kps, kds) in Isaac order, or None
        self._cmd = None  # stored explicit-torque target (Isaac order), or None
        self._max_effort = None  # per-joint effort clip for explicit torque, or None
        self._parallel_ankles = None  # [(pitch_idx, roll_idx, J), ...] Isaac order, or None
        self._art._articulation_view.set_gains(
            kps=np.zeros((1, NUM_JOINTS), dtype=np.float32),
            kds=np.zeros((1, NUM_JOINTS), dtype=np.float32),
        )

        # ── IMU sensor at base_link (D8) — now OPTIONAL ─────────────────────
        # read_imu() derives orientation + body-frame gyro from the articulation ROOT state
        # (the Isaac IMUSensor doesn't update in our manual step loop). So the sensor prim
        # is unused; attach it best-effort and never fail the load on a base_link mismatch
        # (the URDF-imported model may name/merge links differently than the shipped USD).
        try:
            self._imu = self._attach_imu(imu_offset_pitch, imu_offset_roll)
        except Exception as e:  # pragma: no cover - defensive
            self._imu = None
            print(f"[oli] IMU sensor attach skipped ({e}); read_imu uses root state",
                  flush=True)

        # ── Cameras (oli-perception, MAY-149) — flag-gated (render cost, D7/D8) ──
        # The camera prims are baked into the USD sensor layer (build_camera_usd.py);
        # here we just wrap them with the render sensor. Off by default so the walk
        # path pays nothing. The host owns render(); get_rgba/depth read the last frame.
        self._cameras = {}  # name -> (Camera sensor, CameraMount)
        if cameras:
            try:
                self._attach_cameras(camera_resolution)
            except Exception as e:  # pragma: no cover - defensive
                self._cameras = {}
                print(f"[oli] camera attach skipped ({e})", flush=True)

        # ── Head stereo pair (MAY-173 locdev T1) — separate, opt-in rig ─────
        # RGB-only (no depth annotator): cuVGL/map-bake input, never part of the
        # RGBD table the CameraPublisher iterates with read_camera_rgbd.
        self._stereo_cameras = {}  # name -> (Camera sensor, CameraMount)
        if stereo_cameras:
            try:
                self._attach_stereo_cameras(camera_resolution)
            except Exception as e:  # pragma: no cover - defensive
                self._stereo_cameras = {}
                print(f"[oli] stereo camera attach skipped ({e})", flush=True)

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
            prim_path=imu_path, name="oli_imu",
            translation=np.zeros(3), orientation=orientation,
        )
        imu.initialize()
        return imu

    def _attach_cameras(self, resolution: Tuple[int, int]) -> None:
        """Wrap the baked D435i camera prims (build_camera_usd.py) with render sensors."""
        from isaacsim.sensors.camera import Camera

        from humanoid.logic.oli.camera_mounts import CAMERAS

        self._camera_resolution = (int(resolution[0]), int(resolution[1]))
        for mount in CAMERAS:
            path = f"{self._prim_path}/{mount.parent_link}/{mount.name}_camera"
            cam = Camera(prim_path=path, resolution=self._camera_resolution)
            cam.initialize()
            cam.add_distance_to_image_plane_to_frame()  # planar Z depth, meters
            self._cameras[mount.name] = (cam, mount)

    def _attach_stereo_cameras(self, resolution: Tuple[int, int]) -> None:
        """Wrap the baked head stereo prims (build_camera_usd.py) with RGB-only
        render sensors — no depth annotator, the pair exists for cuVGL/map-bake."""
        from isaacsim.sensors.camera import Camera

        from humanoid.logic.oli.camera_mounts import STEREO_CAMERAS

        self._camera_resolution = (int(resolution[0]), int(resolution[1]))
        for mount in STEREO_CAMERAS:
            path = f"{self._prim_path}/{mount.parent_link}/{mount.name}_camera"
            cam = Camera(prim_path=path, resolution=self._camera_resolution)
            cam.initialize()
            self._stereo_cameras[mount.name] = (cam, mount)

    @property
    def camera_names(self) -> List[str]:
        return list(self._cameras.keys())

    @property
    def stereo_camera_names(self) -> List[str]:
        return list(self._stereo_cameras.keys())

    def read_camera_rgb(self, name: str) -> np.ndarray:
        """(H,W,3) uint8 RGB for a stereo camera. The host must have rendered
        (`world.step(render=True)`) this tick for the frame to be current."""
        cam, _mount = self._stereo_cameras[name]
        rgba = cam.get_rgba()
        return np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)

    def read_camera_rgbd(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """(rgb (H,W,3) uint8, depth (H,W) float32 m) for a camera. The host must have
        rendered (`world.step(render=True)`) this tick for the frame to be current."""
        cam, _mount = self._cameras[name]
        rgba = cam.get_rgba()
        rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
        depth = np.asarray(
            cam.get_current_frame()["distance_to_image_plane"], dtype=np.float32
        )
        return rgb, depth

    def camera_intrinsics(self, name: str):
        """D435i-like intrinsics at this camera's render resolution (CameraIntrinsics).
        Covers both tables — RGBD cameras and the RGB-only stereo pair (MAY-173)."""
        from humanoid.logic.oli.camera_mounts import rgb_intrinsics

        table = self._cameras if name in self._cameras else self._stereo_cameras
        _cam, mount = table[name]
        w, h = self._camera_resolution
        return rgb_intrinsics(width=w, height=h, hfov_deg=mount.hfov_deg)

    # ── Public properties ───────────────────────────────────────────────────

    @property
    def dof_names(self) -> List[str]:
        """Isaac DOF order (SimComm builds the PR↔Isaac permutation from this)."""
        return list(self._art.dof_names)

    @property
    def num_dof(self) -> int:
        return int(self._art.num_dof)

    @property
    def base_link_path(self) -> str:
        return self._base_link_path

    # ── Body interface (Isaac DOF order; SimComm does PR↔Isaac) ──────────────

    def read_joints_isaac(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, dq, tau) in Isaac DOF order."""
        q = np.asarray(self._art.get_joint_positions(), dtype=np.float32)
        dq = np.asarray(self._art.get_joint_velocities(), dtype=np.float32)
        tau = np.asarray(self._art.get_measured_joint_efforts(), dtype=np.float32)
        return q, dq, tau

    def read_imu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(acc, gyro, quat_wxyz) — base frame.

        Derived from the articulation ROOT STATE, NOT Isaac's `IMUSensor`: that sensor's
        orientation + angular velocity DO NOT update inside our manual `world.step()` loop
        — it returns identity quat + zero gyro no matter the real motion (verified by
        `imu_probe.py`: the base tilts to 54° while the sensor still reads upright). That
        left the walk policy blind to its own tilt → it walked open-loop and toppled, immune
        to every physics/actuator change. `get_world_pose` is probe-confirmed correct and
        matches MuJoCo's `framequat`; the body-frame gyro matches MuJoCo's `Body_Gyro`.
        `acc` is the body-frame gravity reaction (a sane accelerometer proxy; the walk obs
        ignores it).
        """
        _, quat_wxyz = self._art.get_world_pose()
        quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)[:4]
        ang_world = np.asarray(
            self._art.get_angular_velocity(), dtype=np.float32).reshape(-1)[:3]
        gyro = _quat_rotate_inverse(quat, ang_world)  # world → body frame
        acc = _quat_rotate_inverse(quat, np.array([0.0, 0.0, 9.81], dtype=np.float32))
        return acc, gyro, quat

    def base_world_position(self) -> np.ndarray:
        """Base-link world position (x, y, z) — for tracking translation / height."""
        pos, _ = self._art.get_world_pose()
        return np.asarray(pos, dtype=np.float32).reshape(-1)[:3]

    def base_world_quat_wxyz(self) -> np.ndarray:
        """Base-link world orientation as a wxyz quaternion (ground-truth tilt)."""
        _, quat = self._art.get_world_pose()
        return np.asarray(quat, dtype=np.float32).reshape(-1)[:4]

    def set_base_pose(self, position, quat_wxyz) -> None:
        """Set the base (root) world pose — position (x,y,z) + orientation (wxyz).

        For probes / initial conditions: drops the articulation at a known tilt so the IMU
        can be checked against ground truth. The articulation must be initialized.
        """
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        self._art.set_world_pose(position=pos, orientation=quat)

    def set_base_velocity(self, linear, angular) -> None:
        """Set the base spatial velocity — linear (3,) + angular (3,), world frame.
        For the gyro check: spin the base at a known rate and read the IMU back."""
        vel = np.concatenate([
            np.asarray(linear, dtype=np.float32).reshape(3),
            np.asarray(angular, dtype=np.float32).reshape(3),
        ])
        self._art.set_world_velocity(vel)

    def set_joint_state(self, q_isaac: FloatArray, dq_isaac: Optional[FloatArray] = None) -> None:
        """Set joint positions (and velocities, default zero) directly — Isaac order.

        Used to spawn Oli into an initial condition (the home crouch) before the
        World starts serving; writes the physics state so the next read/step sees it.
        The PR→Isaac permutation is the caller's job (SimComm) — this stays pure.
        """
        q = np.asarray(q_isaac, dtype=np.float32).reshape(-1)
        if q.shape != (NUM_JOINTS,):
            raise ValueError(f"set_joint_state expects {NUM_JOINTS} positions, got {q.shape}")
        dq = (
            np.zeros(NUM_JOINTS, dtype=np.float32)
            if dq_isaac is None
            else np.asarray(dq_isaac, dtype=np.float32).reshape(-1)
        )
        self._art.set_joint_positions(q)
        self._art.set_joint_velocities(dq)

    def set_actuation_limits(self, names, effort=None, velocity=None) -> None:
        """Cap max effort (N·m) and/or max velocity (rad/s) on the NAMED joints (Isaac
        index resolved from dof_names). Used to match the TRAINING URDF's ankle limits
        (effort 42, velocity 13.6) on the serial ankle: the deploy walk_param allows ~80,
        which over-torques the bare serial ankle into the whip the policy never saw in
        training (the real ankle is geared down by the achilles, so 80 is safe there).
        """
        dn = list(self.dof_names)
        idxs = [dn.index(n) for n in names if n in dn]
        if not idxs:
            return
        view = self._art._articulation_view
        j = np.asarray(idxs, dtype=np.int64)
        if effort is not None:
            view.set_max_efforts(
                np.full((1, len(idxs)), float(effort), dtype=np.float32), joint_indices=j)
        if velocity is not None:
            view.set_max_joint_velocities(
                np.full((1, len(idxs)), float(velocity), dtype=np.float32), joint_indices=j)

    def set_armature(self, armature_isaac: FloatArray) -> None:
        """Set per-joint rotor inertia (armature) on the drive — Isaac DOF order.

        Isaac's USD ships armature=0 on every joint; the walk policy was tuned against
        real rotor inertia (HU_D04 MJCF). A high-kp drive (legs kp≈139) on a massless
        rotor at 1 kHz buzzes into divergence — independent of loop timing, which is why
        lock-step didn't help. Mirrors IsaacLab's ImplicitActuatorCfg.armature; the
        PR→Isaac permutation is the caller's job (SimComm), so this stays pure.
        """
        a = np.asarray(armature_isaac, dtype=np.float32).reshape(-1)
        if a.shape != (NUM_JOINTS,):
            raise ValueError(f"set_armature expects {NUM_JOINTS} values, got {a.shape}")
        self._art._articulation_view.set_armatures(a.reshape(1, -1))

    def set_solver_iteration_counts(
        self, position: Optional[int] = None, velocity: Optional[int] = None
    ) -> None:
        """Override the articulation's PhysX solver iteration counts (TGS).

        IsaacLab trains HU_D04 with position=4 / velocity=4; our USD defaults to a low
        velocity-iteration count, which under-resolves the stiff leg drives. Each arg is
        applied only if given (None = leave the USD value untouched), so callers can move
        one knob at a time.
        """
        view = self._art._articulation_view
        if position is not None:
            view.set_solver_position_iteration_counts(
                np.full((1,), int(position), dtype=np.int32))
        if velocity is not None:
            view.set_solver_velocity_iteration_counts(
                np.full((1,), int(velocity), dtype=np.int32))

    # ── Explicit per-substep torque control (legged_gym / TRON1 reproduction) ────

    def set_effort_mode(self) -> Optional[np.ndarray]:
        """Zero the implicit drive gains so PhysX adds NO PD of its own — control becomes
        pure applied effort (legged_gym `default_dof_drive_mode = EFFORT`). Pair with
        `set_command_isaac` + `apply_torque_isaac`. Call once before explicit control.

        Also caches the per-joint effort limit (USD drive maxForce = URDF effort limit =
        legged_gym `torque_limits`) so explicit torque is clipped EVERY substep — without
        the clip the high-kp PD positive-feedbacks into divergence. Returns the cached
        limit (Isaac order) for logging, or None if unavailable.
        """
        view = self._art._articulation_view
        z = np.zeros((1, NUM_JOINTS), dtype=np.float32)
        view.set_gains(kps=z, kds=z)
        self._drive_gains = (np.zeros(NUM_JOINTS, dtype=np.float32),
                             np.zeros(NUM_JOINTS, dtype=np.float32))
        try:
            eff = np.asarray(view.get_max_efforts(), dtype=np.float32).reshape(-1)
            self._max_effort = eff if eff.shape == (NUM_JOINTS,) else None
        except Exception:  # pragma: no cover - defensive; clip is optional
            self._max_effort = None
        return self._max_effort

    def configure_parallel_ankle(self, ankles) -> None:
        """Enable the faithful dual-motor achilles law for the given ankle pairs (the "free A/B
        + software kinematic constraint" path). `ankles` = list of (pitch_idx, roll_idx, J) in
        ISAAC DOF order (SimComm resolves names→indices + picks J_LEFT/J_RIGHT). Under explicit
        control, `apply_torque_isaac` then OVERWRITES these ankle joints' joint-space PD with the
        motor-space parallel law (per-motor clip + Jᵀ). No effect under implicit control.
        """
        self._parallel_ankles = list(ankles) if ankles else None

    def set_command_isaac(self, q_des, dq_des, tau_ff, kp, kd) -> None:
        """Store a PD target (Isaac DOF order) for explicit per-substep torque control.

        The World recomputes torque every physics substep from this target + the latest
        q/dq via `apply_torque_isaac` — unlike `apply_isaac`, which hands the PD to the
        PhysX drive once. SimComm does the PR→Isaac permutation, so this stays pure.
        """
        self._cmd = (
            np.asarray(q_des, dtype=np.float32).reshape(-1),
            np.asarray(dq_des, dtype=np.float32).reshape(-1),
            np.asarray(tau_ff, dtype=np.float32).reshape(-1),
            np.asarray(kp, dtype=np.float32).reshape(-1),
            np.asarray(kd, dtype=np.float32).reshape(-1),
        )

    def apply_torque_isaac(self) -> None:
        """Apply ONE substep of explicit PD torque (legged_gym `_compute_torques`): recompute
        τ from the LATEST q/dq against the stored target and push it as pure joint effort.
        Requires `set_effort_mode` (drive gains 0) so PhysX doesn't double the PD.
        """
        if self._cmd is None:
            return
        q_des, dq_des, tau_ff, kp, kd = self._cmd
        q = np.asarray(self._art.get_joint_positions(), dtype=np.float32).reshape(-1)
        dq = np.asarray(self._art.get_joint_velocities(), dtype=np.float32).reshape(-1)
        tau = pd_torque(q_des, q, dq_des, dq, kp, kd, tau_ff)
        if self._max_effort is not None:  # legged_gym clips τ to torque_limits each substep
            np.clip(tau, -self._max_effort, self._max_effort, out=tau)
        if self._parallel_ankles is not None:
            # Faithful dual-motor achilles: replace the ankle joints' joint-space PD with the
            # motor-space law (per-motor ±42 clip BEFORE Jᵀ). Applied AFTER the joint-effort clip
            # so the ankle's authority is governed by the per-motor clip, not the joint cap.
            from humanoid.logic.simulation.isaacsim.ankle_parallel import apply_to_torque_vector
            tau = apply_to_torque_vector(tau, q, dq, q_des, dq_des, self._parallel_ankles)
        self._art.set_joint_efforts(tau)

    def apply_isaac(
        self,
        q_des: FloatArray,
        dq_des: FloatArray,
        tau_ff: FloatArray,
        kp: FloatArray,
        kd: FloatArray,
    ) -> None:
        """Apply a PD command (Isaac DOF order) via the implicit drive.

        τ = Kp(q_des−q) + Kd(dq_des−dq) integrated by PhysX from the drive gains +
        position/velocity targets (stable); τ_ff added as explicit effort. Gains are
        re-written only when they change.
        """
        view = self._art._articulation_view
        kps = np.asarray(kp, dtype=np.float32)
        kds = np.asarray(kd, dtype=np.float32)
        if (
            self._drive_gains is None
            or not np.array_equal(self._drive_gains[0], kps)
            or not np.array_equal(self._drive_gains[1], kds)
        ):
            view.set_gains(kps=kps.reshape(1, -1), kds=kds.reshape(1, -1))
            self._drive_gains = (kps.copy(), kds.copy())
        view.set_joint_position_targets(np.asarray(q_des, dtype=np.float32).reshape(1, -1))
        view.set_joint_velocity_targets(np.asarray(dq_des, dtype=np.float32).reshape(1, -1))
        self._art.set_joint_efforts(np.asarray(tau_ff, dtype=np.float32))
