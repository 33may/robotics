"""policy_runner.py — the Action layer: obs encoding, ONNX, resolution to a PD cmd.

Selection is by `Intent.mode` (D5/D6, kept simple): WALK runs the LimX walk ONNX,
STAND is analytic. All policy-specifics live here — per-component obs scales, the
quaternion `wxyz→xyzw` reorder + projected gravity, the 5-deep proprio history ring,
`last_actions`, action clipping, the per-joint torque-limit clamp, and resolution
`q_des = action·scale + default`. The contracts that cross the World seam carry none
of this. Pure of isaacsim/limxsdk; runs in the `brain` env.

Authoritative reference: `vendor/humanoid-rl-deploy-python/controllers/HU_D04_01/
walk_controller/{walk_controller.py,walk_param.yaml}` — replicated component-by-component.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

from ..contracts import NUM_JOINTS, Intent, Mode, PolicyIn, PolicyOut

FloatArray = Union[Sequence[float], np.ndarray]

_GRAVITY = np.array([0.0, 0.0, -1.0])

# Vendored LimX deploy controllers for HU_D04_01.
_CTRL_DIR = (
    Path(__file__).resolve().parents[3]
    / "vendor" / "humanoid-rl-deploy-python" / "controllers" / "HU_D04_01"
)
DEFAULT_WALK_PARAM = _CTRL_DIR / "walk_controller" / "walk_param.yaml"
DEFAULT_WALK_ONNX = _CTRL_DIR / "walk_controller" / "policy" / "default" / "policy.onnx"
DEFAULT_STAND_PARAM = _CTRL_DIR / "stand_controller" / "joint_params.yaml"

_OBS_SIZE = 102
_HISTORY_LEN = 5


def encode_walk_obs(
    obs,
    intent: Intent,
    last_actions: FloatArray,
    default_angle: FloatArray,
    ang_vel_scale: float = 0.25,
    dof_vel_scale: float = 0.05,
) -> np.ndarray:
    """Build the 102-dim walk observation (gait terms omitted).

    Layout (matches `walk_controller.compute_observation`):
        [ base_ang_vel·ang_vel_scale (3)
        | projected_gravity            (3)
        | commands [v_x, v_y, w_z]     (3)   — unscaled
        | (q − default_angle)·1.0     (31)
        | dq·dof_vel_scale            (31)
        | last_actions               (31) ]

    `projected_gravity` is `Rᵀ·[0,0,-1]` with the quaternion reordered `wxyz→xyzw`
    (scipy is scalar-last). This is the algebraic identity of the deploy code's
    quat→euler(zyx)→quat detour.
    """
    quat = np.asarray(obs.quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    projected_gravity = R.from_quat(quat_xyzw).inv().as_matrix() @ _GRAVITY

    base_ang_vel = np.asarray(obs.gyro, dtype=np.float64) * ang_vel_scale
    joint_pos = np.asarray(obs.q, dtype=np.float64) - np.asarray(default_angle, dtype=np.float64)
    joint_vel = np.asarray(obs.dq, dtype=np.float64) * dof_vel_scale
    commands = np.array([intent.v_x, intent.v_y, intent.w_z], dtype=np.float64)

    return np.concatenate([
        base_ang_vel,
        projected_gravity,
        commands,
        joint_pos,        # dof_pos scale = 1.0
        joint_vel,
        np.asarray(last_actions, dtype=np.float64),
    ]).astype(np.float32)


class WalkPolicy:
    """The LimX walk policy (ONNX). Selected by ``Intent.mode == Mode.WALK``.

    Owns the obs encoding, the 5-deep proprio history ring, ``last_actions``, and the
    resolution of raw actions to a PD command. Faithfully replicates
    ``walk_controller.py`` (incl. the ``last_actions`` aliasing → the policy sees the
    torque-clamped action). All params come from ``walk_param.yaml``, used positionally
    against PR-ordered observations (head 15/16 swap is symmetric → harmless).
    """

    def __init__(
        self,
        param_path: Union[str, Path] = DEFAULT_WALK_PARAM,
        onnx_path: Union[str, Path] = DEFAULT_WALK_ONNX,
    ) -> None:
        with open(param_path) as f:
            cfg = yaml.safe_load(f)["HumanoidRobotCfg"]
        ctrl, norm = cfg["control"], cfg["normalization"]

        self.action_scale = np.asarray(ctrl["action_scale"], dtype=np.float32)
        self.kp = np.asarray(ctrl["kp"], dtype=np.float32)
        self.kd = np.asarray(ctrl["kd"], dtype=np.float32)
        self.default_angle = np.asarray(ctrl["default_angle"], dtype=np.float32)
        self.user_torque_limit = np.asarray(ctrl["user_torque_limit"], dtype=np.float32)
        self.decimation = int(ctrl.get("decimation", 10))
        self.clip_actions = float(norm["clip_scales"]["clip_actions"])
        self.ang_vel_scale = float(norm["obs_scales"]["ang_vel"])
        self.dof_vel_scale = float(norm["obs_scales"]["dof_vel"])
        self._soft_torque_limit = 0.95

        for name, arr in (
            ("action_scale", self.action_scale), ("kp", self.kp), ("kd", self.kd),
            ("default_angle", self.default_angle),
            ("user_torque_limit", self.user_torque_limit),
        ):
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(
                    f"walk_param.yaml '{name}' must be length {NUM_JOINTS}, got {arr.shape}"
                )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self._history = None
        self._last_actions = np.zeros(NUM_JOINTS, dtype=np.float32)

    def _advance_history(self, obs102: np.ndarray) -> np.ndarray:
        """Push the newest obs to the front of the 5-deep ring; first obs fills ×5."""
        obs102 = np.asarray(obs102, dtype=np.float32)
        if self._history is None:
            self._history = np.tile(obs102, _HISTORY_LEN)
        else:
            # newest-first; drop the oldest (concatenate avoids overlapping in-place ops)
            self._history = np.concatenate([obs102, self._history[:-_OBS_SIZE]])
        return self._history.copy()

    def _resolve(self, actions: np.ndarray, obs) -> tuple:
        """Clip to ±clip_actions, torque-limit clamp on live q/dq, resolve q_des.

        Per-joint bound (walk_controller.py): the action is limited so the resulting
        PD torque |τ| ≤ 0.95·τlim, then ``q_des = action·scale + default``.
        """
        actions = np.clip(
            np.asarray(actions, dtype=np.float32), -self.clip_actions, self.clip_actions
        )
        q = np.asarray(obs.q, dtype=np.float32)
        dq = np.asarray(obs.dq, dtype=np.float32)
        base = q - self.default_angle + self.kd * dq / self.kp
        margin = self.user_torque_limit * self._soft_torque_limit / self.kp
        a_min = (base - margin) / self.action_scale
        a_max = (base + margin) / self.action_scale
        clamped = np.clip(actions, a_min, a_max)
        q_des = clamped * self.action_scale + self.default_angle
        return clamped.astype(np.float32), q_des.astype(np.float32)

    def step(self, policy_in: PolicyIn) -> PolicyOut:
        """One policy step: encode → history → ONNX → resolve → PolicyOut."""
        obs = policy_in.observation
        obs102 = encode_walk_obs(
            obs, policy_in.intent, self._last_actions, self.default_angle,
            self.ang_vel_scale, self.dof_vel_scale,
        )
        history = self._advance_history(obs102)
        raw = self._session.run(
            None, {self._input_name: history[np.newaxis, :].astype(np.float32)}
        )[0].flatten()
        clamped, q_des = self._resolve(raw, obs)
        self._last_actions = clamped  # aliasing fidelity: next obs sees the clamped action
        zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
        return PolicyOut(
            stamp_ns=obs.stamp_ns,
            q_des=q_des, dq_des=zeros, tau_ff=zeros,
            kp=self.kp, kd=self.kd, mode=np.zeros(NUM_JOINTS, dtype=np.int32),
        )


class StandPolicy:
    """Analytic STAND: linearly ramp from the captured spawn pose to `stand_pos` over
    `ramp_seconds`, paced by the Observation stamp (rate-independent). Stiff stand gains
    hold the pose. No ONNX. Selected by ``Intent.mode == Mode.STAND``. Faithful to
    ``stand_controller.py`` (capture init pose on entry, interpolate to stand_pos).
    """

    def __init__(
        self,
        param_path: Union[str, Path] = DEFAULT_STAND_PARAM,
        ramp_seconds: float = 2.0,
    ) -> None:
        with open(param_path) as f:
            cfg = yaml.safe_load(f)
        self.stand_pos = np.asarray(cfg["stand_pos"], dtype=np.float32)
        self.stand_kp = np.asarray(cfg["stand_kp"], dtype=np.float32)
        self.stand_kd = np.asarray(cfg["stand_kd"], dtype=np.float32)
        for name, arr in (
            ("stand_pos", self.stand_pos), ("stand_kp", self.stand_kp),
            ("stand_kd", self.stand_kd),
        ):
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(
                    f"joint_params.yaml '{name}' must be length {NUM_JOINTS}, got {arr.shape}"
                )
        self._ramp_ns = float(ramp_seconds) * 1e9
        self.reset()

    def reset(self) -> None:
        self._start_stamp = None
        self._init_q = None

    def step(self, policy_in: PolicyIn) -> PolicyOut:
        obs = policy_in.observation
        if self._start_stamp is None:  # capture the spawn pose on entry
            self._start_stamp = obs.stamp_ns
            self._init_q = np.asarray(obs.q, dtype=np.float32)
        progress = min(1.0, max(0.0, (obs.stamp_ns - self._start_stamp) / self._ramp_ns))
        q_des = (1.0 - progress) * self._init_q + progress * self.stand_pos
        zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
        return PolicyOut(
            stamp_ns=obs.stamp_ns,
            q_des=q_des.astype(np.float32), dq_des=zeros, tau_ff=zeros,
            kp=self.stand_kp, kd=self.stand_kd, mode=np.zeros(NUM_JOINTS, dtype=np.int32),
        )


class PolicyRunner:
    """Selects the active policy by ``Intent.mode`` (D5/D6). On a mode switch it re-seeds
    the entered policy (`reset()`), so WALK enters with a fresh 5-deep history from the
    current stance and STAND re-captures the spawn pose — the cold-start mitigation.
    Deliberately simple (no ABC/registry); a richer policy bank is a later design.
    """

    def __init__(self, walk: "WalkPolicy" = None, stand: StandPolicy = None) -> None:
        self.walk = walk if walk is not None else WalkPolicy()
        self.stand = stand if stand is not None else StandPolicy()
        self._active_mode = None

    def _policy_for(self, mode: Mode):
        if mode == Mode.WALK:
            return self.walk
        if mode == Mode.STAND:
            return self.stand
        raise ValueError(f"no policy registered for mode {mode!r}")

    def step(self, policy_in: PolicyIn) -> PolicyOut:
        mode = policy_in.intent.mode
        policy = self._policy_for(mode)
        if mode != self._active_mode:
            policy.reset()  # fresh history / re-captured spawn pose on (re)entry
            self._active_mode = mode
        return policy.step(policy_in)

    def reset(self) -> None:
        self.walk.reset()
        self.stand.reset()
        self._active_mode = None
