"""world.py — WorldComm, the World-side server half of Communication (D4/D10/D11).

WorldComm is the only world-aware layer: it owns the PR↔native joint permutation
(by joint NAME, so it is robust to any DOF reordering), reads the body (native order)
into a PR-space Observation, and applies a PR-space PolicyOut back via the body's
native-order apply. It is the connection SERVER — the World is always-on, brain connects in.

Engine-agnostic by construction: the body is duck-typed and injected, so this module
imports NO isaacsim and NO limxsdk. The body is the Isaac `Oli` articulation (native =
Isaac DOF order) OR the `LimxBody` bus peer (native = limxsdk motor order) OR a test fake
— the same WorldComm serves all three (the `_isaac` suffix below is historical and just
means "native body order"). This lets the SAME brain drive Isaac, MuJoCo, and the real robot.

Body protocol (native body order):
    body.dof_names               -> list[str]
    body.read_joints_isaac()     -> (q, dq, tau)            each (num_dof,) array
    body.read_imu()              -> (acc, gyro, quat_wxyz)  (3,), (3,), (4,)
    body.apply_isaac(q_des, dq_des, tau_ff, kp, kd)         each (num_dof,) array
"""

from __future__ import annotations

import os
import select as _select
import socket as _sock
from typing import Optional

import numpy as np

from humanoid.logic.oli.comm import codec
from humanoid.logic.oli.comm import protocol as p
from humanoid.logic.oli.contracts import NUM_JOINTS, PR_ORDER, Observation, PolicyOut
from humanoid.logic.oli.glide import GlideCmd

_DEFAULT_SOCKET = "/tmp/oli-world.sock"


class WorldCommError(RuntimeError):
    """Raised on bind/accept/handshake failure or a dead brain connection."""


class WorldComm:
    """World-side comm server: socket endpoint + PR↔Isaac permutation + body I/O."""

    def __init__(self, body, socket_path: str = _DEFAULT_SOCKET) -> None:
        self._body = body
        self._socket_path = socket_path
        self._srv: Optional[_sock.socket] = None
        self._conn: Optional[_sock.socket] = None
        self._closed = False
        self._seq = 0

        # Build the permutation from the body's Isaac DOF order vs canonical PR.
        isaac_names = list(body.dof_names)
        pr_set, isaac_set = set(PR_ORDER), set(isaac_names)
        if pr_set != isaac_set:
            raise WorldCommError(
                f"joint-name set mismatch: in Isaac not PR "
                f"{sorted(isaac_set - pr_set)}; in PR not Isaac "
                f"{sorted(pr_set - isaac_set)}"
            )
        # pr_to_isaac[pr_idx] = isaac index of the PR joint -> read permute (Isaac→PR)
        self._pr_to_isaac = np.array(
            [isaac_names.index(n) for n in PR_ORDER], dtype=np.int64
        )
        # inverse -> apply permute (PR→Isaac)
        self._isaac_to_pr = np.empty(NUM_JOINTS, dtype=np.int64)
        self._isaac_to_pr[self._pr_to_isaac] = np.arange(NUM_JOINTS)

    # ── Permutation helpers ──────────────────────────────────────────────────

    def pr_to_isaac_vector(self, pr_vec) -> np.ndarray:
        """Permute a PR-ordered joint vector into Isaac DOF order.

        The same PR→Isaac transform `apply()` uses on commands. The World uses it
        to place a PR-space home/spawn pose onto the Isaac articulation (D4: this
        layer owns the permutation, the body stays pure Isaac order).
        """
        v = np.asarray(pr_vec, dtype=np.float32).reshape(-1)
        if v.shape != (NUM_JOINTS,):
            raise WorldCommError(
                f"pr_to_isaac_vector expects {NUM_JOINTS} values, got {v.shape}"
            )
        return v[self._isaac_to_pr]

    # ── Serve + handshake ────────────────────────────────────────────────────

    def serve(self, timeout: float = 30.0) -> None:
        """Bind, listen, accept ONE brain client, and complete the handshake."""
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_SEQPACKET)
        srv.bind(self._socket_path)
        srv.listen(1)
        srv.settimeout(timeout)
        self._srv = srv
        try:
            conn, _ = srv.accept()
        except _sock.timeout as e:
            raise WorldCommError(f"no brain connected within {timeout}s") from e
        self._conn = conn
        self._handshake(timeout)

    def _handshake(self, timeout: float) -> None:
        assert self._conn is not None
        self._conn.settimeout(timeout)
        hello = self._conn.recv(p.HELLO_MSG_SIZE)
        if not hello:
            raise WorldCommError("brain closed during handshake (EOF)")
        try:
            _seq, count, names = p.unpack_hello(hello)
        except p.ProtocolError as e:
            raise WorldCommError(f"bad HELLO: {e}") from e
        if count != NUM_JOINTS or set(names) != set(PR_ORDER):
            raise WorldCommError(f"brain HELLO joint mismatch: count={count}")
        self._conn.send(p.pack_header(p.MsgType.HELLO, 0, 0))  # ack
        self._conn.setblocking(False)

    # ── Per-tick I/O ─────────────────────────────────────────────────────────

    def publish(self, stamp_ns: int) -> None:
        """Read the body, permute Isaac→PR, and send an Observation (non-blocking).

        `stamp_ns` is supplied by the World loop (sim time) — the body does not own
        the authoritative clock (D8).
        """
        assert self._conn is not None
        q_isaac, dq_isaac, tau_isaac = self._body.read_joints_isaac()
        acc, gyro, quat_wxyz = self._body.read_imu()
        obs = Observation(
            stamp_ns=stamp_ns,
            q=np.asarray(q_isaac)[self._pr_to_isaac],
            dq=np.asarray(dq_isaac)[self._pr_to_isaac],
            tau=np.asarray(tau_isaac)[self._pr_to_isaac],
            acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
        )
        buf = codec.encode_observation(obs, seq=self._seq)
        self._seq += 1
        try:
            self._conn.send(buf)
        except BlockingIOError:
            pass  # buffer full — drop (next publish supersedes)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise WorldCommError(f"brain gone on send: {e}") from e

    def receive_latest(self) -> Optional[PolicyOut]:
        """Drain to the newest CMD frame and decode it to a PR PolicyOut."""
        assert self._conn is not None
        latest: Optional[bytes] = None
        while True:
            try:
                buf = self._conn.recv(p.CMD_MSG_SIZE)
            except BlockingIOError:
                break
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                raise WorldCommError(f"brain gone on recv: {e}") from e
            if not buf:
                raise WorldCommError("brain closed the connection (EOF)")
            latest = buf
        if latest is None:
            return None
        try:
            return codec.decode_policy_out(latest)
        except p.ProtocolError:
            return None

    def receive_blocking(self, timeout: float) -> Optional[PolicyOut]:
        """Wait up to `timeout`s for a CMD, then drain to the newest and decode it.

        For lock-step pacing (pace World ← brain): the World blocks on the brain's
        command for the just-published Observation so each command is applied to the
        exact state it was computed from (zero sensor→actuator latency). Returns None
        on timeout, letting the caller run its watchdog instead of deadlocking.
        """
        assert self._conn is not None
        ready, _, _ = _select.select([self._conn], [], [], timeout)
        if not ready:
            return None
        return self.receive_latest()

    def receive_glide_latest(self) -> Optional[GlideCmd]:
        """Drain to the newest GLIDE_CMD frame and decode it (glide mode — MAY-172).

        The glide World loop consumes these instead of CMD; the walk `receive_latest`
        above is untouched. In glide mode the brain sends only GLIDE_CMD frames, so a
        `recv(GLIDE_CMD_MSG_SIZE)` reads each whole datagram (SEQPACKET preserves bounds).
        """
        assert self._conn is not None
        latest: Optional[bytes] = None
        while True:
            try:
                buf = self._conn.recv(p.GLIDE_CMD_MSG_SIZE)
            except BlockingIOError:
                break
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                raise WorldCommError(f"brain gone on recv: {e}") from e
            if not buf:
                raise WorldCommError("brain closed the connection (EOF)")
            latest = buf
        if latest is None:
            return None
        try:
            return codec.decode_glide_cmd(latest)
        except p.ProtocolError:
            return None

    def receive_glide_blocking(self, timeout: float) -> Optional[GlideCmd]:
        """Wait up to `timeout`s for a GLIDE_CMD, then drain to the newest and decode it.

        Glide analogue of `receive_blocking` — lets the glide World loop pace lock-step
        off the brain and fall through to its watchdog on silence.
        """
        assert self._conn is not None
        ready, _, _ = _select.select([self._conn], [], [], timeout)
        if not ready:
            return None
        return self.receive_glide_latest()

    def apply(self, policy_out: PolicyOut) -> None:
        """Permute the PR PolicyOut into native order and apply via the implicit drive."""
        po = policy_out
        self._body.apply_isaac(
            q_des=po.q_des[self._isaac_to_pr],
            dq_des=po.dq_des[self._isaac_to_pr],
            tau_ff=po.tau_ff[self._isaac_to_pr],
            kp=po.kp[self._isaac_to_pr],
            kd=po.kd[self._isaac_to_pr],
        )

    def set_command(self, policy_out: PolicyOut) -> None:
        """Permute the PR PolicyOut into native order and store it as the explicit-torque
        target (the World recomputes τ per substep via the body). Same permutation as
        `apply()`; only the actuator path differs — implicit drive vs explicit per-substep
        torque. Requires a body that implements `set_command_isaac` (the Isaac `Oli`).
        """
        po = policy_out
        self._body.set_command_isaac(
            q_des=po.q_des[self._isaac_to_pr],
            dq_des=po.dq_des[self._isaac_to_pr],
            tau_ff=po.tau_ff[self._isaac_to_pr],
            kp=po.kp[self._isaac_to_pr],
            kd=po.kd[self._isaac_to_pr],
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except Exception:  # pragma: no cover
                    pass
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except OSError:  # pragma: no cover
            pass
