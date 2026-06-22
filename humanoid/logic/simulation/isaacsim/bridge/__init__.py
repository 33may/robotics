"""
bridge package — Isaac ↔ limxsdk MROS sim peer.

  - `protocol.py`  — wire format (HELLO/CMD/STATE_IMU), Py3.8/3.11 byte-identical.
  - `sidecar.py`   — Py 3.8 process owning `limxsdk.Robot` (run in the `limx` env).
  - `OliBridge`    — Py 3.11 client + sidecar lifecycle manager (this module).

`OliBridge` is the object an `Oli` instance talks to. It satisfies the duck-typed
`BridgeLike` interface (`handshake`, `send_state_imu`, `poll_cmd`). Two named
constructors signal subprocess ownership:

    with OliBridge.spawn_sidecar(ip="127.0.0.1") as bridge:   # owns the sidecar
        oli = Oli(world, bridge=bridge)
        ...

    bridge = OliBridge.connect(socket="/tmp/limx-isaac-bridge.sock")  # external
    ...
    bridge.close()

See `humanoid/openspec/changes/may-147-isaac-limx-sdk-bridge/design.md` D6, D11, D15.
"""

from __future__ import annotations

import logging
import os
import socket as _sock
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from . import protocol as p

log = logging.getLogger("oli_bridge")

# Default Py 3.8 interpreter for the sidecar (override via env var).
_DEFAULT_SIDECAR_PY = "/home/may33/miniconda3/envs/limx/bin/python"
_SIDECAR_MODULE_PATH = str(Path(__file__).parent / "sidecar.py")
_DEFAULT_SOCKET = "/tmp/limx-isaac-bridge.sock"


class BridgeClosedError(RuntimeError):
    """Raised when an I/O op is attempted after the peer/socket closed."""


class BridgeProtocolError(RuntimeError):
    """Raised on handshake failure or an unexpected frame."""


class OliBridge:
    """UDS SEQPACKET client to the Py 3.8 sidecar; optional sidecar ownership.

    Construct via `spawn_sidecar(...)` (owns + tears down the subprocess) or
    `connect(...)` (attaches to an externally-managed sidecar). Not meant to be
    constructed directly.
    """

    def __init__(
        self,
        sock: "_sock.socket",
        socket_path: str,
        *,
        proc: Optional[subprocess.Popen] = None,
        stdout_thread: Optional[threading.Thread] = None,
    ) -> None:
        self._sock = sock
        self._socket_path = socket_path
        self._proc = proc  # non-None only when we spawned the sidecar
        self._stdout_thread = stdout_thread
        self._closed = False
        self._handshaked = False

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def connect(
        cls,
        socket: str = _DEFAULT_SOCKET,
        timeout: float = 10.0,
    ) -> "OliBridge":
        """Attach to an already-running sidecar. Does NOT own any subprocess."""
        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_SEQPACKET)
        while time.monotonic() < deadline:
            try:
                sock.connect(socket)
                log.info("connected to sidecar at %s", socket)
                return cls(sock, socket)
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.1)
        sock.close()
        raise ConnectionError(
            f"could not connect to sidecar at {socket} within {timeout}s: {last_err}"
        )

    @classmethod
    def spawn_sidecar(
        cls,
        ip: str = "127.0.0.1",
        socket: str = _DEFAULT_SOCKET,
        sidecar_py: Optional[str] = None,
        debug: bool = False,
        startup_timeout: float = 30.0,
    ) -> "OliBridge":
        """Start the Py 3.8 sidecar subprocess and connect to it.

        The returned bridge owns the subprocess: `__exit__`/`close()` tears it
        down (SIGINT → grace → SIGTERM → SIGKILL). Sidecar stdout/stderr is
        forwarded to ours with a `[sidecar]` prefix.
        """
        sidecar_py = (
            sidecar_py
            or os.environ.get("LIMX_BRIDGE_SIDECAR_PY")
            or _DEFAULT_SIDECAR_PY
        )
        if not os.path.exists(sidecar_py):
            raise FileNotFoundError(f"sidecar interpreter not found: {sidecar_py}")

        # Remove a stale socket so we can detect the fresh one appearing.
        if os.path.exists(socket):
            os.unlink(socket)

        cmd = [sidecar_py, _SIDECAR_MODULE_PATH, "--ip", ip, "--socket", socket]
        if debug:
            cmd.append("--debug")
        log.info("spawning sidecar: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

        # Forward sidecar output with a prefix on a daemon thread.
        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stderr.write(f"[sidecar] {line}")
                sys.stderr.flush()

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()

        # Wait for the socket file to appear (sidecar bound) or the proc to die.
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                # Sidecar exited before binding — surface its return code.
                raise BridgeProtocolError(
                    f"sidecar exited early with code {proc.returncode} "
                    f"(see [sidecar] log above)"
                )
            if os.path.exists(socket):
                break
            time.sleep(0.05)
        else:
            cls._kill(proc)
            raise BridgeProtocolError(
                f"sidecar did not create socket {socket} within {startup_timeout}s"
            )

        # Socket exists — connect.
        try:
            bridge = cls.connect(socket, timeout=5.0)
        except ConnectionError:
            cls._kill(proc)
            raise
        bridge._proc = proc
        bridge._stdout_thread = pump
        return bridge

    # ── Handshake ─────────────────────────────────────────────────────────────

    def handshake(self, dof_names: Sequence[str]) -> None:
        """Send HELLO(dof_names) and await the sidecar's ack. Idempotent."""
        if self._handshaked:
            return
        self._check_open()
        try:
            self._sock.send(p.pack_hello(seq=0, dof_names=dof_names))
            ack = self._sock.recv(p.HEADER_SIZE)
        except OSError as e:
            raise BridgeClosedError(f"handshake I/O failed: {e}") from e
        if not ack:
            raise BridgeProtocolError("sidecar closed during handshake (EOF)")
        try:
            msg_type, _version, payload_len, _seq = p.unpack_header(ack)
        except p.ProtocolError as e:
            raise BridgeProtocolError(f"bad handshake ack: {e}") from e
        if msg_type is not p.MsgType.HELLO or payload_len != 0:
            raise BridgeProtocolError(
                f"unexpected handshake ack: type={msg_type.name} len={payload_len}"
            )
        self._handshaked = True
        log.info("handshake complete (%d joints)", len(dof_names))
        # Make subsequent recvs non-blocking for poll_cmd().
        self._sock.setblocking(False)

    # ── Per-tick I/O (called by Oli.tick) ─────────────────────────────────────

    def send_state_imu(
        self,
        seq: int,
        stamp_ns: int,
        q: Sequence[float],
        dq: Sequence[float],
        tau: Sequence[float],
        acc: Sequence[float],
        gyro: Sequence[float],
        quat_wxyz: Sequence[float],
    ) -> None:
        self._check_open()
        buf = p.pack_state_imu(seq, stamp_ns, q, dq, tau, acc, gyro, quat_wxyz)
        try:
            self._sock.send(buf)
        except BlockingIOError:
            # Send buffer full — drop this state frame (next tick supersedes it).
            pass
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise BridgeClosedError(f"sidecar gone on send: {e}") from e

    def poll_cmd(self) -> Optional[Dict]:
        """Return the next pending CMD as a dict, or None if none queued.

        Non-blocking. Oli drains in a loop and keeps the most recent.
        Dict keys: seq, stamp_ns, mode, q, dq, tau, kp, kd, parallel_solve_required.
        """
        self._check_open()
        try:
            buf = self._sock.recv(p.CMD_MSG_SIZE)
        except BlockingIOError:
            return None
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise BridgeClosedError(f"sidecar gone on recv: {e}") from e
        if not buf:  # EOF
            raise BridgeClosedError("sidecar closed the connection (EOF)")
        try:
            (seq, stamp_ns, mode, q, dq, tau, kp, kd,
             parallel) = p.unpack_cmd(buf)
        except p.ProtocolError as e:
            log.error("dropping malformed CMD: %s", e)
            return None
        return {
            "seq": seq, "stamp_ns": stamp_ns, "mode": mode,
            "q": q, "dq": dq, "tau": tau, "kp": kp, "kd": kd,
            "parallel_solve_required": parallel,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "OliBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent. Closes socket; if we spawned the sidecar, tears it down."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except Exception:  # pragma: no cover
            pass
        if self._proc is not None:
            self._kill(self._proc)
            self._proc = None
        # The sidecar unlinks its own socket on exit; clean up if it didn't.
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except OSError:  # pragma: no cover
            pass

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_open(self) -> None:
        if self._closed:
            raise BridgeClosedError("bridge is closed")

    @staticmethod
    def _kill(proc: subprocess.Popen, grace: float = 5.0) -> None:
        """SIGINT → grace → SIGTERM → SIGKILL escalation."""
        import signal
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()


__all__ = [
    "OliBridge",
    "BridgeClosedError",
    "BridgeProtocolError",
]
