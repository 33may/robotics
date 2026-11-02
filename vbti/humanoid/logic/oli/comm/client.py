"""client.py — BrainComm, the brain-side UDS client to the World server (D10/D11).

Re-homes the old bridge client half, with the roles inverted: the World is now the
always-on server and the brain connects in (mirroring a controller `init()`-ing
into the real robot). BrainComm sends `PolicyOut` and drains the latest
`Observation`; both directions are non-blocking with latest-wins semantics. Pure:
stdlib socket + the pure wire/codec — no isaacsim, no limxsdk.
"""

from __future__ import annotations

import socket as _sock
import time
from typing import Optional, Sequence

from ..contracts import PR_ORDER, CameraFrame, Observation, PolicyOut
from . import codec
from . import protocol as p
from .base import Comm
from .frame_channel import FrameChannelClient

_DEFAULT_SOCKET = "/tmp/oli-world.sock"


class CommClosedError(RuntimeError):
    """Raised when an I/O op is attempted after the World/socket closed."""


class CommProtocolError(RuntimeError):
    """Raised on handshake failure or an unexpected frame."""


class BrainComm(Comm):
    """UDS SEQPACKET client to the World server."""

    def __init__(
        self,
        socket_path: str = _DEFAULT_SOCKET,
        dof_names: Sequence[str] = PR_ORDER,
        frame_socket_path: Optional[str] = None,
    ) -> None:
        self._socket_path = socket_path
        self._dof_names = list(dof_names)
        self._sock: Optional[_sock.socket] = None
        self._closed = False
        self._handshaked = False
        # Cameras travel a SEPARATE stream socket (design.md D6). Opt-in: only when a
        # frame path is given does the brain connect a frame reader.
        self._frame_socket_path = frame_socket_path
        self._frames: Optional[FrameChannelClient] = None

    # ── Connect + handshake ──────────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_SEQPACKET)
        while time.monotonic() < deadline:
            try:
                sock.connect(self._socket_path)
                self._sock = sock
                self._handshake()
                self._connect_frames(timeout)
                return
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.1)
        sock.close()
        raise ConnectionError(
            f"could not connect to World at {self._socket_path} "
            f"within {timeout}s: {last_err}"
        )

    def _handshake(self) -> None:
        assert self._sock is not None
        # The brain announces the canonical PR order it speaks; the World acks.
        self._sock.send(p.pack_hello(seq=0, dof_names=self._dof_names))
        ack = self._sock.recv(p.HEADER_SIZE)
        if not ack:
            raise CommProtocolError("World closed during handshake (EOF)")
        try:
            msg_type, _v, payload_len, _seq = p.unpack_header(ack)
        except p.ProtocolError as e:
            raise CommProtocolError(f"bad handshake ack: {e}") from e
        if msg_type is not p.MsgType.HELLO or payload_len != 0:
            raise CommProtocolError(
                f"unexpected handshake ack: {msg_type.name} len={payload_len}"
            )
        self._handshaked = True
        self._sock.setblocking(False)  # subsequent I/O is non-blocking

    def _connect_frames(self, timeout: float) -> None:
        """Connect the (optional) camera frame channel — a second stream socket."""
        if self._frame_socket_path is None:
            return
        frames = FrameChannelClient(self._frame_socket_path)
        frames.connect(timeout=timeout)
        self._frames = frames

    # ── Per-step I/O ─────────────────────────────────────────────────────────

    def read_camera_frame(self) -> Optional[CameraFrame]:
        """Newest CameraFrame off the frame channel (latest-wins), or None if none
        pending / no camera channel configured."""
        self._check_open()
        if self._frames is None:
            return None
        raw = self._frames.read_latest()
        if raw is None:
            return None
        try:
            return codec.decode_camera_frame(raw)
        except Exception:
            return None  # malformed frame — next read supersedes

    def read_observation(self) -> Optional[Observation]:
        """Drain to the newest STATE_IMU frame and decode it (latest-wins)."""
        self._check_open()
        assert self._sock is not None
        latest: Optional[bytes] = None
        while True:
            try:
                buf = self._sock.recv(p.STATE_IMU_MSG_SIZE)
            except BlockingIOError:
                break
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                raise CommClosedError(f"World gone on recv: {e}") from e
            if not buf:
                raise CommClosedError("World closed the connection (EOF)")
            latest = buf
        if latest is None:
            return None
        try:
            return codec.decode_observation(latest)
        except p.ProtocolError:
            return None  # malformed frame — next read supersedes

    def write_policy_out(self, policy_out: PolicyOut) -> None:
        self._check_open()
        assert self._sock is not None
        buf = codec.encode_policy_out(policy_out, seq=0)
        try:
            self._sock.send(buf)
        except BlockingIOError:
            pass  # send buffer full — drop (next cmd supersedes)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise CommClosedError(f"World gone on send: {e}") from e

    def write_glide_cmd(self, glide_cmd) -> None:
        """Glide mode: encode a GlideCmd as a GLIDE_CMD frame and send it (MAY-172).

        Same non-blocking, drop-if-full path as `write_policy_out` — only the codec
        differs, so the socket/latest-wins semantics are identical across modes.
        """
        self._check_open()
        assert self._sock is not None
        buf = codec.encode_glide_cmd(glide_cmd, seq=0)
        try:
            self._sock.send(buf)
        except BlockingIOError:
            pass  # send buffer full — drop (next cmd supersedes)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise CommClosedError(f"World gone on send: {e}") from e

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._frames is not None:
            self._frames.close()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "BrainComm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise CommClosedError("comm is closed")
