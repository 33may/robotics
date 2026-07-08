"""frame_channel.py — the dedicated camera-frame transport (§7, design.md D6/D7).

A SEPARATE `AF_UNIX` **`SOCK_STREAM`** channel from the fixed-size SEQPACKET control
socket, because frames are large (~MB) and slow (~30 Hz). Both ends run a background
thread over a single-slot latest-wins mailbox:

  - `FrameChannelServer.publish()` is O(1) — it drops the encoded frame into the mailbox
    and returns immediately, so the World's control/render loop is NEVER blocked by a
    slow or absent brain (D7). A sender thread does the blocking `sendall`; if the
    producer overwrites the mailbox first, only the newest frame goes out (latest-wins).
  - `FrameChannelClient.read_latest()` returns the newest complete frame received since
    the last read (or None). A receiver thread reassembles length-prefixed frames off
    the byte stream (see `frame_protocol`).

PURE: stdlib sockets/threads + the pure `frame_protocol` — no isaacsim/limxsdk, no
numpy. The bytes it carries are produced by `codec.encode_camera_frame`.
"""

from __future__ import annotations

import os
import socket as _sock
import threading
import time
from typing import Dict, Optional

from . import frame_protocol as fp


class FrameChannelError(RuntimeError):
    """Raised on bind/connect failure of the frame channel."""


def _recv_exact(conn: _sock.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes, or None on EOF / socket error (channel closing)."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(conn: _sock.socket) -> Optional[bytes]:
    """Reassemble ONE complete length-prefixed frame (header + RGB + depth) off a
    stream socket, or None on EOF / socket error / malformed header (channel closing).
    Shared by every stream consumer so the framing logic lives in one place."""
    header = _recv_exact(conn, fp.HEADER_SIZE)
    if header is None:
        return None
    try:
        rgb_len, depth_len = fp.payload_lengths(header)
    except fp.FrameProtocolError:
        return None
    body = _recv_exact(conn, rgb_len + depth_len)
    if body is None:
        return None
    return header + body


class FrameChannelServer:
    """World-side frame publisher. `serve()` is non-blocking (spawns an accept+send
    thread); `publish(frame_bytes)` never blocks the caller."""

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._srv: Optional[_sock.socket] = None
        self._conn: Optional[_sock.socket] = None
        self._pending: Dict[str, bytes] = {}  # per-camera-name latest-wins mailbox
        self._cond = threading.Condition()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def serve(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)
        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        srv.bind(self._path)
        srv.listen(1)
        srv.settimeout(1.0)
        self._srv = srv
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        conn = None
        while self._running and conn is None:  # accept one client (close breaks out)
            try:
                conn, _ = self._srv.accept()
            except _sock.timeout:
                continue
            except OSError:
                return
        if conn is None:
            return
        self._conn = conn
        while self._running:  # flush every pending stream whenever one appears
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait(timeout=1.0)
                batch = list(self._pending.values())  # newest per name
                self._pending.clear()
            if not self._running:
                continue
            try:
                for frame in batch:
                    conn.sendall(frame)
            except OSError:
                break  # client gone

    def publish(self, frame_bytes: bytes) -> None:
        """Latest-wins mailbox drop, keyed by camera name so two cameras never clobber
        each other; O(1), never blocks. No-op if nobody reads it."""
        name = fp.frame_name(frame_bytes)
        with self._cond:
            self._pending[name] = frame_bytes
            self._cond.notify()

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass


class FrameChannelClient:
    """Brain-side frame reader. A receiver thread reassembles frames off the stream
    into a latest-wins mailbox; `read_latest()` returns the newest, or None."""

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._sock: Optional[_sock.socket] = None
        self._latest: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def connect(self, timeout: float = 10.0) -> None:
        sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock.connect(self._path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    sock.close()
                    raise FrameChannelError(
                        f"could not connect frame channel at {self._path}"
                    )
                time.sleep(0.02)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        conn = self._sock
        while self._running:
            buf = recv_frame(conn)
            if buf is None:
                break
            with self._lock:
                self._latest = buf

    def read_latest(self) -> Optional[bytes]:
        """Newest complete frame since the last read (latest-wins), or None."""
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(_sock.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
