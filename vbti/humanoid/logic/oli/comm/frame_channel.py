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


class _ClientSlot:
    """One connected consumer: its own latest-wins mailbox + sender thread, so a slow
    client only ever loses ITS frames — it never stalls the World or its siblings."""

    def __init__(self, conn: _sock.socket) -> None:
        self.conn = conn
        self.pending: Dict[str, bytes] = {}  # per-camera-name latest-wins mailbox
        self.cond = threading.Condition()
        self.alive = True


class FrameChannelServer:
    """World-side frame publisher, MULTI-CLIENT. `serve()` is non-blocking (spawns an
    accept thread; each accepted consumer gets its own mailbox + sender thread);
    `publish(frame_bytes)` fans out to every connected client and never blocks the
    caller. Consumers are plural by design — dev app, recorder, localizer (MAY-173:
    the old single-accept server silently starved every consumer after the first)."""

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._srv: Optional[_sock.socket] = None
        self._slots: list = []
        self._slots_lock = threading.Lock()
        self._last: Dict[str, bytes] = {}  # newest frame per stream — seeds new clients
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None

    def serve(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)
        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        srv.bind(self._path)
        srv.listen(8)
        srv.settimeout(1.0)
        self._srv = srv
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except _sock.timeout:
                continue
            except OSError:
                return
            slot = _ClientSlot(conn)
            with self._slots_lock:
                # Seed with the newest frame per stream: a client that connects between
                # publishes (or slightly before its accept) starts from the latest state
                # instead of silently losing the connect→accept window.
                slot.pending.update(self._last)
                self._slots.append(slot)
            threading.Thread(target=self._sender, args=(slot,), daemon=True).start()

    def _sender(self, slot: _ClientSlot) -> None:
        while self._running and slot.alive:
            with slot.cond:
                while self._running and slot.alive and not slot.pending:
                    slot.cond.wait(timeout=1.0)
                batch = list(slot.pending.values())  # newest per name
                slot.pending.clear()
            if not (self._running and slot.alive):
                break
            try:
                for frame in batch:
                    slot.conn.sendall(frame)
            except OSError:
                break  # client gone
        slot.alive = False
        try:
            slot.conn.close()
        except OSError:
            pass
        with self._slots_lock:
            if slot in self._slots:
                self._slots.remove(slot)

    def publish(self, frame_bytes: bytes) -> None:
        """Latest-wins mailbox drop into EVERY connected client's slot, keyed by camera
        name so two cameras never clobber each other; never blocks. No-op with no clients."""
        name = fp.frame_name(frame_bytes)
        with self._slots_lock:
            self._last[name] = frame_bytes
            slots = list(self._slots)
        for slot in slots:
            with slot.cond:
                slot.pending[name] = frame_bytes
                slot.cond.notify()

    def close(self) -> None:
        self._running = False
        with self._slots_lock:
            slots = list(self._slots)
        for slot in slots:
            slot.alive = False
            with slot.cond:
                slot.cond.notify_all()
            try:
                slot.conn.close()
            except OSError:
                pass
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
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
