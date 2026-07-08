"""camera_stream.py — the multi-stream brain/consumer camera reader (C1, MAY-149).

The raw `FrameChannelClient` is SINGLE-SLOT: a chest frame arriving right after a head
frame overwrites it before the consumer reads. With two cameras that clobbers.
`CameraStreamReader` runs its own receiver thread that reassembles frames off the
`SOCK_STREAM` (via the shared `frame_channel.recv_frame`) and demuxes each by camera
name into a per-stream latest-wins mailbox, then hands back a decoded
`contracts.CameraFrame`.

  - `read(name)` is NON-consuming (a display feed re-reads the newest frame until a
    newer one replaces it) and never blocks — a dict lookup under a lock.
  - `stream_names()` lists the streams seen so far (discovered from the wire).

This is the exact class the dev app's `IsaacCameraSource` wraps. Brain-side typed
consumer: imports the codec (numpy) + the pure transport — no isaacsim/limxsdk.
"""

from __future__ import annotations

import socket as _sock
import threading
import time
from typing import Dict, List, Optional

from ..contracts import CameraFrame
from .codec import decode_camera_frame
from .frame_channel import FrameChannelError, recv_frame


class CameraStreamReader:
    """Connects the World's frame channel and demuxes frames by camera name into
    per-stream latest-wins mailboxes. One reader per frame socket (the server accepts
    a single consumer)."""

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._sock: Optional[_sock.socket] = None
        self._latest: Dict[str, CameraFrame] = {}
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
            try:
                frame = decode_camera_frame(buf)
            except Exception:  # a single malformed frame must not kill the receiver
                continue
            with self._lock:
                self._latest[frame.name] = frame

    def read(self, name: str) -> Optional[CameraFrame]:
        """Newest frame for `name`, or None if none has arrived. Non-blocking,
        non-consuming (stays until a newer frame for that stream replaces it)."""
        with self._lock:
            return self._latest.get(name)

    def stream_names(self) -> List[str]:
        """Streams seen so far, discovered from the wire (sorted)."""
        with self._lock:
            return sorted(self._latest.keys())

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
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
