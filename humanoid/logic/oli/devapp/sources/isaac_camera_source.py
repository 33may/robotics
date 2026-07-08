"""isaac_camera_source.py — live CameraSource backed by the World frame channel (MAY-149).

Wraps MAY-149's `CameraStreamReader` (per-camera latest-wins demux over the World's
dedicated `SOCK_STREAM` frame channel) behind the dev app's `CameraSource` protocol, so the
`CameraPanel` shows LIVE Isaac RGBD with ZERO panel change versus the synthetic source.

The frame socket comes up a beat after the control socket (the World binds it only after the
8-tick camera warmup), so we connect on a background thread — the UI thread never blocks. Until
frames arrive `read()` returns None, which the panel renders as a "(no frame)" placeholder.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence

from ...comm.camera_stream import CameraStreamReader
from ...comm.frame_channel import FrameChannelError
from ...contracts import CameraFrame


class IsaacCameraSource:
    """CameraSource reading the World camera frame channel via `CameraStreamReader`."""

    def __init__(
        self,
        socket_path: str,
        names: Sequence[str] = ("chest", "head"),
        connect_timeout: float = 30.0,
    ) -> None:
        self._names: List[str] = list(names)
        self._reader = CameraStreamReader(socket_path)
        self.error: Optional[Exception] = None
        # Connect off the UI thread: the frame socket may not exist yet when the app boots.
        self._connect_thread = threading.Thread(
            target=self._connect, args=(connect_timeout,), daemon=True)
        self._connect_thread.start()

    def _connect(self, timeout: float) -> None:
        try:
            self._reader.connect(timeout=timeout)
        except FrameChannelError as e:   # World never served cameras → stay empty, no crash
            self.error = e

    def stream_names(self) -> List[str]:
        # Advertise the configured streams up front so the panel lays out immediately;
        # append any extra streams discovered on the wire (keeps configured order stable).
        discovered = self._reader.stream_names()
        return self._names + [n for n in discovered if n not in self._names]

    def read(self, name: str) -> Optional[CameraFrame]:
        return self._reader.read(name)

    def close(self) -> None:
        self._reader.close()
        if self._connect_thread.is_alive():
            self._connect_thread.join(timeout=2.0)
