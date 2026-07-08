"""camera_publisher.py — the World-side camera → frame-channel unit (C2, MAY-149).

Reads every camera off the body each render tick, wraps each as a `CameraFrame`, and
serves them on the dedicated `SOCK_STREAM` frame channel (separate from the control
socket). Latest-wins per camera name (via `FrameChannelServer`), so a slow or absent
consumer NEVER backs up the World's loop (design.md D6/D7).

Engine-agnostic by construction: the body is duck-typed and injected, so this module
imports NO isaacsim/limxsdk — the same publisher serves the Isaac `Oli`, a MuJoCo body,
or a test fake. It is the camera sibling of `WorldComm`.

Body protocol (the camera-enabled Oli surface, §4):
    body.camera_names               -> list[str]
    body.read_camera_rgbd(name)     -> (rgb uint8 H×W×3, depth float32 H×W meters)
    body.camera_intrinsics(name)    -> CameraIntrinsics
"""

from __future__ import annotations

from typing import Optional

from humanoid.logic.oli.comm.codec import encode_camera_frame
from humanoid.logic.oli.comm.frame_channel import FrameChannelServer
from humanoid.logic.oli.contracts import CameraFrame

_DEFAULT_FRAME_SOCKET = "/tmp/oli-world-frames.sock"


class CameraPublisher:
    """Owns the frame-channel server; `publish(tick, stamp_ns)` ships every camera on
    the render sub-tick. Never blocks the caller (O(1) mailbox drop per camera)."""

    def __init__(self, body, socket_path: str = _DEFAULT_FRAME_SOCKET, every: int = 1) -> None:
        self._body = body
        self._every = max(1, int(every))
        self._server = FrameChannelServer(socket_path=socket_path)
        self._server.serve()
        self._seq = 0
        self._warned: set = set()  # cameras we've already warned about (log once)

    def publish(self, tick: int, stamp_ns: Optional[int] = None) -> None:
        """Render, wrap, and ship each camera IF `tick` is on the sub-tick. The World
        supplies the stamp (sim time, design.md D8); falls back to `tick` if omitted.

        A camera that is not ready yet (Isaac annotators need a render tick to populate,
        so the very first read can return an empty buffer) must NEVER crash the World's
        loop — an uncaught error would kill the whole simulation. Such a camera is skipped
        (warned once) and ships frames as soon as it renders."""
        if tick % self._every != 0:
            return
        stamp = int(stamp_ns) if stamp_ns is not None else int(tick)
        for name in self._body.camera_names:
            try:
                rgb, depth = self._body.read_camera_rgbd(name)
                intrinsics = self._body.camera_intrinsics(name)
                frame = CameraFrame(
                    stamp_ns=stamp, name=name, rgb=rgb, depth=depth, intrinsics=intrinsics
                )
                encoded = encode_camera_frame(frame, seq=self._seq)
            except Exception as exc:  # not-ready camera / transient render hiccup
                if name not in self._warned:
                    self._warned.add(name)
                    print(f"[camera-publisher] {name} not ready "
                          f"({type(exc).__name__}: {exc}); skipping until it renders",
                          flush=True)
                continue
            self._server.publish(encoded)
            self._seq += 1

    def close(self) -> None:
        self._server.close()
