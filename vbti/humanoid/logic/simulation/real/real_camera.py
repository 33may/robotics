"""real_camera.py — the RealComm camera edge, DEFERRED stub (§9, design.md D9).

On the real robot, camera frames come from the RealSense stack — `realsense_mros` on
the Jetson, published on the MROS bus as `/camera/{name}` (see
`oli-corpus://oli-main-2.2.12#install/etc/realsense_mros/realsense_mros.yaml`). They are
NOT rendered. The real World edge (the py3.8 limxsdk edge — see `limx_world_main`) will
subscribe to those topics, convert each realsense-ros image into our `CameraFrame`
(RGB uint8 + depth float32 m + D435i intrinsics from `camera_mounts`), and publish it on
the SAME `FrameChannelServer` that the sim uses. The brain therefore consumes a
byte-identical `CameraFrame` stream whether the World is Isaac or the physical robot.

This module only LOCKS that contract so the real path drops in later without touching the
brain. It is intentionally unimplemented — there is no hardware yet.
"""

from __future__ import annotations

from ...oli.contracts import CameraFrame


class RealCameraSource:
    """Deferred real RealSense → `CameraFrame` source. Locks the signature; no hardware."""

    def read(self, name: str) -> CameraFrame:
        raise NotImplementedError(
            "RealComm camera edge is deferred (no hardware). The real path subscribes to "
            f"/camera/{name} (realsense_mros), converts the realsense-ros image to a "
            "CameraFrame with D435i intrinsics from camera_mounts, and publishes it via "
            "the shared FrameChannelServer."
        )
