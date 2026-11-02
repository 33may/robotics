"""The RealComm camera edge is a deferred stub (§9) — no hardware yet.

It exists only to LOCK the contract: the real path (RealSense → CameraFrame, published
on the same frame channel SimComm uses) drops in later without touching the brain. This
test pins that the interface is present and intentionally unimplemented. Brain-pure.
"""

import pytest

from humanoid.logic.simulation.real.real_camera import RealCameraSource

pytestmark = pytest.mark.brain


def test_real_camera_source_is_deferred():
    src = RealCameraSource()
    with pytest.raises(NotImplementedError):
        src.read("chest")
