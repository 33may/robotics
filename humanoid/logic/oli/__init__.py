"""humanoid.logic.oli — the deployment-invariant Oli brain (Reason + Action).

Pure of `isaacsim`/`limxsdk` by construction: importing this package or any of its
submodules (`contracts`, `comm.client`, `reason`, `action`, `runtime`) must never
pull in a world SDK. World-specific code (the PR↔Isaac permutation, the
articulation) lives under `logic/simulation/isaacsim` behind the Communication
boundary — never here. That import-cleanliness is the enforceable invariant that
lets the same brain binary drive the Isaac sim and the physical robot.
"""

from .contracts import (
    NUM_JOINTS,
    PR_ORDER,
    CameraFrame,
    CameraIntrinsics,
    Intent,
    Mode,
    Observation,
    PolicyIn,
    PolicyOut,
)

__all__ = [
    "NUM_JOINTS",
    "PR_ORDER",
    "CameraFrame",
    "CameraIntrinsics",
    "Intent",
    "Mode",
    "Observation",
    "PolicyIn",
    "PolicyOut",
]
