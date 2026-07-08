"""state.py — AppState: the latest-wins buffer shared UI thread ↔ brain thread.

The brain worker thread (BrainLink) writes the newest contracts via `set_brain`; UI panels
read them each frame via `brain_snapshot`. Single-writer / single-reader, guarded by a lock
for tear-free reads. Camera frames are NOT buffered here — a CameraSource owns its own
latest-wins buffer — so the source can be swapped sim↔real without touching AppState.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AppState:
    """Shared, time-varying app/brain state read by panels each frame."""

    #: Incremented once per rendered UI frame (by the shell). Handy for liveness.
    frame_index: int = 0

    # Latest brain contracts (None until a brain is attached and stepping).
    brain_attached: bool = False
    latest_obs: Optional[object] = None          # contracts.Observation
    latest_policy_out: Optional[object] = None    # contracts.PolicyOut
    mode_name: str = "—"                          # current Intent mode (STAND/WALK)
    latest_intent: Optional[object] = None        # contracts.Intent (joystick-derived command)
    latest_joy: Optional[object] = None           # raw JoyPacket (axes/buttons); None if no packet

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def tick(self) -> None:
        """Advance the UI frame counter (called by the shell each frame)."""
        with self._lock:
            self.frame_index += 1

    def set_brain(self, obs, policy_out, mode_name: str, intent=None, joy=None) -> None:
        """Publish the newest brain contracts (called from the brain worker thread)."""
        with self._lock:
            self.brain_attached = True
            self.latest_obs = obs
            self.latest_policy_out = policy_out
            self.mode_name = mode_name
            self.latest_intent = intent
            self.latest_joy = joy

    def brain_snapshot(self) -> Tuple[bool, object, object, str]:
        """Read the latest brain state: (attached, obs, policy_out, mode_name)."""
        with self._lock:
            return self.brain_attached, self.latest_obs, self.latest_policy_out, self.mode_name

    def teleop_snapshot(self) -> Tuple[object, object]:
        """Read the latest joystick input: (intent, joy). Either may be None."""
        with self._lock:
            return self.latest_intent, self.latest_joy
