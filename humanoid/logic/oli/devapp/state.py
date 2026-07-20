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

    # Nav — written by BrainLink when a debug-pose stream is attached (glide/nav mode).
    latest_pose: Optional[object] = None          # nav.RobotPose (map-frame x, y, yaw)
    latest_path: Optional[object] = None          # list[(x, y)] planned waypoints, or None
    nav_goal: Optional[object] = None             # nav.GoalCoordinate (UI-set), or None
    nav_armed: bool = False                        # UI "Engage": armed → Nav drives, else teleop

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

    # Nav has two independent writers: the brain worker thread owns `pose` (written every
    # tick), the UI thread owns `goal` (set on map click). `path` is owned by whoever plans
    # — the MapPanel today (click preview), the Nav reason once execution is wired. Split
    # setters so a per-tick pose write never clobbers a click-set goal/path (and vice-versa).

    def set_pose(self, pose) -> None:
        """Publish the latest localized pose (brain worker thread, every tick)."""
        with self._lock:
            self.latest_pose = pose

    def set_goal(self, goal) -> None:
        """Set/clear the nav goal — a `GoalCoordinate` (UI thread, on map click). `None` clears."""
        with self._lock:
            self.nav_goal = goal

    def get_goal(self):
        """Read the current nav goal (brain thread, to feed the Nav layer). `GoalCoordinate|None`."""
        with self._lock:
            return self.nav_goal

    def set_armed(self, armed: bool) -> None:
        """Engage/disengage autonomy (UI thread, on the map button)."""
        with self._lock:
            self.nav_armed = bool(armed)

    def get_armed(self) -> bool:
        """Read the arm flag (brain thread, to gate Teleop↔Nav)."""
        with self._lock:
            return self.nav_armed

    def set_path(self, path) -> None:
        """Publish the latest planned path (planner owner). `None` clears it."""
        with self._lock:
            self.latest_path = path

    def nav_snapshot(self):
        """Read the latest nav state: (pose, path, goal). Any may be None."""
        with self._lock:
            return self.latest_pose, self.latest_path, self.nav_goal

    # Localization validation overlays (slam-demo-loop D8): GT ghost + loc host state are
    # DISPLAY-ONLY — nothing in the demo loop reads them back. Brain worker thread writes.

    def set_gt_pose(self, pose) -> None:
        """Publish the latest GT oracle pose for the map ghost (brain thread). `None` = none."""
        with self._lock:
            self.gt_pose = pose

    def set_loc_state(self, state) -> None:
        """Publish the localization host's state string + latest status name (brain thread)."""
        with self._lock:
            self.loc_state = state

    def loc_snapshot(self):
        """Read (gt_pose, loc_state) for the map overlay/readout. Either may be None."""
        with self._lock:
            return getattr(self, "gt_pose", None), getattr(self, "loc_state", None)

    def set_loc_diag(self, diag) -> None:
        """Publish the localizer's display-only diagnostics dict (brain thread)."""
        with self._lock:
            self.loc_diag = diag

    def get_loc_diag(self):
        with self._lock:
            return getattr(self, "loc_diag", None)

    # Panel → brain commands (same UI-writes/brain-reads split as the nav goal): the
    # Localization panel sets one pending command; the brain loop consumes it exactly once.

    def set_loc_command(self, command) -> None:
        """Request a localizer lifecycle action from the UI: 'rehint' | 'stop' | None."""
        with self._lock:
            self.loc_command = command

    def pop_loc_command(self):
        """Consume the pending command (brain thread). Returns None when there is none."""
        with self._lock:
            cmd = getattr(self, "loc_command", None)
            self.loc_command = None
            return cmd
