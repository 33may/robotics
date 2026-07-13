"""service/host.py — ServiceHost: the one-object assembly `brain_main --service` bolts on.

Owns both channels of the seam: the W4 `GoalChannelServer` (call `poll()` every brain-loop
iteration — cheap, non-blocking, latest-wins into Nav) and the W5 `TelemetryServer` behind a
`TelemetryPublisher` (`recorder` is the callable to hand the Orchestrator's recorder slot).
`est_source` stays None until the in-brain localization host (§5) plugs its latest
`LocalizationOut` in. Stdlib only; never isaacsim/limxsdk, never devapp.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..reason.localization import LocalizationOut
from .goal_channel import DEFAULT_GOAL_SOCKET, GoalChannelServer
from .telemetry import DEFAULT_TELEMETRY_SOCKET, TelemetryPublisher, TelemetryServer


class ServiceHost:
    """Both service channels, one lifecycle: construct → poll()/recorder each tick → close()."""

    def __init__(
        self,
        nav,
        goal_socket: str = DEFAULT_GOAL_SOCKET,
        telemetry_socket: str = DEFAULT_TELEMETRY_SOCKET,
        est_source: Optional[Callable[[], Optional[LocalizationOut]]] = None,
    ) -> None:
        self._goal_server = GoalChannelServer(goal_socket, nav)
        self._telemetry_server = TelemetryServer(telemetry_socket)
        self.recorder = TelemetryPublisher(self._telemetry_server, nav, est_source=est_source)

    def poll(self):
        """Drain W4 and apply the newest goal command to Nav. Call once per loop iteration."""
        return self._goal_server.poll()

    def close(self) -> None:
        self._goal_server.close()
        self._telemetry_server.close()
