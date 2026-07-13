"""service/host.py — ServiceHost: the one-object assembly `brain_main --service` bolts on.

Owns every channel of the seam: the W4 `GoalChannelServer` (poll each loop iteration —
cheap, non-blocking, latest-wins into Nav), the W5 `TelemetryServer` behind a
`TelemetryPublisher` (`recorder` goes in the Orchestrator's recorder slot), and — when a
localization host is attached (`--shadow`) — the loc-ctrl lifecycle channel plus the est /
loc_state telemetry feed. With a host attached, `recorder` also feeds it the per-tick
(obs, intent) it needs to assemble `LocalizationIn` bundles. Stdlib only; never
isaacsim/limxsdk, never devapp.
"""

from __future__ import annotations

from typing import Optional

from .goal_channel import DEFAULT_GOAL_SOCKET, GoalChannelServer
from .loc_ctrl import DEFAULT_LOC_CTRL_SOCKET, LocCtrlServer
from .telemetry import DEFAULT_TELEMETRY_SOCKET, TelemetryPublisher, TelemetryServer


class ServiceHost:
    """All service channels, one lifecycle: construct → poll()/recorder each tick → close()."""

    def __init__(
        self,
        nav,
        goal_socket: str = DEFAULT_GOAL_SOCKET,
        telemetry_socket: str = DEFAULT_TELEMETRY_SOCKET,
        loc_host=None,                       # LocalizationHost when --shadow/--localizer
        loc_ctrl_socket: str = DEFAULT_LOC_CTRL_SOCKET,
    ) -> None:
        self._goal_server = GoalChannelServer(goal_socket, nav)
        self._telemetry_server = TelemetryServer(telemetry_socket)
        self._loc_host = loc_host
        self._loc_ctrl: Optional[LocCtrlServer] = (
            LocCtrlServer(loc_ctrl_socket, loc_host) if loc_host is not None else None
        )
        publisher = TelemetryPublisher(
            self._telemetry_server, nav,
            est_source=(loc_host.latest if loc_host is not None else None),
            loc_status=((lambda: (loc_host.state, loc_host.last_error))
                        if loc_host is not None else None),
        )
        if loc_host is None:
            self.recorder = publisher
        else:
            def recorder(obs, policy_in, action_out, joy):
                loc_host.on_tick(obs, policy_in.intent)   # feed the host its obs/intent
                publisher(obs, policy_in, action_out, joy)
            self.recorder = recorder

    def poll(self):
        """Drain W4 (+ loc-ctrl when attached). Call once per loop iteration."""
        if self._loc_ctrl is not None:
            self._loc_ctrl.poll()
        return self._goal_server.poll()

    def close(self) -> None:
        self._goal_server.close()
        self._telemetry_server.close()
        if self._loc_ctrl is not None:
            self._loc_ctrl.close()
        if self._loc_host is not None:
            self._loc_host.close()
