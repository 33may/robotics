"""oli/service — the brain's goal-in / telemetry-out service seam (locbench design.md D5).

The brain runs isolated (`brain_main`, own process); this package is the small socket seam any
client uses to steer and observe it: W4 carries `GoalCoordinate` set/clear IN (→
`Nav.set_goal/clear_goal`), W5 carries the brain's telemetry snapshot OUT (pose, path, goal
status, est `LocalizationOut`, intent, loop rate). Clients today: the locbench evaluator;
dev_app migrates onto this seam later to become pure visuals. Pure brain-side code — stdlib
only, no isaacsim/limxsdk, no devapp imports (guarded by the architecture tests).
"""

from .goal_channel import DEFAULT_GOAL_SOCKET, GoalChannelClient, GoalChannelServer
from .host import ServiceHost
from .protocol import (
    GOAL_NBYTES,
    TelemetrySnapshot,
    decode_goal,
    decode_telemetry,
    encode_goal_clear,
    encode_goal_set,
    encode_telemetry,
)
from .telemetry import (
    DEFAULT_TELEMETRY_SOCKET,
    TelemetryClient,
    TelemetryPublisher,
    TelemetryServer,
)

__all__ = [
    "DEFAULT_GOAL_SOCKET",
    "DEFAULT_TELEMETRY_SOCKET",
    "GOAL_NBYTES",
    "GoalChannelClient",
    "GoalChannelServer",
    "ServiceHost",
    "TelemetryClient",
    "TelemetryPublisher",
    "TelemetryServer",
    "TelemetrySnapshot",
    "decode_goal",
    "decode_telemetry",
    "encode_goal_clear",
    "encode_goal_set",
    "encode_telemetry",
]
