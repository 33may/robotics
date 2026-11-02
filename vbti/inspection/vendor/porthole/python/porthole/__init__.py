"""Porthole — a window into a running Python process.

The Python side of the Porthole UI toolkit: a WebSocket topic bus that streams
data to one or more UIs and hands back the commands a human sent.

The backend is the only long-lived process. The UI is IO only: it renders what
you publish and reports what a human clicked. Every decision, every validation,
and every action on hardware happens here, in Python.

See `docs/bus-protocol.md` for the wire contract.
"""

from .bus import PROTOCOL_VERSION, PortholeBus
from .codec import array_payload, image_payload, jpeg_payload

__all__ = [
    "PortholeBus",
    "array_payload",
    "image_payload",
    "jpeg_payload",
    "PROTOCOL_VERSION",
]
__version__ = "0.1.0"
