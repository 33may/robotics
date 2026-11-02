"""The public Python API for a Porthole backend.

The bus is deliberately thin. It publishes data and hands you commands. It does
not interpret commands, does not validate them, and does not act on them — see
`docs/bus-protocol.md` §7. All authority lives in your application.

Typical use::

    from porthole import PortholeBus

    bus = PortholeBus(app="inspection").start()
    bus.declare("camera/wrist", qos="stream", kind="image/jpeg")
    bus.declare("decision/request", qos="event")

    bus.publish("camera/wrist", bus.jpeg_payload(rgb))
    bus.publish("scene/cloud", {"points": bus.array_payload(points.astype("float32"))})

    for command in bus.commands():
        if command["cmd"] == "decision/answer":
            ...
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any, Literal

import msgpack

from . import codec
from ._server import _BusServer

QoS = Literal["stream", "event"]

PROTOCOL_VERSION = 1
_VERSION = "0.1.0"


#: How many messages a topic keeps for replay to late-joining clients.
#: Stream topics keep only the newest — an old camera frame is worthless.
#: Event topics keep a backlog, so a log panel opens with recent history.
STREAM_HISTORY = 1
EVENT_HISTORY = 256


class _TopicSpec:
    __slots__ = ("qos", "kind", "retain", "history", "seq")

    def __init__(self, qos: QoS, kind: str, retain: bool, history: int) -> None:
        self.qos: QoS = qos
        self.kind = kind
        self.retain = retain
        self.history = history
        self.seq = 0


class PortholeBus:
    """A WebSocket topic bus that a Porthole UI connects to.

    One bus per application. Safe to publish from any thread.
    """

    # Re-exported so a backend needs exactly one import.
    array_payload = staticmethod(codec.array_payload)
    image_payload = staticmethod(codec.image_payload)
    jpeg_payload = staticmethod(codec.jpeg_payload)

    def __init__(
        self,
        app: str = "app",
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.app = app
        self.host = host
        self.port = port

        self._topics: dict[str, _TopicSpec] = {}
        self._commands: queue.Queue[dict] = queue.Queue()
        self._server = _BusServer(self._build_hello, self._commands.put, host, port)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._client_arrived = threading.Event()

    # ── declaration ─────────────────────────────────────────────────────────

    def declare(
        self,
        topic: str,
        qos: QoS = "stream",
        kind: str = "json",
        retain: bool = True,
        history: int | None = None,
    ) -> "PortholeBus":
        """Declare a topic's QoS and payload kind.

        Declaring is optional — publishing to an undeclared topic works and
        registers it with defaults. Declare anyway: the declarations are sent
        to every client in ``$/hello``, which is how a developer discovers what
        a backend offers without reading its source.

        ``qos="event"`` for anything a human must not miss (log lines, prompts,
        state transitions). ``qos="stream"`` for sampled data, where receiving
        message N makes N-1 worthless.

        ``retain=False`` only for one-shot notifications that would be wrong to
        replay to a client that connects later.

        ``history`` overrides how many messages are kept for replay. It defaults
        to 1 for streams and 256 for events; raise it for a log you want fully
        available to a UI opened late, lower it for large payloads.
        """
        if qos not in ("stream", "event"):
            raise ValueError(f"qos must be 'stream' or 'event', got {qos!r}")
        if history is None:
            history = EVENT_HISTORY if qos == "event" else STREAM_HISTORY
        if history < 1:
            raise ValueError("history must be at least 1; use retain=False instead")
        self._topics[topic] = _TopicSpec(qos, kind, retain, history)
        return self

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self, timeout: float = 5.0) -> "PortholeBus":
        """Start the server on a background thread and return self."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._server.main(self._ready)),
            name="porthole-bus",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError(f"bus failed to bind ws://{self.host}:{self.port}")
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._call_on_loop(self._server.request_stop)
        self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "PortholeBus":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ── publishing ──────────────────────────────────────────────────────────

    def publish(self, topic: str, payload: Any, ts: float | None = None) -> None:
        """Publish one message. Thread-safe.

        Publish *state*, not deltas: the last message on a topic is retained and
        replayed to late-joining clients, so a payload that only makes sense
        applied on top of the previous one leaves late clients permanently wrong.
        """
        spec = self._topics.get(topic)
        if spec is None:
            spec = _TopicSpec("stream", "json", True, STREAM_HISTORY)
            self._topics[topic] = spec

        frame = codec.encode_envelope(topic, spec.seq, payload, ts)
        spec.seq += 1
        self._call_on_loop(
            self._server.dispatch, topic, spec.qos, frame, spec.retain, spec.history
        )

    # ── commands ────────────────────────────────────────────────────────────

    def commands(self) -> Iterator[dict]:
        """Block and yield commands as the UI sends them.

        Commands are untrusted input from a human. Validate everything.
        """
        while True:
            yield self._commands.get()

    def poll(self, timeout: float | None = 0.0) -> dict | None:
        """Return the next command, or None if none arrived within ``timeout``."""
        try:
            return self._commands.get(timeout=timeout) if timeout else \
                self._commands.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[dict]:
        """Return every command queued so far, without blocking."""
        out: list[dict] = []
        while (command := self.poll()) is not None:
            out.append(command)
        return out

    # ── clients ─────────────────────────────────────────────────────────────

    @property
    def clients(self) -> int:
        """Number of currently connected UIs."""
        return self._server.client_count

    def wait_for_client(self, timeout: float | None = None) -> bool:
        """Block until at least one UI is connected. Returns False on timeout.

        Useful at startup so the first frames are not published into the void —
        though retention (§5) means a late client still sees current state.
        """
        step = 0.05
        waited = 0.0
        while self.clients == 0:
            if timeout is not None and waited >= timeout:
                return False
            time.sleep(step)
            waited += step
        return True

    # ── internals ───────────────────────────────────────────────────────────

    def _build_hello(self) -> bytes:
        payload = {
            "protocol": PROTOCOL_VERSION,
            "app": self.app,
            "server": f"porthole {_VERSION}",
            "topics": [
                {"topic": name, "qos": spec.qos, "kind": spec.kind}
                for name, spec in self._topics.items()
            ],
        }
        return msgpack.packb(
            {"topic": "$/hello", "seq": 0, "ts": time.time(), "payload": payload},
            use_bin_type=True,
        )

    def _call_on_loop(self, fn, *args) -> None:
        loop = self._server._loop
        if loop is None:
            raise RuntimeError("bus not started — call Bus.start() first")
        loop.call_soon_threadsafe(fn, *args)
