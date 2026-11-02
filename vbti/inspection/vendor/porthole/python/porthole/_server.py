"""WebSocket server internals: client fan-out, QoS scheduling, backpressure.

This module is private. Application code uses `porthole.PortholeBus`.

The scheduling policy here is the whole point of the file, and it implements
protocol §6:

- `stream` topics keep only the newest undelivered frame per client. A slow
  client falls behind in *time*, never in memory.
- `event` topics are buffered in order and never dropped. If the buffer would
  overflow, the client is disconnected — a visible failure beats a silently
  swallowed decision request.
- Events are flushed before streams, so a prompt never queues behind a burst
  of point clouds.

Backpressure is per-client: a stalled browser tab on a slow link cannot slow
down the native window on loopback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("porthole.bus")

EVENT_BUFFER_LIMIT = 4096


class _BusClient:
    """One connected UI, with its own send scheduling state."""

    __slots__ = ("ws", "stream_pending", "stream_order", "events", "wake", "overflow")

    def __init__(self, ws) -> None:
        self.ws = ws
        self.stream_pending: dict[str, bytes] = {}
        self.stream_order: deque[str] = deque()
        self.events: deque[bytes] = deque()
        self.wake = asyncio.Event()
        self.overflow = False

    def offer(self, topic: str, qos: str, frame: bytes) -> None:
        """Queue a frame for this client according to its topic's QoS."""
        if qos == "event":
            if len(self.events) >= EVENT_BUFFER_LIMIT:
                self.overflow = True
            else:
                self.events.append(frame)
        else:
            if topic not in self.stream_pending:
                self.stream_order.append(topic)
            self.stream_pending[topic] = frame  # drop-oldest
        self.wake.set()

    def has_work(self) -> bool:
        return bool(self.events or self.stream_order or self.overflow)


class _BusServer:
    """Owns the asyncio loop, the client set, and retained state."""

    def __init__(self, hello_builder, on_command, host: str, port: int) -> None:
        self._hello_builder = hello_builder
        self._on_command = on_command
        self.host = host
        self.port = port

        self._clients: set[_BusClient] = set()
        self._retained: dict[str, deque[bytes]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Future | None = None

    # ── called from the bus thread ──────────────────────────────────────────

    def dispatch(
        self, topic: str, qos: str, frame: bytes, retain: bool, history: int
    ) -> None:
        """Fan one encoded frame out to every client. Runs on the loop thread."""
        if retain:
            backlog = self._retained.get(topic)
            if backlog is None or backlog.maxlen != history:
                backlog = deque(backlog or (), maxlen=history)
                self._retained[topic] = backlog
            backlog.append(frame)
        for client in self._clients:
            client.offer(topic, qos, frame)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── loop lifecycle ──────────────────────────────────────────────────────

    async def main(self, ready) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = self._loop.create_future()
        async with serve(self._handle, self.host, self.port, max_size=None):
            log.info("porthole bus listening on ws://%s:%d", self.host, self.port)
            ready.set()
            await self._stop

    def request_stop(self) -> None:
        if self._stop is not None and not self._stop.done():
            self._stop.set_result(None)

    # ── per-connection ──────────────────────────────────────────────────────

    async def _handle(self, ws) -> None:
        client = _BusClient(ws)
        self._clients.add(client)
        peer = getattr(ws, "remote_address", None)
        log.info("client connected (%s); %d total", peer, len(self._clients))

        pump = asyncio.create_task(self._pump(client))
        try:
            await ws.send(self._hello_builder())
            # Replay retained state so a late joiner renders immediately (§5).
            # Stream topics retain their newest frame; event topics retain a
            # bounded backlog, so a log panel opens with recent history rather
            # than a single orphaned line.
            for backlog in [list(b) for b in self._retained.values()]:
                for frame in backlog:
                    await ws.send(frame)
            await self._read_commands(ws)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(client)
            pump.cancel()
            log.info("client disconnected; %d remain", len(self._clients))

    async def _read_commands(self, ws) -> None:
        """Client→server frames are text JSON (§7). Binary from a client is a bug."""
        async for message in ws:
            if isinstance(message, bytes):
                log.warning("ignoring binary frame from client (%d bytes)", len(message))
                continue
            try:
                command = json.loads(message)
            except json.JSONDecodeError:
                log.warning("ignoring malformed command: %.120s", message)
                continue
            if not isinstance(command, dict) or "cmd" not in command:
                log.warning("ignoring command without 'cmd': %.120s", message)
                continue
            self._on_command(command)

    async def _pump(self, client: _BusClient) -> None:
        """Drain one client's queues: events first, then one frame per stream topic."""
        try:
            while True:
                if not client.has_work():
                    await client.wake.wait()
                client.wake.clear()

                if client.overflow:
                    log.error("event buffer overflow; disconnecting slow client")
                    await client.ws.close(1009, "event buffer overflow")
                    return

                while client.events:
                    await client.ws.send(client.events.popleft())

                while client.stream_order:
                    topic = client.stream_order.popleft()
                    frame = client.stream_pending.pop(topic, None)
                    if frame is not None:
                        await client.ws.send(frame)
        except (ConnectionClosed, asyncio.CancelledError):
            pass
