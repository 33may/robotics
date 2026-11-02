"""Conformance tests for the bus against docs/bus-protocol.md.

These assert protocol behaviour, not implementation detail. If a test here
fails, either the server is wrong or the contract changed — and if the contract
changed, the document changes first.
"""

from __future__ import annotations

import asyncio

import msgpack
import numpy as np
import pytest
import websockets

from porthole import PortholeBus

PORT = 8799


async def recv_envelope(ws) -> dict:
    return msgpack.unpackb(await asyncio.wait_for(ws.recv(), timeout=3), raw=False)


@pytest.fixture()
def bus():
    b = PortholeBus(app="test", port=PORT).start()
    yield b
    b.stop()


async def test_hello_is_first_and_declares_topics(bus):
    """§2: $/hello precedes every other message and advertises declared topics."""
    bus.declare("camera/wrist", qos="stream", kind="image/jpeg")
    bus.declare("log/events", qos="event")

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        hello = await recv_envelope(ws)

    assert hello["topic"] == "$/hello"
    assert hello["payload"]["protocol"] == 1
    assert hello["payload"]["app"] == "test"
    declared = {t["topic"]: t["qos"] for t in hello["payload"]["topics"]}
    assert declared == {"camera/wrist": "stream", "log/events": "event"}


async def test_retained_state_replays_to_late_joiner(bus):
    """§5: a client connecting after the fact still sees current state."""
    bus.publish("state/loop", {"step": 7})

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)  # hello
        retained = await recv_envelope(ws)

    assert retained["topic"] == "state/loop"
    assert retained["payload"] == {"step": 7}
    assert retained["seq"] == 0, "retained frames keep their original seq"


async def test_event_backlog_replays_to_late_joiner(bus):
    """§5: an event topic keeps history, so a log panel opened late is not blank."""
    bus.declare("log/events", qos="event")
    for i in range(5):
        bus.publish("log/events", {"i": i})

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)  # hello
        replayed = [(await recv_envelope(ws))["payload"]["i"] for _ in range(5)]

    assert replayed == [0, 1, 2, 3, 4]


async def test_stream_backlog_is_only_the_newest(bus):
    """§5: streams retain one frame — replaying stale camera frames is misleading."""
    bus.declare("camera/wrist", qos="stream")
    for i in range(5):
        bus.publish("camera/wrist", {"i": i})

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        first = await recv_envelope(ws)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await recv_envelope(ws)

    assert first["payload"]["i"] == 4


async def test_seq_counts_per_topic_not_globally(bus):
    """§2: seq is per-topic. Verified on event topics, which never drop."""
    bus.declare("a/x", qos="event")
    bus.declare("b/y", qos="event")

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        for i in range(3):
            bus.publish("a/x", {"i": i})
            bus.publish("b/y", {"i": i})
        seqs: dict[str, list[int]] = {"a/x": [], "b/y": []}
        for _ in range(6):
            env = await recv_envelope(ws)
            seqs[env["topic"]].append(env["seq"])

    assert seqs["a/x"] == [0, 1, 2]
    assert seqs["b/y"] == [0, 1, 2]


async def test_stream_topics_may_drop_but_never_lose_the_newest(bus):
    """§6: stream QoS coalesces under load. The guarantee is latest-wins, and
    seq gaps are normal — a panel must tolerate them."""
    bus.declare("camera/wrist", qos="stream")

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        for i in range(50):
            bus.publish("camera/wrist", {"i": i})

        received: list[int] = []
        while True:
            try:
                received.append((await recv_envelope(ws))["seq"])
            except (TimeoutError, asyncio.TimeoutError):
                break
            if received[-1] == 49:
                break

    assert received == sorted(received), "frames never arrive out of order"
    assert received[-1] == 49, "the newest frame is always delivered"
    assert len(received) <= 50


async def test_event_topics_are_never_dropped(bus):
    """§6: every event message is delivered, in order."""
    bus.declare("log/events", qos="event")

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        for i in range(50):
            bus.publish("log/events", {"i": i})
        got = [(await recv_envelope(ws))["payload"]["i"] for _ in range(50)]

    assert got == list(range(50))


async def test_ndarray_roundtrip_is_c_contiguous_little_endian(bus):
    """§3.1: arrays cross the wire as raw buffers with dtype and shape."""
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    bus.publish("scene/cloud", {"points": bus.array_payload(points)})

    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        env = await recv_envelope(ws)

    tagged = env["payload"]["points"]["$nd"]
    assert tagged["dtype"] == "float32"
    assert tagged["shape"] == [4, 3]
    restored = np.frombuffer(tagged["data"], dtype="<f4").reshape(tagged["shape"])
    np.testing.assert_array_equal(restored, points)


async def test_unsupported_dtype_is_rejected_loudly(bus):
    with pytest.raises(TypeError, match="cannot cross the bus"):
        bus.array_payload(np.zeros(4, dtype=np.float16))


async def test_raw_pixels_rejected_as_image(bus):
    with pytest.raises(TypeError, match="encoded image bytes"):
        bus.image_payload(np.zeros((4, 4, 3), dtype=np.uint8))


async def test_commands_arrive_as_dicts(bus):
    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        await ws.send('{"cmd": "view/select", "cell": [3, 1]}')
        await asyncio.sleep(0.2)

    command = bus.poll()
    assert command == {"cmd": "view/select", "cell": [3, 1]}


async def test_malformed_commands_are_ignored_not_fatal(bus):
    async with websockets.connect(f"ws://127.0.0.1:{PORT}") as ws:
        await recv_envelope(ws)
        await ws.send("not json at all")
        await ws.send('{"no_cmd_field": 1}')
        await ws.send('{"cmd": "ok"}')
        await asyncio.sleep(0.2)

    assert bus.drain() == [{"cmd": "ok"}]
