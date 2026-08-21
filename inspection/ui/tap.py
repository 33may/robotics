#!/usr/bin/env python3
"""Watch what the loop is actually publishing. Read-only, commands nothing.

A second pair of eyes on the bus, independent of the window: it answers
"is the backend publishing, or is the UI not drawing it?" — the one question a
screenshot cannot. Run it in another terminal while a run is going:

    p inspection/ui/tap.py                        # rates per topic, once a second
    p inspection/ui/tap.py --topic=scene/poses    # only that topic
    p inspection/ui/tap.py --show                 # ...and decode what it carries

`--show` answers the follow-up question a rate cannot: a pose stream running at
30/s still leaves the 3D frozen if every frame carries the SAME configuration.
It prints the moving node's position, so a stalled reader (publishing a cached
joint value) looks different from a stalled renderer (positions changing on the
wire, nothing moving on screen).

During a move you should see `scene/poses` at ~30/s and `camera/wrist` at ~10/s.
A phase change prints its own line, so the rates can be read against the phase
they belong to.

Never sends a command, so it cannot influence the run — connecting is safe at
any moment, including mid-motion.
"""

import asyncio
import time
from collections import Counter

import msgpack


def _last_node_position(payload):
    """Position of the last node in a pose frame, straight off the wire.

    The transforms arrive as a tagged ndarray (protocol §3.1): float32 [N,4,4]
    row-major. The last row of the visual chain is the end of the arm, which is
    the thing whose movement you would notice.
    """
    import numpy as np

    names = payload.get("names") or []
    nd = (payload.get("transforms") or {}).get("$nd")
    if not names or not nd:
        return None
    mats = np.frombuffer(nd["data"], dtype=nd["dtype"]).reshape(nd["shape"])
    return names[-1], np.round(mats[-1][:3, 3], 4)


async def _tap(url, topic_filter, show=False):
    import websockets

    counts = Counter()
    last = time.monotonic()
    sample = None
    async with websockets.connect(url, max_size=None) as ws:
        print(f"tapping {url} — ctrl-c to stop", flush=True)
        async for raw in ws:
            if not isinstance(raw, (bytes, bytearray)):
                continue
            envelope = msgpack.unpackb(raw, raw=False)
            topic = envelope.get("topic", "?")
            if topic_filter and topic != topic_filter:
                continue
            counts[topic] += 1

            if show and topic == "scene/poses":
                sample = _last_node_position(envelope.get("payload") or {})

            # The status strip tells us which phase the rates belong to.
            if topic == "run/status":
                payload = envelope.get("payload") or {}
                print(f"  phase -> {payload.get('phase')} "
                      f"target={payload.get('target')}", flush=True)

            now = time.monotonic()
            if now - last >= 1.0:
                if counts:
                    rates = "  ".join(f"{t} {n}/s" for t, n in sorted(counts.items()))
                    line = f"{time.strftime('%H:%M:%S')}  {rates}"
                    if sample:
                        line += f"   {sample[0]} at {sample[1].tolist()}"
                    print(line, flush=True)
                else:
                    print(f"{time.strftime('%H:%M:%S')}  (nothing)", flush=True)
                counts.clear()
                last = now


def tap(port: int = 8765, host: str = "127.0.0.1", topic: str = "",
        show: bool = False):
    """Print per-second message rates per topic, optionally decoding poses."""
    try:
        asyncio.run(_tap(f"ws://{host}:{port}", topic, show))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(tap)
