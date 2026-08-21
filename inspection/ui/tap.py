#!/usr/bin/env python3
"""Watch what the loop is actually publishing. Read-only, commands nothing.

A second pair of eyes on the bus, independent of the window: it answers
"is the backend publishing, or is the UI not drawing it?" — the one question a
screenshot cannot. Run it in another terminal while a run is going:

    p inspection/ui/tap.py                 # rates per topic, once a second
    p inspection/ui/tap.py --topic=scene/poses    # only that topic

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


async def _tap(url, topic_filter):
    import websockets

    counts = Counter()
    last = time.monotonic()
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

            # The status strip tells us which phase the rates belong to.
            if topic == "run/status":
                payload = envelope.get("payload") or {}
                print(f"  phase -> {payload.get('phase')} "
                      f"target={payload.get('target')}", flush=True)

            now = time.monotonic()
            if now - last >= 1.0:
                if counts:
                    rates = "  ".join(f"{t} {n}/s" for t, n in sorted(counts.items()))
                    print(f"{time.strftime('%H:%M:%S')}  {rates}", flush=True)
                else:
                    print(f"{time.strftime('%H:%M:%S')}  (nothing)", flush=True)
                counts.clear()
                last = now


def tap(port: int = 8765, host: str = "127.0.0.1", topic: str = ""):
    """Print per-second message rates per topic."""
    try:
        asyncio.run(_tap(f"ws://{host}:{port}", topic))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(tap)
