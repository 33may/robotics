#!/usr/bin/env python3
"""Watch what the loop is actually publishing. Read-only, commands nothing.

A second pair of eyes on the bus, independent of the window: it answers
"is the backend publishing, or is the UI not drawing it?" — the one question a
screenshot cannot. Run it in another terminal while a run is going:

    p inspection/ui/tap.py                        # rates per topic, once a second
    p inspection/ui/tap.py --topic=scene/poses    # only that topic
    p inspection/ui/tap.py --show                 # ...and how much the pose MOVED
    p inspection/ui/tap.py --show --rtde          # ...against the arm's own joints

`--show` answers the follow-up question a rate cannot: a pose stream running at
30/s still leaves the 3D frozen if every frame carries the SAME configuration.
It reports how far the published transforms moved during the last second, so a
stalled reader (republishing a cached joint value) reads `moved 0.0000` while a
stalled renderer reads a healthy number with nothing happening on screen.

`--rtde` adds the independent half of that comparison: its OWN read-only
`RTDEReceiveInterface`, polling `getActualQ()` at 30 Hz, printed as degrees
moved per second. It commands nothing — a second receive interface is a
listener, exactly like this tap is on the bus.

Read the two numbers together during a move; each pattern names one component:

    arm moved 12.3 deg   pose moved 0.0000   -> the run's rig.q() is stale
    arm moved 12.3 deg   pose moved 0.1500   -> backend fine, the UI isn't drawing
    arm moved  0.0 deg   pose moved 0.0000   -> RTDE itself is not updating

During a move you should see `scene/poses` at ~30/s and `camera/wrist` at ~10/s.
A phase change prints its own line, so the rates can be read against the phase
they belong to.

Never sends a command, so it cannot influence the run — connecting is safe at
any moment, including mid-motion.
"""

import asyncio
import threading
import time
from collections import Counter

import msgpack


def _transforms(payload):
    """Every node's 4x4, as one flat array, straight off the wire.

    The transforms arrive as a tagged ndarray (protocol §3.1): float32 [N,4,4]
    row-major. Comparing the WHOLE frame rather than one link's origin matters —
    a wrist roll barely translates the last link, so a single-node probe calls a
    real move "static".
    """
    import numpy as np

    nd = (payload.get("transforms") or {}).get("$nd")
    if not nd:
        return None
    return np.frombuffer(nd["data"], dtype=nd["dtype"]).reshape(nd["shape"]).ravel()


class _ArmProbe(threading.Thread):
    """Read-only `getActualQ()` poller on its own RTDE connection.

    Deliberately independent of the run's interface: if the run's reader is the
    thing that has gone stale, sharing it would hide exactly the bug we are
    looking for.
    """

    def __init__(self, ip, hz=30.0):
        super().__init__(daemon=True)
        self.ip, self.hz = ip, hz
        self._lock = threading.Lock()
        self._span = 0.0            # max |dq| seen since the last read, degrees
        self._stale_s = 0.0         # seconds since getTimestamp() last advanced
        self.error = None

    def run(self):
        import numpy as np

        try:
            from rtde_receive import RTDEReceiveInterface
            recv = RTDEReceiveInterface(self.ip)
        except Exception as e:      # no robot, wrong IP, too many clients
            self.error = str(e)
            return
        last, ts_last, ts_wall = None, None, time.monotonic()
        while True:
            try:
                q = np.asarray(recv.getActualQ(), dtype=float)
                ts = recv.getTimestamp()
            except Exception as e:
                self.error = str(e)
                return
            # The probe must not lie the way the bug lies: this very thread's
            # recv died silently during the 2108-b diagnosis and kept printing
            # "arm moved 0.00" through a real move. A frozen controller
            # timestamp is the tell — surface it instead of the cached joints.
            now = time.monotonic()
            if ts != ts_last:
                ts_last, ts_wall = ts, now
            with self._lock:
                self._stale_s = now - ts_wall
            if last is not None:
                d = float(np.degrees(np.abs(q - last).max()))
                with self._lock:
                    self._span += d
            last = q
            time.sleep(1.0 / self.hz)

    def drain(self):
        """`(degrees travelled since the previous call, staleness seconds)`."""
        with self._lock:
            span, self._span = self._span, 0.0
            return span, self._stale_s


async def _tap(url, topic_filter, show=False, probe=None):
    import numpy as np
    import websockets

    counts = Counter()
    last = time.monotonic()
    prev_pose, pose_span = None, 0.0
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
                current = _transforms(envelope.get("payload") or {})
                if current is not None:
                    if prev_pose is not None and len(prev_pose) == len(current):
                        pose_span += float(np.abs(current - prev_pose).max())
                    prev_pose = current

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
                    if show:
                        line += f"   pose moved {pose_span:.4f}"
                    if probe is not None:
                        if probe.error is not None:
                            line += f"   arm probe failed: {probe.error}"
                        else:
                            span, stale = probe.drain()
                            line += (f"   arm probe STALE {stale:.1f}s — "
                                     f"its own recv is dead, ignore its degrees"
                                     if stale > 0.5
                                     else f"   arm moved {span:.2f} deg")
                    print(line, flush=True)
                else:
                    print(f"{time.strftime('%H:%M:%S')}  (nothing)", flush=True)
                counts.clear()
                pose_span = 0.0
                last = now


def tap(port: int = 8765, host: str = "127.0.0.1", topic: str = "",
        show: bool = False, rtde: str = ""):
    """Print per-second message rates per topic, optionally decoding poses.

    `rtde` is a robot IP (pass `--rtde` for the default cell arm). It opens a
    second, read-only receive interface — see the module docstring for how to
    read its number against the pose number.
    """
    probe = None
    if rtde:
        ip = "192.168.2.50" if rtde is True else str(rtde)
        probe = _ArmProbe(ip)
        probe.start()
        print(f"arm probe: read-only RTDE on {ip}", flush=True)
    try:
        asyncio.run(_tap(f"ws://{host}:{port}", topic, show, probe))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(tap)
