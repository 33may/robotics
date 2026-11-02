#!/usr/bin/env python3
"""RTDE stream-death elimination matrix — freshness root-cause hunt.

The receive stream dies with "(asio.misc:2) End of file" in every hardware
run so far, usually near startup. EOF means the CONTROLLER closed our
subscriber on purpose; this script finds out which ingredient of the app's
startup provokes it, by layering them one at a time onto a monitored
RTDEReceiveInterface.

The monitor polls getTimestamp() at 50 Hz. A live stream advances it every
2 ms; the moment it stops advancing for 0.5 s the scenario reports DEATH
with the seconds since the last event marker (e.g. "3.2 s after ctrl
constructed") — that adjacency is the evidence.

READ-ONLY: never commands motion. The `ctrl` and `full_mix` scenarios do
construct RTDEControlInterface, which uploads its control script to the
controller (pendant must be in Remote Control) but moves nothing.

Run with the inspection app STOPPED — its own recv is a confound.

Usage (robo env active, from repo root):
    p inspection/motion/rtde_soak.py recv                # baseline, 10 min
    p inspection/motion/rtde_soak.py recv --minutes=3
    p inspection/motion/rtde_soak.py ctrl                # + RTDEControl, 5 repeats
    p inspection/motion/rtde_soak.py io_leaked           # + leaked RTDEIO (app's bug)
    p inspection/motion/rtde_soak.py preflight_churn     # + recv/dashboard churn
    p inspection/motion/rtde_soak.py full_mix            # exact UR5eArm.__init__ order
    p inspection/motion/rtde_soak.py recv --frequency=125   # H3 probe, any scenario
"""
import gc
import threading
import time

ROBOT_IP = "192.168.2.50"

DEAD_AFTER_S = 0.5      # timestamp must advance within this window
POLL_S = 0.02


class Monitor(threading.Thread):
    """Watches one RTDEReceiveInterface; reports the moment its clock freezes."""

    def __init__(self, recv, label="recv"):
        super().__init__(daemon=True)
        self.recv, self.label = recv, label
        self.stop_flag = threading.Event()
        self.died_at = None            # wall time of death, None = survived
        self.frozen_ts = None
        self._last_event = ("start", time.monotonic())
        self._lock = threading.Lock()

    def mark(self, name):
        """Record an event marker; a death is reported relative to this."""
        with self._lock:
            self._last_event = (name, time.monotonic())
        print(f"  [{time.strftime('%H:%M:%S')}] event: {name}")

    def run(self):
        last_ts = self.recv.getTimestamp()
        last_advance = time.monotonic()
        while not self.stop_flag.is_set():
            ts = self.recv.getTimestamp()
            now = time.monotonic()
            if ts != last_ts:
                last_ts, last_advance = ts, now
            elif now - last_advance > DEAD_AFTER_S:
                with self._lock:
                    ev, t_ev = self._last_event
                self.died_at, self.frozen_ts = now, ts
                print(f"  [{time.strftime('%H:%M:%S')}] *** {self.label} DEAD — "
                      f"timestamp frozen at {ts:.3f}, "
                      f"{now - t_ev:.1f} s after event '{ev}' ***")
                return
            time.sleep(POLL_S)

    def finish(self, note=""):
        self.stop_flag.set()
        self.join(timeout=2.0)
        verdict = "DIED" if self.died_at else "survived"
        print(f"  result[{self.label}]: {verdict} {note}")
        return self.died_at is None


def _open_recv(ip, frequency):
    from rtde_receive import RTDEReceiveInterface
    if frequency:
        return RTDEReceiveInterface(ip, frequency=frequency)
    return RTDEReceiveInterface(ip)


def recv(ip=ROBOT_IP, minutes=10.0, frequency=None):
    """E1 baseline: one receive interface, nothing else, N minutes."""
    r = _open_recv(ip, frequency)
    mon = Monitor(r)
    mon.start()
    mon.mark(f"recv connected (frequency={frequency or 'default'})")
    t_end = time.monotonic() + minutes * 60
    while time.monotonic() < t_end and mon.died_at is None:
        time.sleep(1.0)
    ok = mon.finish(f"after {minutes} min" if mon.died_at is None else "")
    r.disconnect()
    return ok


def ctrl(ip=ROBOT_IP, repeats=5, hold_s=30.0, frequency=None):
    """E2: does constructing RTDEControlInterface kill an existing recv?

    Each repeat: fresh recv, wait 5 s, construct ctrl, hold, disconnect ctrl,
    hold again. Death timestamps land next to the event that caused them.
    """
    from rtde_control import RTDEControlInterface
    deaths = 0
    for i in range(repeats):
        print(f"--- repeat {i + 1}/{repeats}")
        r = _open_recv(ip, frequency)
        mon = Monitor(r)
        mon.start()
        mon.mark("recv connected")
        time.sleep(5.0)
        if mon.died_at is None:
            mon.mark("constructing RTDEControlInterface")
            c = RTDEControlInterface(ip)
            mon.mark("ctrl constructed")
            _wait(mon, hold_s)
            mon.mark("disconnecting ctrl")
            c.disconnect()
            _wait(mon, hold_s)
        deaths += 0 if mon.finish() else 1
        r.disconnect()
    print(f"=== ctrl scenario: {deaths}/{repeats} repeats killed the recv")
    return deaths == 0


def io_leaked(ip=ROBOT_IP, hold_s=60.0, frequency=None):
    """E3: recv + RTDEIOInterface abandoned to GC — the app's exact leak."""
    from rtde_io import RTDEIOInterface
    r = _open_recv(ip, frequency)
    mon = Monitor(r)
    mon.start()
    mon.mark("recv connected")
    time.sleep(5.0)
    mon.mark("RTDEIOInterface created as temporary (leaked)")
    RTDEIOInterface(ip)          # deliberately not kept, not disconnected
    _wait(mon, 10.0)
    mon.mark("forcing gc.collect() — late socket teardown happens here")
    gc.collect()
    _wait(mon, hold_s)
    ok = mon.finish()
    r.disconnect()
    return ok


def preflight_churn(ip=ROBOT_IP, cycles=5, frequency=None):
    """E4: recv + repeated second-recv/dashboard connect-disconnect churn."""
    from dashboard_client import DashboardClient
    r = _open_recv(ip, frequency)
    mon = Monitor(r)
    mon.start()
    mon.mark("recv connected")
    time.sleep(5.0)
    for i in range(cycles):
        if mon.died_at is not None:
            break
        mon.mark(f"churn {i + 1}: second recv + dashboard connect")
        r2 = _open_recv(ip, frequency)
        d = DashboardClient(ip)
        d.connect()
        d.isInRemoteControl()
        time.sleep(2.0)
        mon.mark(f"churn {i + 1}: disconnecting both")
        d.disconnect()
        r2.disconnect()
        _wait(mon, 5.0)
    _wait(mon, 30.0)
    ok = mon.finish()
    r.disconnect()
    return ok


def full_mix(ip=ROBOT_IP, hold_s=120.0, frequency=None, repeats=5):
    """E5: the exact UR5eArm.__init__ sequence, monitored, N repeats."""
    from dashboard_client import DashboardClient
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface

    deaths = 0
    for i in range(repeats):
        print(f"--- repeat {i + 1}/{repeats}")
        # preflight, as the app does it
        r1 = _open_recv(ip, frequency)
        d = DashboardClient(ip)
        d.connect()
        d.isInRemoteControl()
        d.disconnect()
        r1.disconnect()

        r = _open_recv(ip, frequency)          # the long-lived recv
        mon = Monitor(r)
        mon.start()
        mon.mark("long-lived recv connected (after preflight churn)")
        mon.mark("constructing RTDEControlInterface")
        c = RTDEControlInterface(ip)
        mon.mark("ctrl constructed; leaking RTDEIOInterface")
        RTDEIOInterface(ip).setSpeedSlider(0.25)
        mon.mark("startup sequence complete — holding")
        _wait(mon, hold_s)
        deaths += 0 if mon.finish() else 1
        c.disconnect()
        r.disconnect()
        gc.collect()                           # flush the leaked IO between repeats
    print(f"=== full_mix: {deaths}/{repeats} repeats killed the recv")
    return deaths == 0


def ui_launch(ip=ROBOT_IP, hold_s=60.0, frequency=None, repeats=3):
    """E6: does launching the UI window (Chromium, software Vulkan) kill recv?

    Replays the app's exact child spawn: `inspection/ui/app.py serve`. The
    window will appear on screen each repeat and be closed again — it dials
    a bus that isn't there, which is irrelevant to the CPU-spike question.
    """
    import subprocess
    import sys
    from pathlib import Path
    ui_app = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    run_dir = Path("/tmp/soak-ui-run")
    run_dir.mkdir(exist_ok=True)
    deaths = 0
    for i in range(repeats):
        print(f"--- repeat {i + 1}/{repeats}")
        r = _open_recv(ip, frequency)
        mon = Monitor(r)
        mon.start()
        mon.mark("recv connected")
        time.sleep(10.0)
        mon.mark("spawning UI window child (Chromium/software Vulkan)")
        child = subprocess.Popen([sys.executable, str(ui_app), "serve",
                                  f"--run_dir={run_dir}", "--port=8768"])
        _wait(mon, hold_s)
        mon.mark("terminating window child")
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
        _wait(mon, 15.0)
        deaths += 0 if mon.finish() else 1
        r.disconnect()
    print(f"=== ui_launch: {deaths}/{repeats} repeats killed the recv")
    return deaths == 0


def camera(ip=ROBOT_IP, hold_s=60.0, frequency=None):
    """E7: recv + D405 pipeline + 10 Hz grab thread — the app's camera layer."""
    from inspection.perception.camera import grab_aligned, open_camera
    r = _open_recv(ip, frequency)
    mon = Monitor(r)
    mon.start()
    mon.mark("recv connected")
    time.sleep(10.0)
    mon.mark("opening D405 pipeline")
    pipe, _, align, _ = open_camera()
    mon.mark("camera open — grabbing at 10 Hz")
    stop = threading.Event()

    def grab_loop():
        while not stop.is_set():
            try:
                grab_aligned(pipe, align)
            except Exception as e:
                print(f"  [{time.strftime('%H:%M:%S')}] grab error: {e}")
            time.sleep(0.1)

    t = threading.Thread(target=grab_loop, daemon=True)
    t.start()
    _wait(mon, hold_s)
    stop.set()
    t.join(timeout=3)
    mon.mark("stopping camera pipeline")
    pipe.stop()
    _wait(mon, 10.0)
    ok = mon.finish()
    r.disconnect()
    return ok


def threads(ip=ROBOT_IP, hold_s=90.0, frequency=None, repeats=3, hz=30.0):
    """E8: the app's THREADING against one recv — the last structural gap.

    Every scenario above read the recv from a single thread. `RealRig` has
    several: PoseStreamer at 30 Hz, the supervisor's plan/settle reads, and
    execute()'s safety polls at 20 Hz — all funnelled through one lock, and
    each `q()` runs `_assert_fresh()`, which itself polls `getTimestamp()`.
    ur_rtde documents RTDEControl as not thread-safe and says nothing about
    RTDEReceive; if concurrent access corrupts its socket state, the
    controller answers with the clean FIN we keep seeing.

    Reads go through the same lock discipline RealRig installs, so a death
    here indicts the ACCESS PATTERN (rate/interleaving), not missing
    locking. Read-only: no motion is commanded.
    """
    from inspection.motion.execute import UR5eArm
    deaths = 0
    for i in range(repeats):
        print(f"--- repeat {i + 1}/{repeats}")
        arm = UR5eArm(ip)                      # recv + ctrl, as the app builds it
        lock = threading.Lock()
        mon = Monitor(arm.recv)
        mon.start()
        mon.mark("UR5eArm built (recv + ctrl)")
        stop = threading.Event()
        errors = []

        def reader(name, period, fn):
            def loop():
                while not stop.is_set():
                    try:
                        with lock:
                            fn()
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        return
                    time.sleep(period)
            return threading.Thread(target=loop, name=name, daemon=True)

        # PoseStreamer-equivalent (30 Hz q) + safety poll (20 Hz), as during a move
        workers = [reader("pose", 1.0 / hz, lambda: arm.recv.getActualQ()),
                   reader("safety", 0.05, lambda: (arm.recv.getSafetyMode(),
                                                   arm.recv.getRobotMode()))]
        for w in workers:
            w.start()
        mon.mark(f"{len(workers)} reader threads running ({hz} Hz pose, 20 Hz safety)")
        _wait(mon, hold_s)
        stop.set()
        for w in workers:
            w.join(timeout=2)
        if errors:
            print(f"  reader errors: {errors}")
        deaths += 0 if mon.finish() else 1
        arm.close()
        time.sleep(2.0)
    print(f"=== threads: {deaths}/{repeats} repeats killed the recv")
    return deaths == 0


def _wait(mon, seconds):
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end and mon.died_at is None:
        time.sleep(0.5)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"recv": recv, "ctrl": ctrl, "io_leaked": io_leaked,
               "preflight_churn": preflight_churn, "full_mix": full_mix,
               "ui_launch": ui_launch, "camera": camera, "threads": threads})
