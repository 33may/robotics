#!/usr/bin/env python3
"""Execution layer — validated paths onto the real UR5e (MAY-184).

The ONLY module that commands robot motion. Consumes waypoint paths from
plan_viewpoint() and executes them with sequential async moveJ (interruptible
— software stop via stop_event), blends off. The world validator remains the
safety authority: execute() re-checks the full path against the live world
before the first command and refuses on any mismatch — a stale path is a
refused path.

Requires the pendant in Remote Control mode (top-right icon in PolyScope).
State reads work in any mode; motion does not.

Usage (from repo root, robo env active):
    p inspection/motion/execute.py preflight          # read-only gate report
    p inspection/motion/execute.py bringup            # power on + brake release
    p inspection/motion/execute.py demo               # +-5 deg wrist_3 wiggle
"""

import logging
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

log = logging.getLogger(__name__)

ROBOT_IP = "192.168.2.50"

# Conservative first-contact defaults (Anthonio's pendant cap is 50 deg/s;
# we start far below it). Blends off by decision — see AGENTS.md.
SPEED_RAD_S = 0.25      # ~14 deg/s
ACC_RAD_S2 = 0.5
#: The controller's GLOBAL speed slider, and it is authoritative: `__init__`
#: writes it on every connect, so it overwrites whatever the operator set on
#: the PolyScope pendant (visibly — the pendant slider snaps to this value).
#: Set it here, not on the pendant.
#: Effective top joint speed = SPEED_RAD_S * SPEED_SLIDER, so 0.5 gives
#: 0.125 rad/s ~= 7.2 deg/s, still ~7x under the 50 deg/s pendant cap.
#: Raising it does NOT weaken collision safety: the path and its validation
#: are pure geometry (`world.path_valid` at DEFAULT_STEP) and are unaffected
#: by how fast the path is traversed. What it does cost is stopping distance
#: after a software stop, which is the reason to move it in steps.
SPEED_SLIDER = 0.5      # Anton 2026-08-24: 0.25 -> 0.5

START_TOL_RAD = 0.02    # current q must match path[0] within ~1.1 deg

ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1

# ── RTDE subscription: the smallest recipe that answers our four getters ────
#
# UR's RTDE Guide states the failure directly: "The client should read data
# periodically from the socket. The connection is closed by the robot
# controller when the receive buffer overflows." That is the documented
# origin of the `asio.misc:2 End of file` we kept seeing.
#
# ur_rtde subscribes to EVERY output variable when `variables` is left empty
# (~30 fields, rtde_receive_interface.cpp:120-150) and runs an e-Series at
# 500 Hz: ~1180 B x 500 Hz ~= 590 KB/s. A 128 KB socket buffer — the kernel
# default, and ur_rtde never calls setsockopt(SO_RCVBUF) — therefore
# overflows after ~0.22 s of the receive thread not being scheduled. That is
# the entire stall budget on a machine also running a planner, a websocket
# bus and a software-Vulkan Chromium.
#
# We call exactly four getters. Subscribing to only their variables gives
# ~67 B at 125 Hz ~= 8.4 KB/s, so the same buffer takes ~15 s to fill: a
# ~68x larger stall budget. 125 Hz is ample — the fastest consumer is the
# 20 Hz safety poll, and `stale_window` (100 ms) spans ~12 packets.
#
# Names are the RTDE wire names, verified against the getters in
# rtde_receive_interface.cpp (getTimestamp/getActualQ/getRobotMode/
# getSafetyMode look up exactly these keys). Getting one wrong is loud, not
# silent: `getStateData` misses and the getter raises "unable to get state
# data for specified key" — it does NOT return a default. That matters,
# because one of these is the safety mode.
RECV_VARIABLES = ["timestamp", "actual_q", "robot_mode", "safety_mode"]
RECV_FREQUENCY = 125.0


def _wait_async(running_fn, safety_fn, stop_event=None,
                poll_s=0.05, grace_s=0.3):
    """Poll an asynchronous move. Returns 'done' | 'stopped' | 'unsafe'.

    grace_s covers the window between issuing moveJ(async) and the
    controller reporting the op as running — without it a fast first poll
    sees 'not running' and declares done before the arm ever moves.
    """
    t0 = time.monotonic()
    seen_running = False
    while True:
        if stop_event is not None and stop_event.is_set():
            return "stopped"
        if not safety_fn():
            return "unsafe"
        running = running_fn()
        seen_running = seen_running or running
        if not running and (seen_running
                            or time.monotonic() - t0 > grace_s):
            return "done"
        time.sleep(poll_s)


def preflight(ip: str = ROBOT_IP) -> dict:
    """Read-only go/no-go report. Never commands anything."""
    from rtde_receive import RTDEReceiveInterface
    from dashboard_client import DashboardClient

    r = RTDEReceiveInterface(ip, RECV_FREQUENCY, RECV_VARIABLES)
    dash = DashboardClient(ip)
    dash.connect()
    rep = {
        "robot_mode": r.getRobotMode(),
        "robot_mode_ok": r.getRobotMode() == ROBOT_MODE_RUNNING,
        "safety_mode": r.getSafetyMode(),
        "safety_ok": r.getSafetyMode() == SAFETY_MODE_NORMAL,
        "remote_control": dash.isInRemoteControl(),
        "q_deg": list(np.round(np.degrees(r.getActualQ()), 2)),
    }
    dash.disconnect()
    # Disconnect explicitly: a receive interface left to garbage collection
    # tears its socket down late and prints "RTDEReceiveInterface boost system
    # Exception: (asio.misc:2) End of file" into the middle of a run's log.
    r.disconnect()
    rep["go"] = rep["robot_mode_ok"] and rep["safety_ok"] and rep["remote_control"]
    return rep


def bringup(ip: str = ROBOT_IP) -> dict:
    """Power on + brake release via the dashboard, then preflight.

    Equivalent of pendant Initialize -> ON -> START. Run only when Anton
    asks for it; the arm's joints click but do not travel.
    """
    from dashboard_client import DashboardClient

    dash = DashboardClient(ip)
    dash.connect()
    dash.powerOn()
    dash.brakeRelease()
    # brake release takes a few seconds; poll until RUNNING (max 20 s)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(ip)
    for _ in range(40):
        if r.getRobotMode() == ROBOT_MODE_RUNNING:
            break
        time.sleep(0.5)
    dash.disconnect()
    return preflight(ip)


def stream_autopsy(recv, ip: str = ROBOT_IP) -> str:
    """Why the receive stream stopped: kicked out, or reader thread died?

    Six scenarios reproducing the app's ingredients in isolation (see
    `rtde_soak.py`: preflight churn, RTDEControl, the RTDEIO leak, the
    Chromium window, the D405 pipeline, multi-threaded reads) all SURVIVED,
    while every real run dies. So the next real death has to explain
    itself, and one observation splits the fix in half:

      socket GONE / FIN_WAIT / CLOSE_WAIT -> the controller hung up on us.
          Cause is on the wire or in the controller (client limit, a
          malformed request, a protocol violation) — the pendant log names
          it, and the fix is ours.
      socket still ESTABLISHED, no data    -> ur_rtde's reader thread died
          under a live socket. That is a library bug (#235 calls the
          sporadic EOF a post-1.5.0 regression; we run 1.6.5) and no amount
          of app-side care fixes it — the answer is a version change or our
          own RTDE client.

    `isConnected()` is recorded but NOT trusted: it is the flag that lies
    during this failure. The thread dump names who was mid-read. Bounded
    and exception-proof: this runs on the failure path of a robot in
    motion, so it must never be the thing that raises.
    """
    lines = []
    try:
        lines.append(f"isConnected()={recv.isConnected()} (not trusted)")
    except Exception as e:
        lines.append(f"isConnected() raised: {e}")
    try:
        out = subprocess.run(
            ["ss", "-tnop", "state", "all"], capture_output=True, text=True,
            timeout=5).stdout
        socks = [ln.strip() for ln in out.splitlines() if ip in ln]
        lines.append("sockets to the controller:")
        lines.extend(f"    {s}" for s in socks or ["    (NONE — socket gone)"])
    except Exception as e:
        lines.append(f"ss failed: {e}")
    try:
        frames = sys._current_frames()
        lines.append("threads:")
        for th in threading.enumerate():
            f = frames.get(th.ident)
            where = "".join(traceback.format_stack(f)[-2:]).strip() if f else "?"
            lines.append(f"    {th.name}: {where.splitlines()[0].strip()}")
    except Exception as e:
        lines.append(f"thread dump failed: {e}")
    return "\n  ".join(lines)


class UR5eArm:
    """Motion gateway. One instance = one RTDEControl session.

    Context manager; always leaves via stop() on error. All motion passes
    through execute(), which refuses anything the world won't vouch for.
    """

    #: Window `_assert_fresh` polls for a controller-timestamp advance before
    #: declaring the receive stream dead. Packets arrive every 2 ms, so 100 ms
    #: is ~50 missed packets — unambiguous, and short enough that a safety
    #: poll failing stale still stops the arm within a fraction of a second.
    stale_window = 0.10

    #: Minimum seconds between reconnect ATTEMPTS. Callers are plural and
    #: fast — `PoseStreamer` swallows the RuntimeError and retries at 30 Hz,
    #: the safety poll runs at 20 Hz — so an unrecoverable stream would
    #: otherwise mean ~50 reconnects a second against the controller. That
    #: connect storm is a plausible way to get hung up on, i.e. the guard
    #: would be manufacturing the failure it exists to survive.
    reconnect_backoff = 2.0

    #: monotonic() of the last reconnect attempt. Class-level default so an
    #: instance built with __new__ (the no-hardware test path) still works.
    _last_reconnect = float("-inf")

    def __init__(self, ip: str = ROBOT_IP, speed: float = SPEED_RAD_S,
                 acc: float = ACC_RAD_S2, slider: float = SPEED_SLIDER):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        from rtde_io import RTDEIOInterface

        pf = preflight(ip)
        if not pf["go"]:
            raise RuntimeError(f"preflight NO-GO: {pf}")
        self.ip = ip                           # kept for the reconnect path
        self.recv = RTDEReceiveInterface(ip, RECV_FREQUENCY, RECV_VARIABLES)
        self.ctrl = RTDEControlInterface(ip)   # raises unless Remote mode
        RTDEIOInterface(ip).setSpeedSlider(slider)
        self.speed, self.acc = speed, acc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.stop()
        self.close()

    def _ts_advancing(self) -> bool:
        """Is the controller timestamp advancing within the poll window?

        A single pair compared once is not enough — two immediate reads
        legitimately land on the same 2 ms packet — so poll for an advance.

        The deadline is checked AFTER the read, never before. The earlier
        order (`while monotonic() < deadline:`) could return False having
        read the timestamp only twice: both landing on one 2 ms packet is
        legitimate, and if the process was then descheduled so that
        `sleep(0.002)` returned after the window closed, the loop exited
        without ever looking again. That declares a HEALTHY stream dead and
        tears down a working socket — and it only misfires on a loaded
        machine, which is why 23 isolated reproductions on an idle box all
        survived while every full run died. Now the last thing before
        returning False is always a fresh read.
        """
        t0 = self.recv.getTimestamp()
        deadline = time.monotonic() + self.stale_window
        while True:
            if self.recv.getTimestamp() != t0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.002)

    def _assert_fresh(self, diagnose: bool = True) -> None:
        """The receive stream must be provably alive RIGHT NOW — heal or raise.

        `diagnose=False` suppresses the socket autopsy. The safety poll runs
        it that way: during a move, the thread polling `_safety_ok()` every
        50 ms is the SAME thread that watches `stop_event`, and it holds
        `_recv_lock` throughout. Forking `ss` there would stall STOP — and
        every other recv reader — with the arm travelling. Diagnostics belong
        on the `q()` path, where the arm is parked and a pause is harmless.

        ur_rtde's receive thread dies silently (upstream #307, and #235
        reports the sporadic "End of file" as a regression after 1.5.0):
        every getter then returns cached state forever — joints AND safety
        mode — with `isConnected()` often still true. Run 2108-ui stamped
        six captures with bit-identical joints that way while the arm
        visibly moved, and planned paths from a pose the arm had left.

        `getTimestamp()` is controller-side time, advancing with every
        packet, so "alive" means: the timestamp ADVANCES within the window.

        On a dead stream: reconnect and re-verify — the operator asked for
        a stable stream, not an abort per hiccup (this stream died in all
        three hardware runs on 2026-08-21, once before any motion at all).

        KNOWN WART in `reconnect()` (rtde_receive_interface.cpp:282-320), not
        yet addressed: it hard-resets `frequency_` to 500 Hz on an e-Series,
        so a healed stream runs at 4x the bandwidth this module subscribed
        for. It keeps `variables_`, so the narrow recipe survives and the
        post-heal budget is ~3.8 s of stall rather than the fresh ~15 s —
        still ~17x better than the ~0.22 s that caused the original drops.
        It also spins in an unbounded `while (!getFirstStateReceived())`,
        which with `_recv_lock` held would wedge the app rather than fail.
        Rebuilding the interface instead would fix both; that change is
        unvalidated on this cell, so it waits for its own hardware test.
        The recovery is LOUD, never silent: cached data was a lie, so the
        log must say the stream died even when the heal works. Only a
        reconnect that still yields a frozen timestamp raises.
        """
        if self._ts_advancing():
            return
        t_frozen = self.recv.getTimestamp()
        log.warning("RTDE receive stream DEAD (timestamp frozen at %.3f) — "
                    "reconnecting...", t_frozen)
        # Autopsy BEFORE the reconnect: reconnect() replaces the very socket
        # state that says who killed the stream.
        since = time.monotonic() - self._last_reconnect
        if since < self.reconnect_backoff:
            # Another thread just tried and it did not take. Fail this caller
            # immediately rather than piling another connect onto the wire.
            raise RuntimeError(
                f"RTDE receive stale — frozen at controller t={t_frozen:.3f}s; "
                f"a reconnect {since:.1f}s ago did not take, backing off. "
                f"Joint and safety reads are cached lies from here on.")
        self._last_reconnect = time.monotonic()
        if diagnose:
            log.warning("stream autopsy:\n  %s", stream_autopsy(self.recv))
        try:
            healed = bool(self.recv.reconnect())
        except Exception:
            log.exception("RTDE reconnect raised")
            healed = False
        if healed and self._ts_advancing():
            log.warning("RTDE receive stream RECONNECTED — reads are live "
                        "again (values between the freeze at t=%.3f and now "
                        "were cached)", t_frozen)
            return
        raise RuntimeError(
            f"RTDE receive stale — pose/safety state frozen at controller "
            f"t={t_frozen:.3f}s and reconnect failed. Joint and safety reads "
            f"are cached lies from here on; aborting.")

    def q(self) -> np.ndarray:
        self._assert_fresh()
        return np.array(self.recv.getActualQ())

    def _safety_ok(self) -> bool:
        # Raises (rather than returning False) on a stale stream: execute()'s
        # except path calls stop() on the way out, and the error names the
        # actual failure instead of a generic "safety mode changed".
        # diagnose=False: this is the mid-move poll — see _assert_fresh.
        self._assert_fresh(diagnose=False)
        return (self.recv.getSafetyMode() == SAFETY_MODE_NORMAL
                and self.recv.getRobotMode() == ROBOT_MODE_RUNNING)

    def execute(self, path, world=None, step=None, stop_event=None) -> dict:
        """Run a validated waypoint path. Returns an execution report.

        Refuses (raises, no motion) when: start mismatch, world says the
        path is no longer valid, or safety is not NORMAL. Motion is
        asynchronous per waypoint so it can be interrupted: stop_event set
        -> stopJ deceleration, report {"stopped": True} (a normal outcome,
        not an error). Safety change mid-path still raises.
        """
        path = [np.asarray(q, dtype=float) for q in path]
        if len(path) < 2:
            raise ValueError("path needs >= 2 waypoints")

        q_now = self.q()
        d0 = np.abs(q_now - path[0]).max()
        if d0 > START_TOL_RAD:
            raise RuntimeError(
                f"start mismatch: {np.degrees(d0):.2f} deg from path[0] — replan")
        if not self._safety_ok():
            raise RuntimeError("safety/robot mode not NORMAL/RUNNING")
        if world is not None:
            kw = {"step": step} if step is not None else {}
            ok, bad = world.path_valid(path, **kw)
            if not ok:
                raise RuntimeError(f"world refuses path (segment {bad}) — replan")

        t0 = time.perf_counter()
        done, stopped = 0, False
        try:
            for q_goal in path[1:]:
                if not self._safety_ok():
                    raise RuntimeError("safety mode changed mid-path")
                self.ctrl.moveJ(list(q_goal), self.speed, self.acc, True)
                outcome = _wait_async(
                    lambda: (self.ctrl.getAsyncOperationProgressEx()
                             .isAsyncOperationRunning()),
                    self._safety_ok, stop_event)
                if outcome == "stopped":
                    self.ctrl.stopJ(2.0)
                    stopped = True
                    break
                if outcome == "unsafe":
                    raise RuntimeError("safety mode changed mid-path")
                done += 1
        except Exception:
            self.stop()
            raise
        err = np.degrees(np.abs(self.q() - path[-1]).max())
        return {"waypoints_done": done, "s": time.perf_counter() - t0,
                "final_err_deg": float(err), "stopped": stopped}

    def stop(self):
        try:
            self.ctrl.stopJ(2.0)
        except Exception:
            pass

    def close(self):
        self.ctrl.disconnect()
        self.recv.disconnect()


# ── CLI ─────────────────────────────────────────────────────────────────────

def _demo(ip: str = ROBOT_IP):
    """First-motion test: validated +-5 deg wrist_3 wiggle from the current q."""
    from inspection.cell.world import RobotCell

    world = RobotCell()
    with UR5eArm(ip) as arm:
        q0 = arm.q()
        q1 = q0.copy()
        q1[5] += np.radians(5.0)
        for leg in ([q0, q1], [q1, q0]):
            if not world.path_valid([leg[0], leg[1]])[0]:
                raise SystemExit("world refuses the wiggle — arm parked badly?")
        print(f"executing wiggle from q={np.round(np.degrees(q0),1)}")
        print("  out:", arm.execute([q0, q1], world=world))
        print("  back:", arm.execute([q1, q0], world=world))
    print("demo OK")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"preflight": preflight, "bringup": bringup, "demo": _demo})
