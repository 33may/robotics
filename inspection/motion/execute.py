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

import time

import numpy as np

ROBOT_IP = "192.168.2.50"

# Conservative first-contact defaults (Anthonio's pendant cap is 50 deg/s;
# we start far below it). Blends off by decision — see AGENTS.md.
SPEED_RAD_S = 0.25      # ~14 deg/s
ACC_RAD_S2 = 0.5
SPEED_SLIDER = 0.25

START_TOL_RAD = 0.02    # current q must match path[0] within ~1.1 deg

ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


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

    r = RTDEReceiveInterface(ip)
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

    def __init__(self, ip: str = ROBOT_IP, speed: float = SPEED_RAD_S,
                 acc: float = ACC_RAD_S2, slider: float = SPEED_SLIDER):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        from rtde_io import RTDEIOInterface

        pf = preflight(ip)
        if not pf["go"]:
            raise RuntimeError(f"preflight NO-GO: {pf}")
        self.recv = RTDEReceiveInterface(ip)
        self.ctrl = RTDEControlInterface(ip)   # raises unless Remote mode
        RTDEIOInterface(ip).setSpeedSlider(slider)
        self.speed, self.acc = speed, acc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.stop()
        self.close()

    def _assert_fresh(self) -> None:
        """Raise unless the receive stream is provably alive RIGHT NOW.

        ur_rtde's receive thread can die silently (upstream #307): every
        getter then returns cached state forever — joints AND safety mode —
        with `isConnected()` still true. Run 2108-ui stamped six captures
        with bit-identical joints that way while the arm visibly moved, and
        planned paths from a pose the arm was no longer at.

        `getTimestamp()` is controller-side time, advancing with every 2 ms
        packet, so "alive" means: the timestamp ADVANCES within the poll
        window. A single pair compared once is not enough — two immediate
        reads legitimately land on the same packet.

        Deliberately no auto-reconnect: a stopped run is safe, a silently
        recovered one hides that every read since the freeze was a lie.
        """
        t0 = self.recv.getTimestamp()
        deadline = time.monotonic() + self.stale_window
        while time.monotonic() < deadline:
            if self.recv.getTimestamp() != t0:
                return
            time.sleep(0.002)
        raise RuntimeError(
            f"RTDE receive stale — pose/safety state frozen at controller "
            f"t={t0:.3f}s (no packet for {self.stale_window:.2f}s). Joint "
            f"and safety reads are cached lies from here on; aborting. "
            f"Remedy: restart the run (or recv.reconnect() by hand).")

    def q(self) -> np.ndarray:
        self._assert_fresh()
        return np.array(self.recv.getActualQ())

    def _safety_ok(self) -> bool:
        # Raises (rather than returning False) on a stale stream: execute()'s
        # except path calls stop() on the way out, and the error names the
        # actual failure instead of a generic "safety mode changed".
        self._assert_fresh()
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
