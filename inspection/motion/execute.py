#!/usr/bin/env python3
"""Execution layer — validated paths onto the real UR5e (MAY-184).

The ONLY module that commands robot motion. Consumes waypoint paths from
plan_viewpoint() and executes them with sequential blocking moveJ, blends
off. The world validator remains the safety authority: execute() re-checks
the full path against the live world before the first command and refuses
on any mismatch — a stale path is a refused path.

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

    def q(self) -> np.ndarray:
        return np.array(self.recv.getActualQ())

    def _safety_ok(self) -> bool:
        return (self.recv.getSafetyMode() == SAFETY_MODE_NORMAL
                and self.recv.getRobotMode() == ROBOT_MODE_RUNNING)

    def execute(self, path, world=None, step=None) -> dict:
        """Run a validated waypoint path. Returns an execution report.

        Refuses (raises, no motion) when: start mismatch, world says the
        path is no longer valid, or safety is not NORMAL. Between
        waypoints, any safety change -> stopJ + raise.
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
        done = 0
        try:
            for q_goal in path[1:]:
                if not self._safety_ok():
                    raise RuntimeError("safety mode changed mid-path")
                self.ctrl.moveJ(list(q_goal), self.speed, self.acc)
                done += 1
        except Exception:
            self.stop()
            raise
        err = np.degrees(np.abs(self.q() - path[-1]).max())
        return {"waypoints_done": done, "s": time.perf_counter() - t0,
                "final_err_deg": float(err)}

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
