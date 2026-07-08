"""brain_link.py — run the deployment-invariant brain inside the dev app process.

Builds the same Orchestrator that `brain_main.py` runs (BrainComm client → World, Teleop
reason, PolicyRunner action, joystick source) and drives it on a DAEMON thread so the UI
thread stays responsive. The Orchestrator's `recorder` hook publishes the latest contracts
into AppState for panels to render. This is the "app IS the brain" seam: same brain loop,
plus a window — the app attaches to whatever World already serves the socket (Isaac /
MuJoCo / real).

The comm / action / reason / joystick can be injected (defaults build the real ones), which
lets the attach loop be integration-tested against a fake World with no Isaac and no ONNX.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..comm.client import BrainComm, CommClosedError
from ..contracts import Mode
from ..reason.teleoperation.joystick import JoystickAdapter
from ..runtime import Orchestrator
from .state import AppState


class BrainLink:
    """Own the brain Orchestrator on a background thread, attached to a World socket."""

    def __init__(
        self,
        state: AppState,
        *,
        socket: str = "/tmp/oli-world.sock",
        mode: str = "walk",
        glide_scale: float = 1.0,
        joystick: str = "fixed",
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        joy_host: str = "127.0.0.1",
        joy_port: int = 9001,
        walk_after: Optional[float] = None,
        duration: float = 0.0,
        comm=None,
        action=None,
        reason=None,
        joystick_source=None,
    ) -> None:
        self._state = state
        self._walk_after = walk_after
        self._duration = duration
        self.error: Optional[BaseException] = None
        self._rec_n = 0   # for throttling the telemetry log line

        start_mode = Mode.STAND if (walk_after is not None or mode == "stand") else Mode.WALK
        self._reason = reason if reason is not None else _make_teleop(
            start_mode, glide=(mode == "glide"))
        self._comm = comm if comm is not None else BrainComm(socket_path=socket)
        self._action = action if action is not None else _make_action(mode, glide_scale)
        self._joystick = (
            joystick_source
            if joystick_source is not None
            else _make_joystick(joystick, vx, vy, wz, joy_host, joy_port)
        )
        self._orch = Orchestrator(
            self._comm, self._reason, self._action,
            joystick=self._joystick, recorder=self._record,
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _record(self, obs, policy_in, policy_out, joy=None) -> None:
        mode = getattr(self._reason, "mode", None)
        intent = getattr(policy_in, "intent", None)
        self._state.set_brain(
            obs, policy_out, mode.name if mode is not None else "—",
            intent=intent, joy=joy)
        self._telemetry(intent, joy, policy_out)

    def _telemetry(self, intent, joy, policy_out) -> None:
        """Every ~50 steps, print joystick→intent→glide to stdout — captured in the launcher
        log (/tmp/oli-launch.log) so a run can be inspected after the fact."""
        self._rec_n += 1
        if self._rec_n % 50 != 0:
            return
        axes = list(getattr(joy, "axes", None) or [])[:4]
        axes_s = "  ".join(f"{a:+.2f}" for a in axes) if axes else "none"
        iv = (f"vx {intent.v_x:+.2f} vy {intent.v_y:+.2f} wz {intent.w_z:+.2f}"
              if intent is not None else "none")
        ov = (f"vx {policy_out.v_x:+.2f} vy {policy_out.v_y:+.2f} wz {policy_out.w_z:+.2f}"
              if policy_out is not None and hasattr(policy_out, "v_x") else "—")
        print(f"[brain] joy axes [{axes_s}] -> intent {iv} -> glide {ov}", flush=True)

    def start(self) -> None:
        """Connect + run the brain loop on a daemon thread."""
        self._thread = threading.Thread(target=self._run, name="brain", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._comm.connect()
        except Exception as e:  # noqa: BLE001 — surface any connect failure to the UI
            self.error = e
            return
        t0 = time.monotonic()
        switched = False
        try:
            while not self._stop.is_set():
                if self._duration and (time.monotonic() - t0) > self._duration:
                    break
                if (self._walk_after is not None and not switched
                        and (time.monotonic() - t0) > self._walk_after):
                    self._reason.set_mode(Mode.WALK)
                    switched = True
                if self._orch.step_once() is None:
                    time.sleep(0.0005)
        except CommClosedError as e:
            self.error = e
        finally:
            self._comm.close()
            _maybe_close(self._joystick)

    def stop(self) -> None:
        """Signal the brain loop to exit and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


# ── default component builders (kept out of __init__ so tests can inject stubs) ──

class GlideJoystickAdapter(JoystickAdapter):
    """Glide-only remap: yaw on the right-stick HORIZONTAL axis so drag-right = turn right.

    The vendor virtual pad puts w_z on axis 3 (right-stick VERTICAL), so up→turn-left /
    down→turn-right — impossible to steer by dragging with a mouse. For the kinematic glide
    demo we take yaw from axis 2 (right-stick horizontal): dragging the stick RIGHT
    (axis2 > 0) turns Oli RIGHT, LEFT turns Oli LEFT (sign verified live with the operator).
    Forward (axis 1) and strafe (axis 0) are unchanged; walk/real keep the LimX mapping.
    """

    def axes_to_velocity(self, axes):
        v_x, v_y, _ = super().axes_to_velocity(axes)
        w_z = max(-self.max_vz, min(self.max_vz, float(axes[2])))
        return (v_x, v_y, w_z)


def _make_teleop(mode: Mode, glide: bool = False):
    from ..reason.teleoperation.joystick import Teleop
    adapter = GlideJoystickAdapter() if glide else None
    return Teleop(mode=mode, adapter=adapter)


def _make_action(mode: str, glide_scale: float = 1.0):
    """The Action for this mode: GlideAction for kinematic glide, else the walk PolicyRunner.

    A drop-in swap in the same Orchestrator loop (both expose `.step(policy_in)`); glide
    forwards the velocity Intent as a GlideCmd instead of running the walk ONNX (MAY-172).
    `glide_scale` multiplies stick→velocity for the glide demo (launcher default 3.5).
    """
    if mode == "glide":
        from ..glide import GlideAction
        return GlideAction(speed_scale=glide_scale)   # launcher passes 5.0 → full-stick 2.5 m/s (walk path untouched)
    from ..action.policy_runner import PolicyRunner
    return PolicyRunner()


def _make_joystick(kind: str, vx: float, vy: float, wz: float, host: str, port: int):
    from ..reason.teleoperation.joystick import FixedJoystick, SocketJoystickSource
    if kind == "socket":
        return SocketJoystickSource(host=host, port=port)
    return FixedJoystick(vx, vy, wz)


def _maybe_close(joystick) -> None:
    close = getattr(joystick, "close", None)
    if callable(close):
        close()
