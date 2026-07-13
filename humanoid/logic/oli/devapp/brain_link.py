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

# Robot/policy planning knobs (footprint + clearance) — constructor args of Planner, live-tunable.
_ROBOT_RADIUS_M = 0.30       # hard footprint: cells within this of a wall are impassable
_INFLATION_RADIUS_M = 1.0    # soft clearance reach (> robot radius) — path prefers this much gap
_CLEARANCE_WEIGHT = 3.0      # how hard to trade path length for clearance (0 = shortest path)
_HEURISTIC_WEIGHT = 1.2      # weighted A*: ~30× fewer nodes on open routes, near-lossless clearance
_HORIZON_M = 2.0             # local re-plan only re-solves this far ahead; the far tail is reused
_REPLAN_DT = 0.25            # seconds between (local) re-plans while a goal holds
_NAV_SPEED_MS = 1.0          # armed autonomy target forward speed [m/s] (after glide rescale)
_NAV_YAW_RS = 1.2            # armed autonomy target yaw rate [rad/s] (after glide rescale)


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
        debug_pose: Optional[str] = None,
        map_dir: Optional[str] = None,
        comm=None,
        action=None,
        reason=None,
        joystick_source=None,
        nav=None,
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
        # Optional debug-pose stream → Localizer → the map panel (via AppState.set_nav). This is
        # the DEBUG-mode coordinate source: Isaac's ground truth today, a real localizer later,
        # behind the same seam.
        self._pose_client = None
        self._localizer = None
        if debug_pose:
            from ..comm.debug_pose import DebugPoseClient
            from ..reason.localization import DebugPoseLocalizer
            self._pose_client = DebugPoseClient(debug_pose)
            self._localizer = DebugPoseLocalizer(self._pose_client)
        # Nav layer (brain-side): consumes the UI-set GoalCoordinate, plans via its Planner on the
        # emitted Map (StaticMapping), publishes the path back to AppState for the map panel to
        # render. ArmedNav below gates who drives: disarmed = joystick, armed = Nav.
        self._nav = nav
        self._last_goal = None
        self._last_plan_t = 0.0
        if self._nav is None and map_dir and self._localizer is not None:
            from ..reason.mapping import StaticMapping
            from ..reason.nav import Nav, Planner, PurePursuit
            # Pursuit emits real m/s; GlideAction rescales by glide_scale, so pre-divide the caps
            # to land Oli at ~_NAV_SPEED_MS when armed (glide demo). Non-glide: no rescale.
            s = glide_scale if mode == "glide" else 1.0
            self._nav = Nav(
                StaticMapping(map_dir), self._localizer,
                controller=PurePursuit(max_lin=_NAV_SPEED_MS / s, max_wz=_NAV_YAW_RS / s),
                planner=Planner(
                    robot_radius_m=_ROBOT_RADIUS_M,
                    inflation_radius_m=_INFLATION_RADIUS_M,
                    clearance_weight=_CLEARANCE_WEIGHT,
                    heuristic_weight=_HEURISTIC_WEIGHT,
                    horizon_m=_HORIZON_M,
                ),
            )
        # Arm gate: wrap operator Teleop + Nav so disarmed = joystick drives, armed = Nav drives.
        # (Only when we built the reason ourselves — an injected reason is used as-is.)
        if self._nav is not None and reason is None:
            from ..reason.nav import ArmedNav
            self._reason = ArmedNav(self._reason, self._nav)
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
        pose = self._localizer.estimate(obs) if self._localizer is not None else None
        if pose is not None:              # stream the debug/real pose onto the map dot
            self._state.set_pose(pose)
        if self._nav is not None:         # goal→nav→path shuttle + arm gate (pose may be None)
            self._update_nav(pose)
        self._telemetry(intent, joy, policy_out)

    def _update_nav(self, pose) -> None:
        """Move the UI-set goal INTO the Nav layer and publish its planned path back OUT.

        Nav owns the planning; this only shuttles goal(UI)→nav and path→AppState, and paces it: a
        new goal replans immediately (full plan), then it re-plans every `_REPLAN_DT` s — each a
        cheap LOCAL re-plan of the near horizon inside Nav — so A* never runs at control rate.
        """
        if self._nav is None:
            return
        armed = self._state.get_armed()
        set_armed = getattr(self._reason, "set_armed", None)
        if set_armed is not None:
            set_armed(armed)                     # gate the reason: teleop (disarmed) ↔ Nav (armed)
        goal = self._state.get_goal()
        changed = goal != self._last_goal
        if changed:
            self._last_goal = goal
            if goal is None:                     # goal cleared (right-click)
                self._nav.clear_goal()
                self._state.set_path(None)
                return
            self._nav.set_goal(goal)             # clears cached path → next plan is a full plan
        if goal is None:
            return
        if armed:
            # Nav drives → its to_policy_in already (re)planned this tick from its own localizer;
            # just publish the path it produced (no BrainLink pose needed).
            self._state.set_path(self._nav.path)
            return
        if pose is None:                         # disarmed preview needs a pose to plan from
            return
        now = time.monotonic()                   # pace the preview planning
        if changed or (now - self._last_plan_t) >= _REPLAN_DT:
            self._state.set_path(self._nav.plan(pose))
            self._last_plan_t = now

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
            _maybe_close(self._pose_client)

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
