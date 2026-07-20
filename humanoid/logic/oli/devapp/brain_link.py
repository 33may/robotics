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

import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

from ..comm.client import BrainComm, CommClosedError
from ..contracts import Mode
from ..reason.teleoperation.joystick import JoystickAdapter
from ..runtime import Orchestrator
from .state import AppState

# Planner/controller knobs live in `reason/nav/factory.py` (`build_nav`) — shared by every
# brain host so dev_app and `brain_main --service` drive the IDENTICAL tuned Nav.
_REPLAN_DT = 0.25            # seconds between (local) re-plans while a goal holds


class _GtToMap:
    """GT(world) → map-frame SE(2) for the DISPLAY ghost when the demo runs in the
    baked M frame (bake-time alignment, 17-07). Pure data + math so the brain suite
    can pin it; loaded from the bake's registration_gt.json (an EVAL artifact —
    nothing in the control loop reads it). pose_W = R·pose_M + t  ⇒  inverse here."""

    def __init__(self, R, t, theta: float) -> None:
        self._r = R          # 2×2 row-major list
        self._t = t
        self._theta = theta

    @classmethod
    def load(cls, path) -> "Optional[_GtToMap]":
        try:
            d = json.loads(Path(path).read_text())
            return cls(d["R"], d["t"], float(d["theta_rad"]))
        except Exception:  # noqa: BLE001 — missing/bad file = ghost simply off
            return None

    def to_map(self, pose):
        from ..reason.localization import RobotPose
        dx, dy = pose.x - self._t[0], pose.y - self._t[1]
        r = self._r
        return RobotPose(     # R.T @ (p - t), yaw - theta   (R is orthonormal)
            stamp_ns=pose.stamp_ns,
            x=r[0][0] * dx + r[1][0] * dy,
            y=r[0][1] * dx + r[1][1] * dy,
            yaw=pose.yaw - self._theta,
        )


def _trace_line(out, gt, diag) -> dict:
    """Flatten one localization step into a JSONL record (drift-diagnosis trace, 17-07).

    Pure so the brain suite can pin it. Copies only serializable diag fields — the frame
    (`rgb`) and raw observations stay out; `n_obs` keeps the feature-count signal.
    """
    est = out.pose
    diag = diag or {}
    obs = diag.get("observations")
    err = (math.hypot(est.x - gt.x, est.y - gt.y)
           if est is not None and gt is not None else None)
    return {
        "wall_t": time.time(),
        "stamp_ns": out.stamp_ns,
        "status": out.status.name,
        "est": {"x": est.x, "y": est.y, "yaw": est.yaw} if est is not None else None,
        "gt": {"x": gt.x, "y": gt.y, "yaw": gt.yaw} if gt is not None else None,
        "err": err,
        "lc_status": diag.get("lc_status"),
        "pgo_status": diag.get("pgo_status"),
        "lc_events": diag.get("lc_events"),
        "lc_good_landmarks": diag.get("lc_good_landmarks"),
        "reloc_ok": diag.get("reloc_ok"),
        "reloc_fail": diag.get("reloc_fail"),
        "n_obs": len(obs) if obs is not None else None,
    }


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
        localizer: Optional[str] = None,
        loc_map: Optional[str] = None,
        camera_socket: Optional[str] = None,
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
        self._gt_localizer = None
        if debug_pose:
            from ..comm.debug_pose import DebugPoseClient
            from ..reason.localization import DebugPoseLocalizer
            self._pose_client = DebugPoseClient(debug_pose)
            self._localizer = DebugPoseLocalizer(self._pose_client)
        # `--localizer <name>` = the slam-demo-loop D7 flip inside the dev app's brain: attach
        # the in-brain LocalizationHost (same seam as brain_main --localizer) and steer Nav on
        # the candidate's ESTIMATE. GT (debug_pose) demotes to the D8 display ghost + the
        # known-start hint source (fired in `_run` when the first GT sample lands).
        self._loc_host = None
        self._frame_reader = None
        self._loc_map = loc_map
        self._hint_sent = True
        self._start_pose = None                  # M-frame dock hint (start_pose.json), if baked
        self._gt_to_map = None                   # W→M display converter for the ghost, if baked
        self._trace_f = None                     # drift-diagnosis JSONL (one line per module step)
        self._trace_last_stamp = -1
        if localizer:
            if not (debug_pose and loc_map and camera_socket):
                raise ValueError("localizer requires debug_pose (hint+ghost), loc_map and "
                                 "camera_socket")
            from ..comm.camera_stream import CameraStreamReader
            from ..reason.localization import (
                HostLocalizer, LocalizationHost, load_realization)
            load_realization(localizer)          # fail FAST on unknown name / broken import
            # M-frame demo mode (bake-time alignment, 17-07): a start_pose.json beside
            # the localizer map = every artifact is baked in the cuVSLAM frame. Then the
            # module runs registration-less (emits raw M poses), the known-start hint is
            # the FILE (the dock — zero GT in the loop), and the GT ghost converts W→M
            # for display via the bake's eval registration.
            sp = (Path(loc_map).parent / "start_pose.json"
                  if not (Path(loc_map) / "start_pose.json").exists()
                  else Path(loc_map) / "start_pose.json")
            self._start_pose = None
            self._gt_to_map = None
            overrides = None
            if sp.exists():
                d = json.loads(sp.read_text())
                self._start_pose = (float(d["x"]), float(d["y"]), float(d["yaw"]))
                overrides = {"localization": {"registration_file": None}}
                self._gt_to_map = _GtToMap.load(Path(loc_map) / "registration_gt.json")
                print(f"[devapp-brain] M-frame mode: hint from {sp} "
                      f"(dock {self._start_pose}), module registration OFF", flush=True)
            self._frame_reader = CameraStreamReader(camera_socket)
            print(f"[devapp-brain] localizer (Nav steers on est): {localizer} — "
                  f"connecting frames at {camera_socket}...", flush=True)
            self._frame_reader.connect()
            self._loc_host = LocalizationHost(
                lambda: load_realization(localizer, overrides), self._frame_reader)
            self._loc_host.start()
            trace_path = time.strftime("/tmp/oli-loc-trace-%Y%m%d-%H%M%S.jsonl")
            self._trace_f = open(trace_path, "w", buffering=1)   # line-buffered: tail -f-able
            print(f"[devapp-brain] drift trace → {trace_path}", flush=True)
            self._gt_localizer = self._localizer     # GT → ghost only (D8)
            self._localizer = HostLocalizer(self._loc_host)   # est → map dot + Nav
            self._hint_sent = False
        # Nav layer (brain-side): consumes the UI-set GoalCoordinate, plans via its Planner on the
        # emitted Map (StaticMapping), publishes the path back to AppState for the map panel to
        # render. ArmedNav below gates who drives: disarmed = joystick, armed = Nav.
        self._nav = nav
        self._last_goal = None
        self._last_plan_t = 0.0
        if self._nav is None and map_dir and self._localizer is not None:
            from ..reason.mapping import StaticMapping
            from ..reason.nav import build_nav
            # The factory pre-divides the controller caps by speed_scale (GlideAction rescales
            # downstream, so the product stays at the tuned speed). Non-glide: no rescale.
            self._nav = build_nav(
                StaticMapping(map_dir), self._localizer,
                speed_scale=glide_scale if mode == "glide" else 1.0,
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
        if self._loc_host is not None:    # feed the localization host its obs/intent (per tick)
            self._loc_host.on_tick(obs, intent)
            out = self._loc_host.latest()
            diag = self._loc_host.diagnostics()
            self._state.set_loc_state(
                f"{self._loc_host.state}"
                + (f"/{out.status.name}" if out is not None else ""))
            self._state.set_loc_diag(diag)
            gt = (self._gt_localizer.estimate(obs)
                  if self._gt_localizer is not None else None)
            if gt is not None and self._gt_to_map is not None:
                gt = self._gt_to_map.to_map(gt)           # M-frame demo: ghost drawn in M
            if gt is not None:                            # D8 ghost: display-only GT oracle
                self._state.set_gt_pose(gt)
            # drift trace: one JSONL line per NEW module output (dedupe on stamp) — the
            # offline answer to "did LC fire during the drift stretch?" (diagnosis, 17-07)
            if (self._trace_f is not None and out is not None
                    and out.stamp_ns != self._trace_last_stamp):
                self._trace_last_stamp = out.stamp_ns
                try:
                    self._trace_f.write(json.dumps(_trace_line(out, gt, diag)) + "\n")
                except Exception:  # noqa: BLE001 — tracing must never darken the pose path
                    pass
            self._apply_loc_command()
        pose = self._localizer.estimate(obs) if self._localizer is not None else None
        if pose is not None:              # stream the debug/real pose onto the map dot
            self._state.set_pose(pose)
        if self._nav is not None:         # goal→nav→path shuttle + arm gate (pose may be None)
            self._update_nav(pose)
        self._telemetry(intent, joy, policy_out)

    def _apply_loc_command(self) -> None:
        """Consume one panel-issued lifecycle command ('rehint'/'stop') — UI writes the
        request into AppState, this brain thread applies it (the nav-goal pattern)."""
        cmd = self._state.pop_loc_command()
        if cmd is None or self._loc_host is None:
            return
        if cmd == "stop":
            self._loc_host.request_stop()
            print("[devapp-brain] localizer stop (panel)", flush=True)
        elif cmd == "rehint" and self._pose_client is not None:
            sample = self._pose_client.latest()
            if sample is None:
                print("[devapp-brain] re-hint requested but no GT sample yet", flush=True)
                return
            from ..reason.localization import LocalizationSetup, RobotPose
            stamp, x, y, yaw = sample
            src = "GT"
            if self._start_pose is not None:   # M-frame mode: re-hint = robot is AT the dock
                x, y, yaw = self._start_pose
                src = "dock"
            self._loc_host.request_start(LocalizationSetup(
                map_dir=self._loc_map,
                initial_pose=RobotPose(stamp_ns=stamp, x=x, y=y, yaw=yaw)))
            print(f"[devapp-brain] re-hint (panel, {src}): x={x:.2f} y={y:.2f} "
                  f"yaw={yaw:.2f}", flush=True)

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
                if not self._hint_sent and self._pose_client is not None:
                    sample = self._pose_client.latest()
                    if sample is not None:   # world is alive → send the known-start hint
                        from ..reason.localization import LocalizationSetup, RobotPose
                        stamp, x, y, yaw = sample
                        src = "GT known start"
                        if self._start_pose is not None:
                            # M-frame mode: coords from the bake's dock file — the GT
                            # sample only told us the world is up (timing, not position)
                            x, y, yaw = self._start_pose
                            src = "start_pose.json dock"
                        self._loc_host.request_start(LocalizationSetup(
                            map_dir=self._loc_map,
                            initial_pose=RobotPose(stamp_ns=stamp, x=x, y=y, yaw=yaw)))
                        print(f"[devapp-brain] localizer hint ({src}): "
                              f"x={x:.2f} y={y:.2f} yaw={yaw:.2f}", flush=True)
                        self._hint_sent = True
                if self._orch.step_once() is None:
                    time.sleep(0.0005)
        except CommClosedError as e:
            self.error = e
        finally:
            self._comm.close()
            _maybe_close(self._joystick)
            _maybe_close(self._pose_client)
            if self._loc_host is not None:
                self._loc_host.close()
            _maybe_close(self._frame_reader)
            if self._trace_f is not None:
                self._trace_f.close()

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
