"""Supervisor: the command-driven state machine for loop v2.

Spec: inspection/2026-08-20-ui-driven-loop-design.md ("State machine &
commands", "Architecture"). One dispatcher thread (`run()`) owns every bit
of run state; named worker threads (plan/preview/exec) talk back only by
putting events on `self.events`. A generation counter makes "a stale button
can never move the arm" (design FR11) mechanical: every `view/request` bumps
`self.gen`, every worker event carries the gen it was spawned with, and the
dispatcher drops anything whose gen has gone stale.

Two narrow, deliberate exceptions to "only the dispatcher mutates run
state" (both safe because exactly one action worker is alive at a time):
the settle leg of `_exec_worker` calls `self.acc.add` and `self._recenter()`
directly, and `_reach` (the reachability cache, expensive to compute) is
refreshed by `_plan_worker`/`_exec_worker` rather than the dispatcher.

"Exactly one action worker alive at a time" is enforced, not assumed:
`plan_viewpoint`/OMPL cannot be cancelled mid-call, so a superseding
`view/request` that arrives while `phase == "planning"` never spawns a
second planner — it is deferred (`self._pending_request`, last one wins)
and only started once the in-flight planner's (now-stale) result comes
back. `_cancel_preview()` joins the outgoing preview thread (bounded,
1 s) before any new worker is spawned, and the preview loop itself only
ever touches `pub` — every `world.is_colliding()` check it needs is
precomputed once, at preview-worker start, before the loop runs — so
even an unjoined straggler can't race a planner or `_recenter()`'s
`world.set_object()` on the shared, non-thread-safe `RobotCell`.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import CloudAccumulator, object_in_base
from inspection.cell.world import DEFAULT_STEP, RobotCell
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.view.viewsphere import ViewSphere

log = logging.getLogger(__name__)

# Phases in which the arm is either moving or unrecoverable without operator
# action at the pendant — no new request may be accepted while in one of these.
_BUSY_PHASES = ("executing", "capturing", "fusing", "fault")


class Supervisor:
    def __init__(self, rig, pub, outdir, q_survey, world=None, ik=None,
                 r=0.35, seed=0):
        self.rig, self.pub = rig, pub
        self.outdir = Path(outdir); self.q_survey = np.asarray(q_survey, float)
        self.world = world if world is not None else RobotCell()
        self.ik = ik if ik is not None else UR5eIK()
        self.r, self.seed = r, seed
        self.events = queue.Queue()
        self.stop_event = getattr(rig, "stop_event", None) or threading.Event()
        rig.stop_event = self.stop_event
        self.pose_active = threading.Event()
        self.phase, self.target, self.gen = "idle", None, 0
        self.visited, self.blocked = set(), set()
        self.current, self.sphere = None, None
        self.acc = CloudAccumulator()
        self.captures = {}          # cell -> {"step", "dir"}
        self.survey_state = "available"
        self.turns, self.t0 = [], time.time()
        self._preview_cancel = None
        self._preview_thread = None  # joined before any new worker spawns
        self._path = None           # path of the current preview
        self._reach = {}            # cached sphere.reachability(), worker-refreshed
        self._exit_after = False    # run/exit arrived mid-executing/capturing/fusing
        self._last_cap_dir = None   # capture dir of the most recent capture()
        self._pending_request = None  # target deferred while a plan is in flight

    # ----------------------------------------------------------------- run
    def run(self):
        self._publish_all()
        self._set_phase("idle")
        while self.phase != "done":
            try:
                ev = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.handle(ev)
            except Exception:
                log.exception("dispatcher error on %s", ev)
        self._save()

    def request_shutdown(self):
        """Thread-safe, SIGINT-safe: stop the arm, ask the dispatcher to exit."""
        self.stop_event.set()
        self.events.put({"cmd": "run/exit"})

    # ------------------------------------------------------------- dispatch
    def handle(self, ev):
        if "cmd" in ev:
            self._handle_cmd(ev)
        elif "ev" in ev:
            if ev.get("gen") != self.gen:
                log.debug("dropping stale %s (gen %s != current %s)",
                         ev.get("ev"), ev.get("gen"), self.gen)
                return
            self._handle_worker_event(ev)
        else:
            log.warning("unrecognized item on the bus: %s", ev)

    def _handle_cmd(self, ev):
        cmd = ev.get("cmd")
        if cmd == "view/request":
            self._on_view_request(ev)
        elif cmd == "view/confirm":
            self._on_view_confirm(ev)
        elif cmd == "run/stop":
            self._on_run_stop()
        elif cmd == "run/exit":
            self._on_run_exit()
        else:
            log.warning("unknown command: %r", cmd)
            self.pub.log("warn", f"unknown command: {cmd!r}")

    def _handle_worker_event(self, ev):
        kind = ev.get("ev")
        if kind == "plan_done":
            self._on_plan_done(ev)
        elif kind == "phase":
            self._on_worker_phase(ev)
        elif kind == "exec_done":
            self._on_exec_done(ev)
        elif kind == "settle_done":
            self._on_settle_done(ev)
        else:
            log.warning("unknown worker event: %r", kind)

    # ------------------------------------------------------------ commands
    def _on_view_request(self, ev):
        target = self._normalize_target(ev.get("target"))
        if self.phase in _BUSY_PHASES:
            log.info("view/request %s dropped: busy (%s)", target, self.phase)
            self.pub.log("warn", f"view/request dropped: busy ({self.phase})")
            return
        if target == "survey":
            if self.survey_state == "visited":
                log.info("view/request survey dropped: already visited")
                self.pub.log("warn", "survey already visited")
                return
        else:
            if (self.sphere is None or self._reach.get(target) is None
                    or target in self.visited):
                log.info("view/request %s dropped: no sphere/unreachable/"
                         "visited", target)
                self.pub.log("warn", f"view/request {target} dropped")
                return
        if self.phase == "planning":
            # A planner is already running real OMPL against the one shared
            # RobotCell (can take seconds, uncancellable mid-call, C++) —
            # never start a second one concurrently. Defer: last request
            # before the in-flight plan_done wins (F1a).
            #
            # A duplicate of the target already queued to run next (or, if
            # none is queued, of the plan already in flight) is a no-op: the
            # UI's `pending` button stays clickable while planning is under
            # way, and re-clicking it must not throw away a completed plan
            # and pay for a second OMPL pass against the identical target.
            already = self._pending_request if self._pending_request is not None else self.target
            if target == already:
                log.info("request for %s already in flight — ignored", target)
                return
            self._pending_request = target
            log.info("view/request %s deferred: still planning %s",
                     target, self.target)
            self.pub.log("info", f"{target} queued — still planning "
                          f"{self.target}")
            return
        self._start_planning(target)

    def _start_planning(self, target):
        self._cancel_preview()
        self.gen += 1
        self._set_phase("planning", target)
        self._publish_views()
        threading.Thread(target=self._plan_worker, args=(self.gen, target),
                         daemon=True).start()

    def _on_view_confirm(self, ev):
        target = self._normalize_target(ev.get("target"))
        if self.phase != "previewing" or target != self.target:
            log.info("view/confirm %s dropped: phase=%s target=%s",
                     target, self.phase, self.target)
            self.pub.log("warn", f"view/confirm {target} dropped")
            return
        self._cancel_preview()
        self.stop_event.clear()
        self.pose_active.set()
        self._set_phase("executing", self.target)
        threading.Thread(target=self._exec_worker,
                         args=(self.gen, self.target, self._path),
                         daemon=True).start()

    def _on_run_stop(self):
        if self.phase == "executing":
            self.stop_event.set()
        else:
            log.info("run/stop no-op in phase %s", self.phase)

    def _on_run_exit(self):
        # exit during executing/capturing/fusing = stop first (if moving),
        # then shut down once the in-flight action reaches a terminal event
        # (F4) — never tear down mid-capture/mid-fuse.
        if self.phase in ("executing", "capturing", "fusing"):
            if self.phase == "executing":
                self.stop_event.set()
            self._exit_after = True
        else:
            self._shutdown()

    # --------------------------------------------------------- worker events
    def _on_plan_done(self, ev):
        target, path = ev["target"], ev["path"]
        if self._pending_request is not None:
            # Superseded while this plan was in flight (F1a): its result —
            # success or refusal — is moot. Start the deferred request now.
            pending, self._pending_request = self._pending_request, None
            log.info("plan_done for %s discarded: %s was queued", target,
                     pending)
            self._start_planning(pending)
            return
        if path is None:
            if target == "survey":
                self.survey_state = "available"
            else:
                self.blocked.add(target)
            self.pub.log("warn", f"plan refused for {target}: "
                          f"{ev.get('detail', '')}")
            self._set_phase("idle", None)
            self._publish_views()
            return
        self._path = path
        self._set_phase("previewing", target)
        self._publish_views()
        self._preview_cancel = threading.Event()
        self._preview_thread = threading.Thread(
            target=self._preview_worker, args=(path, self._preview_cancel),
            daemon=True)
        self._preview_thread.start()

    def _on_worker_phase(self, ev):
        phase = ev.get("phase")
        if phase not in ("capturing", "fusing"):
            log.warning("unexpected worker phase event: %r", phase)
            return
        self._set_phase(phase, self.target)

    def _on_exec_done(self, ev):
        self.pose_active.clear()
        target, outcome = ev["target"], ev["outcome"]
        if outcome == "stopped":
            self._record_turn(target, "software stop — arm halted mid-move",
                              stopped=True)
            self.current = None
            self.blocked.clear()
            self.stop_event.clear()   # cleared only here, between actions
            self._set_phase("idle", None)
            self._publish_views()
            if self._exit_after:
                self._shutdown()
        elif outcome == "fault":
            self._record_turn(target,
                              f"executor halted/refused: {ev.get('detail', '')}")
            self._set_phase("fault", None)
            self.pub.log("error",
                         "executor halted/refused — check pendant; Exit only")
            self._publish_views()
            if self._exit_after:
                self._shutdown()
        # outcome == "done": the worker continues on its own into settling;
        # nothing to do here but wait for settle_done.

    def _on_settle_done(self, ev):
        target = ev["target"]
        if ev["ok"]:
            if target == "survey":
                self.survey_state = "visited"
            else:
                self.visited.add(target)
                self.captures[target] = {"step": self._next_cell_step(),
                                         "dir": self._last_cap_dir}
                self.current = target
            self.blocked.clear()
            self._record_turn(target, f"+{ev.get('npts', 0)} pts, "
                              f"fused {len(self.acc.points)}")
            self._publish_object()
        else:
            self.pub.log("error", ev.get("detail", ""))
            self._record_turn(target, ev.get("detail", ""))
        self._save()
        if self._exit_after:
            self._shutdown()
        else:
            self._set_phase("idle", None)
        self._publish_views()   # recover the retained topic either way (F3)

    # ------------------------------------------------------------- workers
    def _plan_worker(self, gen, target):
        try:
            if self.sphere is not None:
                self._reach = self.sphere.reachability(self.world, self.ik)
            q_now = self.rig.q()
            if target == "survey":
                path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                           self.ik.fk(self.q_survey), seed=self.seed)
                detail = "" if path else str(rep.get("reason", rep))
            else:
                path, prep, roll = self.sphere.plan_to_cell(
                    self.world, self.ik, q_now, *target, seed=self.seed)
                detail = f"tier {prep['tier']}" if path else "plan refused"
        except Exception as e:
            path, detail = None, f"planner error: {e}"
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": path, "detail": detail})

    def _preview_worker(self, path, cancel):
        try:
            dense = self.world.discretize(path, DEFAULT_STEP)
            # Precompute every collision flag up front (F1b): the world is
            # provably static for the life of one preview (nothing else can
            # touch it while this worker runs), so do the one is_colliding()
            # pass now and let the repeating loop below touch only `pub`.
            # That closes the race where a not-yet-joined dying preview
            # thread calls world.is_colliding() again just as a freshly
            # spawned planner starts mutating the same RobotCell.
            frames = [(q, self.world.is_colliding(q)) for q in dense]
            while not cancel.is_set():
                for q, colliding in frames:
                    if cancel.is_set():
                        return
                    self.pub.publish_pose(self.world, q, colliding=colliding)
                    time.sleep(1 / 30.0)
        except Exception:
            log.exception("preview worker failed")
            self.pub.log("error", "preview failed — see logs")

    def _exec_worker(self, gen, target, path):
        try:
            rep = self.rig.move(path)
        except Exception as e:
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "fault", "detail": str(e)})
            return
        if rep.get("stopped"):
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "stopped", "detail": ""})
            return
        self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                         "outcome": "done", "detail": ""})
        self.events.put({"ev": "phase", "gen": gen, "phase": "capturing"})
        try:
            step = self._next_cell_step() if target != "survey" else 0
            cap = self.rig.capture(step)
            view = object_in_base(cap["depth_raw"], self.rig.intr,
                                  self.rig.depth_scale, cap["T_base_cam"])
            self.events.put({"ev": "phase", "gen": gen, "phase": "fusing"})
            npts = len(view["points"]) if view["points"] is not None else 0
            if target == "survey" and view["centroid"] is None:
                self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                                 "ok": False, "npts": 0,
                                 "detail": "NO OBJECT above the table"})
                return
            if npts:
                self.acc.add(view["points"])
            self._recenter()
            if self.sphere is not None:
                self._reach = self.sphere.reachability(self.world, self.ik)
            self.pub.publish_capture(cap)
            self._last_cap_dir = cap.get("dir")
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": True, "npts": npts, "detail": ""})
        except Exception as e:
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": False, "npts": 0, "detail": f"capture failed: {e}"})

    # ------------------------------------------------------------- helpers
    def _recenter(self):
        center = self.acc.centroid
        self.sphere = ViewSphere(center, r=self.r)
        mn, mx = self.acc.aabb()
        dims = np.maximum(mx - mn + 0.04, 0.05)         # 2 cm margin each side
        mid = (mn + mx) / 2
        self.world.set_object("object", dims.tolist(),
                              [*mid.tolist(), 0.0, 0.0, 0.0], parent="base")

    def _save(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "run.json").write_text(json.dumps(
            {"q_survey": self.q_survey.tolist(), "r": self.r,
             "turns": self.turns}, indent=2) + "\n")

    def _next_cell_step(self):
        """1-indexed ordinal for the next non-survey capture: counts prior
        *cell* turns only, so the survey's own boot turn doesn't shift
        every cell's pose id/display step by one."""
        return sum(1 for t in self.turns if t["target"] != "survey") + 1

    def _record_turn(self, target, result, stopped=False):
        self.turns.append({
            "step": len(self.turns) + 1,
            "target": self._target_json(target),
            "t": time.time() - self.t0,
            "result": result,
            "stopped": stopped,
        })

    def _set_phase(self, phase, target=None):
        self.phase, self.target = phase, target
        total = sum(1 for r in self._reach.values() if r is not None)
        self.pub.status(phase=phase, target=self._target_json(target),
                        visited=len(self.visited), total=total)

    def _publish_all(self):
        self.pub.publish_world(self.world)
        self._publish_views()

    def _publish_views(self):
        survey_state = self._survey_display_state()
        if self.sphere is None:
            self.pub.publish_survey_only(survey_state)
            return
        cell_target = self.target if isinstance(self.target, tuple) else None
        pending = cell_target if self.phase == "planning" else None
        previewing = cell_target if self.phase == "previewing" else None
        self.pub.publish_views(
            self.sphere, self._reach, visited=self.visited, blocked=self.blocked,
            current=self.current, captures=self.captures, pending=pending,
            previewing=previewing, survey=survey_state)

    def _survey_display_state(self):
        """`self.survey_state` only ever stores "available"/"visited" (F3) —
        "pending"/"previewing" are transient and derived here from phase +
        target, the same way cell pending/previewing already are, so they
        can never be left stuck after a stop/fault/settle-fail."""
        if self.target == "survey":
            if self.phase == "planning":
                return "pending"
            if self.phase == "previewing":
                return "previewing"
        return self.survey_state

    def _publish_object(self):
        if self.acc.centroid is None:
            return
        mn, mx = self.acc.aabb()
        dims = np.maximum(mx - mn + 0.04, 0.05)
        mid = (mn + mx) / 2
        self.pub.publish_object(self.acc.points, dims.tolist(), mid.tolist())

    def _cancel_preview(self):
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        if self._preview_thread is not None:
            # F1b: block until the outgoing preview thread has actually
            # exited before any new worker (planner or executor) touches
            # the shared RobotCell. Bounded — the loop checks `cancel`
            # every ~33 ms at worst, so this should return almost at once.
            self._preview_thread.join(timeout=1.0)
        self._preview_cancel = None
        self._preview_thread = None

    def _shutdown(self):
        self._cancel_preview()
        self.pose_active.clear()
        self._set_phase("done")

    @staticmethod
    def _normalize_target(raw):
        return "survey" if raw == "survey" else tuple(raw)

    @staticmethod
    def _target_json(target):
        return list(target) if isinstance(target, tuple) else target
