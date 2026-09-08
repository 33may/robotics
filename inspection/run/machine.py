"""Supervisor: the command-driven state machine for loop v2.

Spec: inspection/docs/2026-08-20-ui-driven-loop-design.md ("State machine &
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

The third pose producer, `PoseStreamer`, is handed off structurally
rather than by timing: before the settle leg of `_exec_worker` touches
`_recenter()` it waits for the dispatcher's `_pose_ack` (proof that
`pose_active` has actually been cleared, not merely that `exec_done` was
queued) and then for `pose_quiesce()` (proof that no publish is still in
flight inside the publisher's cached geom_data). Both waits are bounded:
a missed handoff degrades to the old timing-based behaviour and logs,
rather than wedging a run with a robot in it.

RECORDING (task-8). Every byte this machine puts on disk goes through the
injected `RunWriter` — there is no second write path any more. The two
threads that touch it are ordered by the same queue that orders everything
else: the exec worker allocates the step and writes its capture BEFORE it
queues `settle_done`, and the dispatcher writes the fused half only when
that event comes back. Nothing writes concurrently with anything else.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import BOX_PCT, CloudAccumulator
from inspection.cell.world import DEFAULT_STEP, RobotCell
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.record.schema import ViewCandidate, ViewState
from inspection.run.settle import _plane_error, settle_capture
from inspection.view.grid import object_extent, radius_for_extent
from inspection.view.viewsphere import ViewSphere

log = logging.getLogger(__name__)

# Phases in which the arm is either moving or unrecoverable without operator
# action at the pendant — no new request may be accepted while in one of these.
_BUSY_PHASES = ("executing", "capturing", "fusing", "fault")

#: The id of the view method this machine addresses cells with, as declared in
#: the run's `view_methods` (`RunWriter.create`). Every step's `view.method`
#: carries it, and `validate_run` refuses a step whose method the run never
#: declared — so the two must be written from one constant.
VIEW_METHOD = "vs"

# `_plane_error` now lives in `run/settle.py` (the plane gate is part of the
# extracted settle pipeline); re-exported here so any external reference to
# `machine._plane_error` keeps working.


class Supervisor:
    def __init__(self, rig, pub, writer, q_survey, world=None, ik=None,
                 r=None, seed=0, segmenter=None):
        self.rig, self.pub = rig, pub
        #: The run's ONE writer (`record/writer.py`). The machine owns no
        #: other path to disk: capture bytes go into the dir `begin_step`
        #: hands out, every JSON record goes through these verbs.
        self.writer = writer
        #: Optional `run.segmenter.ObjectSegmenter`. None keeps the pure
        #: depth pipeline, which is what the fake rig and most tests want;
        #: the composition root injects a real one. Never constructed here —
        #: that would drag SAM 3 weights into every unit test.
        self.segmenter = segmenter
        self.outdir = Path(writer.dir); self.q_survey = np.asarray(q_survey, float)
        self.world = world if world is not None else RobotCell()
        # The probed tabletop height in the base frame (cell.yaml `frames.
        # table`), for `prompt_box`'s floor extrusion — the fix for the
        # top-anchored prompt (see `cell/geometry.py:prompt_box`). Reaching
        # into `_frame_to_base`/`_frames` is deliberate smallest-footprint:
        # promote to a `RobotCell` property when that file is next open.
        try:
            from inspection.cell.world import _frame_to_base
            self._floor_z = float(
                _frame_to_base(self.world._frames, "table").translation[2])
        except Exception:
            self._floor_z = None    # a world with no probed table: old behavior
        self.ik = ik if ik is not None else UR5eIK()
        self.r, self.seed = r, seed
        self.events = queue.Queue()
        self.stop_event = getattr(rig, "stop_event", None) or threading.Event()
        rig.stop_event = self.stop_event
        self.pose_active = threading.Event()
        #: Injected by the composition root (`app.py`/`mock.py`) with
        #: `PoseStreamer.quiesce`. The default is the honest answer for a
        #: Supervisor with no pose thread behind it: nothing to wait for.
        self.pose_quiesce = lambda timeout=1.0: True
        self.phase, self.target, self.gen = "idle", None, 0
        self.visited, self.blocked = set(), set()
        self.current, self.sphere = None, None
        self.acc = CloudAccumulator()
        self.captures = {}          # cell -> {"step", "dir"}
        self.survey_state = "available"
        self._preview_cancel = None
        self._preview_thread = None  # joined before any new worker spawns
        self._action_thread = None   # the one live plan/exec worker, for join_workers
        self._path = None           # path of the current preview
        self._reach = {}            # cached sphere.reachability(), worker-refreshed
        self._exit_after = False    # run/exit arrived mid-action, shut down at its end
        self._shutting_down = False  # set by request_shutdown (signal thread)
        #: `run/finish`: the operator declared the run DONE, not abandoned.
        #: The machine only remembers it — the app teardown reads it to close
        #: the record "completed" where a bare exit closes "aborted".
        self.finish_requested = False
        self._last_cap_dir = None   # capture dir of the most recent capture()
        self._pending_request = None  # target deferred while a plan is in flight
        self._pose_ack = threading.Event()  # dispatcher -> exec worker: pose_active clear
        #: Who asked for the view now being planned/executed — it rides the
        #: `view/request` command ("operator" from a button, "ai" from the
        #: orchestrator's mover) and lands in the step's `ViewState.decider`,
        #: which is the only place the record says who chose.
        self._decider = "operator"
        self._pending_decider = "operator"

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

    def request_shutdown(self):
        """Thread-safe, SIGINT-safe: stop the arm, ask the dispatcher to exit.

        `_shutting_down` is set *before* the command is queued and is the
        reason a `view/confirm` already sitting in the FIFO cannot undo this
        (I3): the queue is ordered, so that confirm is dispatched before
        `run/exit` ever is, and `_on_view_confirm` clears `stop_event` — which
        would start a motion out of a Ctrl-C. A plain bool is enough: one
        writer (this call, from the signal handler's thread), one reader (the
        dispatcher), and no read-modify-write.
        """
        self._shutting_down = True
        self.stop_event.set()
        self.events.put({"cmd": "run/exit"})

    def join_workers(self, timeout: float = 5.0) -> bool:
        """Bounded join of whatever plan/exec/preview thread is still alive.

        The design's "stop event -> join worker (bounded) -> stopJ -> save ->
        close hardware" step. Without it, `app.py`'s teardown can call
        `rig.close()` -> `recv.disconnect()` while a worker is inside
        `rig.q()` on the same RTDE interface. Returns False if anything was
        still running when the budget ran out (the caller logs; it must still
        close the hardware — a leaked interface is worse than a leaked thread).
        """
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        deadline = time.monotonic() + timeout
        clean = True
        for th in (self._action_thread, self._preview_thread):
            if th is None or not th.is_alive():
                continue
            th.join(max(0.0, deadline - time.monotonic()))
            if th.is_alive():
                clean = False
                log.warning("worker %s still alive after the join budget", th.name)
        return clean

    # ------------------------------------------------------------- dispatch
    def handle(self, ev):
        if "cmd" in ev:
            self._handle_cmd(ev)
        elif "ev" in ev:
            if ev.get("gen") != self.gen:
                log.debug("dropping stale %s (gen %s != current %s)",
                         ev.get("ev"), ev.get("gen"), self.gen)
                # The event's *content* is moot, but a deferred exit is not:
                # `plan_done`/`settle_done` mean "the worker that was holding
                # up the shutdown has finished", and nothing else will ever
                # arrive to release it. `exec_done` is deliberately not in
                # this list — outcome "done" is not the end of the worker,
                # its settle leg keeps running.
                if self._exit_after and ev.get("ev") in ("plan_done", "settle_done"):
                    log.info("stale %s released the deferred exit", ev.get("ev"))
                    self._shutdown()
                return
            self._handle_worker_event(ev)
        else:
            log.warning("unrecognized item on the bus: %s", ev)

    def _handle_cmd(self, ev):
        cmd = ev.get("cmd")
        if self._shutting_down and cmd in ("view/request", "view/confirm"):
            # I3: a Ctrl-C (or an Exit) already decided this run is over.
            # Anything that could start a motion — above all `view/confirm`,
            # which clears `stop_event` — is dead from here on.
            log.info("%s dropped: shutting down", cmd)
            self.pub.log("warn", f"{cmd} dropped: shutting down")
            return
        if cmd == "view/request":
            self._on_view_request(ev)
        elif cmd == "view/confirm":
            self._on_view_confirm(ev)
        elif cmd == "run/stop":
            self._on_run_stop()
        elif cmd == "run/exit":
            self._on_run_exit()
        elif cmd == "run/finish":
            self._on_run_finish()
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
        # Untrusted, like the target: anything that reaches this queue may say
        # who it is, and "operator" is the honest default for a bare command
        # (that is what the browser's buttons send).
        decider = str(ev.get("decider", "operator"))
        if target is None:
            log.warning("view/request with malformed target: %r", ev.get("target"))
            self.pub.log("warn", f"view/request ignored: bad target "
                          f"{ev.get('target')!r}")
            return
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
            self._pending_decider = decider
            log.info("view/request %s deferred: still planning %s",
                     target, self.target)
            self.pub.log("info", f"{target} queued — still planning "
                          f"{self.target}")
            return
        self._start_planning(target, decider)

    def _start_planning(self, target, decider="operator"):
        self._cancel_preview()
        self.gen += 1
        self._decider = decider
        self._set_phase("planning", target)
        self._publish_views()
        self._action_thread = threading.Thread(
            target=self._plan_worker, args=(self.gen, target),
            name=f"plan-{self._target_json(target)}", daemon=True)
        self._action_thread.start()

    def _on_view_confirm(self, ev):
        target = self._normalize_target(ev.get("target"))
        if target is None:
            log.warning("view/confirm with malformed target: %r", ev.get("target"))
            self.pub.log("warn", f"view/confirm ignored: bad target "
                          f"{ev.get('target')!r}")
            return
        if self.phase != "previewing" or target != self.target:
            log.info("view/confirm %s dropped: phase=%s target=%s",
                     target, self.phase, self.target)
            self.pub.log("warn", f"view/confirm {target} dropped")
            return
        self._cancel_preview()
        self.stop_event.clear()
        self._pose_ack.clear()      # armed here so only the dispatcher ever sets it
        self.pose_active.set()
        self._set_phase("executing", self.target)
        self._action_thread = threading.Thread(
            target=self._exec_worker,
            args=(self.gen, self.target, self._path),
            name=f"exec-{self._target_json(self.target)}", daemon=True)
        self._action_thread.start()

    def _on_run_stop(self):
        if self.phase == "executing":
            self.stop_event.set()
        else:
            log.info("run/stop no-op in phase %s", self.phase)

    def _on_run_finish(self):
        # Exactly `run/exit`'s safe shutdown (stop if moving, defer past any
        # in-flight worker), remembered as a DECISION: the flag flips the
        # teardown's closing status to "completed", and the events row makes
        # the record say who ended the run and that it was on purpose.
        if not self.finish_requested:   # a second press is just an exit
            self.finish_requested = True
            self._write("finished event", self.writer.event, "finished",
                        detail="operator finished the run")
        self._on_run_exit()

    def _on_run_exit(self):
        # exit during an action = stop first (if moving), then shut down once
        # the in-flight action reaches a terminal event (F4) — never tear down
        # mid-capture/mid-fuse, and never with a plan worker still live:
        # `_plan_worker` calls `rig.q()`, and `app.py`'s teardown would be
        # disconnecting that very RTDE interface underneath it (I2a).
        self._shutting_down = True      # no view command may start a motion now
        if self.phase in ("planning", "executing", "capturing", "fusing"):
            if self.phase == "executing":
                self.stop_event.set()
            self._exit_after = True
            log.info("exit deferred: %s in flight", self.phase)
        else:
            self._shutdown()

    # --------------------------------------------------------- worker events
    def _on_plan_done(self, ev):
        target, path = ev["target"], ev["path"]
        if self._exit_after:
            # An exit arrived mid-planning and was deferred until exactly
            # here: the plan worker has finished touching `rig`, so tearing
            # the hardware down is now safe (I2a).
            log.info("plan_done for %s discarded: exit requested", target)
            self._pending_request = None
            self._shutdown()
            return
        if self._pending_request is not None:
            # Superseded while this plan was in flight (F1a): its result —
            # success or refusal — is moot. Start the deferred request now.
            pending, self._pending_request = self._pending_request, None
            log.info("plan_done for %s discarded: %s was queued", target,
                     pending)
            self._start_planning(pending, self._pending_decider)
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
            name=f"preview-{self._target_json(target)}", daemon=True)
        self._preview_thread.start()

    def _on_worker_phase(self, ev):
        phase = ev.get("phase")
        if phase not in ("capturing", "fusing"):
            log.warning("unexpected worker phase event: %r", phase)
            return
        self._set_phase(phase, self.target)

    def _on_exec_done(self, ev):
        self.pose_active.clear()
        # The ack the exec worker blocks on before its settle leg (I4). Set
        # unconditionally and only after `pose_active` is clear: on the
        # "done" path it is the handoff, on stopped/fault nobody is waiting.
        self._pose_ack.set()
        target, outcome = ev["target"], ev["outcome"]
        if outcome == "stopped":
            # No step exists yet: the arm is stopped BEFORE `begin_step`, so
            # this turn leaves a run-level event and no step directory —
            # which is the truth (nothing was captured anywhere).
            self._write("stopped event", self.writer.event, "stopped",
                        detail=f"{self._target_json(target)}: software stop "
                               f"— arm halted mid-move")
            self.current = None
            self.blocked.clear()
            self.stop_event.clear()   # cleared only here, between actions
            self._set_phase("idle", None)
            self._publish_views()
            if self._exit_after:
                self._shutdown()
        elif outcome == "fault":
            self._write("fault event", self.writer.event, "fault",
                        detail=f"{self._target_json(target)}: executor "
                               f"halted/refused: {ev.get('detail', '')}")
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
        sid = ev.get("step_id")
        if ev["ok"]:
            if target == "survey":
                self.survey_state = "visited"
            else:
                self.visited.add(target)
                self.captures[target] = {"step": sid,
                                         "dir": self._last_cap_dir}
                self.current = target
            self.blocked.clear()
            # The fused half of the step, written from the dispatcher and only
            # here: `visited`/`current`/`blocked` have just been updated, so
            # the ViewState this attaches is the candidate set as it stands
            # after the fuse — which is what the next choice was made from.
            # A THUNK, not evaluated arguments: `_view_state` walks the whole
            # shell (reachability, poses, the accumulator's AABB) and any
            # throw inside it would otherwise happen BEFORE `_write` is
            # entered — outside its guard, escaping this handler and leaving
            # the dispatcher wedged in `fusing` with the arm on the table.
            # Inside, a failure costs this one fused record (the step stays
            # `captured`, which validates) and the machine still reaches idle.
            self._write("fused step", lambda: self.writer.write_fused(
                sid, ev["geometry"], self._view_state(sid, target)))
            self._write("captured event", self.writer.event, "captured",
                        step_id=sid)
            self._publish_object()
        else:
            # The move DID happen — only the capture/fuse failed (M3). So the
            # arm is standing somewhere new: every earlier plan refusal was
            # judged from a pose that no longer holds, and `current` would
            # otherwise keep pointing at a cell the camera has left.
            self.blocked.clear()
            self.current = None
            self.pub.log("error", ev.get("detail", ""))
            # The step itself was already marked (rejected/failed) by the
            # worker that owns it — nothing to write here.
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

    def _publish_chain(self, cap, view, seg, dropped):
        """Render and announce the identity chain for the capture just fused.

        Four images — what the camera saw, the reprojected cloud and the box
        built from it, the mask, and the points that actually entered the
        cloud — so the question "why did this view contribute THAT?" is
        answerable from the UI instead of from a hand-written replay script.

        Best-effort throughout: this is diagnostics, and the points it
        describes are already fused.
        """
        d = cap.get("dir")
        if not d:
            return
        try:
            from inspection.perception.overlay import save_chain
            files = save_chain(
                d, cap.get("rgb"), cap["T_base_cam"], self.rig.intr_color,
                source_points=view.get("prompt_source"),
                box=view.get("prompt_box") if seg is None else seg.box,
                mask=seg.mask if seg is not None else None,
                kept_points=self.acc.last_kept)
            mn, mx = self.acc.aabb(pct=BOX_PCT) or (None, None)
            self.pub.publish_chain(
                d, files,
                pose_id=cap.get("pose_id"),
                source="mask" if seg is not None else "depth",
                score=round(seg.score, 3) if seg is not None else None,
                mask_px=int(seg.mask.sum()) if seg is not None else None,
                offered=int(len(view.get("points", ()))),
                kept=int(len(self.acc.last_kept)),
                dropped=int(dropped),
                extent_mm=[round(float(v) * 1000, 1) for v in (mx - mn)]
                if mn is not None else None,
                cloud=int(len(self.acc.points)))
        except Exception:
            log.exception("chain publish failed for %s", d)

    def _save_mask(self, cap, seg):
        """Persist the mask bitmap in the step dir; never fail the run over it.

        Best-effort by design: the mask is a debugging and replay artifact,
        and the points it produced are already fused. A disk error here must
        not abort a settle that otherwise succeeded. Its score/box/px are NOT
        written here — they ride the step record (`write_capture`'s
        `segmentation`), which is the one place the schema keeps them.
        """
        d = cap.get("dir")
        if not d:
            return
        try:
            from inspection.perception.capture import save_mask
            save_mask(d, seg.mask, {"T_base_cam": cap["T_base_cam"]})
        except Exception:
            log.exception("could not save object mask to %s", d)

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
        # ---- pose-producer handoff, before anything touches the world (I4).
        # From here this thread runs `_recenter()` -> `world.set_object()`,
        # which resizes the pinocchio geom_model the publisher indexes into.
        # Queueing `exec_done` is not enough: `pose_active` is cleared when
        # the dispatcher *processes* it, and even then a tick may be inside
        # `_pose_payload`. So wait for the dispatcher's ack, then for the
        # streamer to confirm no publish is in flight. Both bounded — a
        # missed handoff must not wedge a run with a robot in it.
        if not self._pose_ack.wait(2.0):
            log.warning("no pose handoff ack in 2 s — continuing to capture")
            self.pub.log("warn", "pose handoff timed out — continuing")
        # Asked for even after a missed ack: an ack that lands a millisecond
        # late still leaves the quiesce able to give the real guarantee.
        if not self.pose_quiesce(1.0):
            log.warning("pose streamer did not quiesce in 1 s")
            self.pub.log("warn", "pose stream did not quiesce — continuing")
        self.events.put({"ev": "phase", "gen": gen, "phase": "capturing"})
        # The step is allocated HERE, after the move: a step directory means
        # "the arm stood somewhere and a camera was pointed at it". A move
        # that stopped or faulted never gets one (`_on_exec_done` logs the
        # event instead) — the id space stays dense over things that happened.
        #
        # Inside the try, and `sid` defined before it: `begin_step` does real
        # filesystem work (mkdir, run.json flush) and can fail on a full or
        # read-only disk. If it did so unguarded, this thread would die
        # without queueing `settle_done` — and `capturing` is a busy phase, so
        # the dispatcher would drop every further command INCLUDING run/exit,
        # leaving the operator no way to stop a run with a robot in it.
        sid, sdir = None, None
        try:
            sid, sdir = self.writer.begin_step(
                {"method": VIEW_METHOD,
                 "address": None if target == "survey" else list(target)})
            cap = self.rig.capture(sid, sdir)
            # `settle_capture` (run/settle.py) is the object-identity + plane-
            # gate + fuse pipeline, verbatim out of this worker: it rejects a
            # bad view BEFORE anything from it is kept (no accumulator touch,
            # no capture publish, no cell marked visited) and fuses into
            # `self.acc` on success. `settle_done ok=False` is the same path
            # a failed capture takes below — the run carries on either way.
            res = settle_capture(cap, self.rig, self.segmenter, self.acc,
                                 is_survey=(target == "survey"),
                                 on_warn=lambda msg: self.pub.log("warn", msg),
                                 floor_z=self._floor_z)
            if not res.ok:
                # An explicit entry, never a silent gap: the bytes are on disk
                # and the record says why they were refused.
                self.writer.mark_step(sid, outcome="rejected", detail=res.detail)
                self.events.put({"ev": "settle_done", "gen": gen,
                                 "target": target, "ok": False, "npts": 0,
                                 "step_id": sid, "detail": res.detail})
                return
            self.writer.write_capture(
                sid, t_captured=cap["t"], joints_rad=_floats(cap["q"]),
                T_base_flange=_mat(self.ik.fk(cap["q"])),
                T_base_cam=_mat(cap["T_base_cam"]),
                rgb_rotation_deg=int(cap["rgb_rotation_deg"]),
                segmentation=_segmentation(res.seg))
            self.events.put({"ev": "phase", "gen": gen, "phase": "fusing"})
            if res.seg is not None:
                self._save_mask(cap, res.seg)
            self._recenter()
            if self.sphere is not None:
                self._reach = self.sphere.reachability(self.world, self.ik)
            self._publish_chain(cap, res.view, res.seg, res.dropped)
            self.pub.publish_capture(cap)
            self._last_cap_dir = cap.get("dir")
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": True, "npts": res.npts, "step_id": sid,
                             "geometry": self._geometry(res), "detail": ""})
        except Exception as e:
            log.exception("capture/settle failed for step %s", sid)
            try:
                if sid is not None:
                    self.writer.mark_step(sid, outcome="failed",
                                          detail=f"capture failed: {e}")
            except Exception:
                # The dispatcher MUST get its terminal event even if the
                # record cannot be written — a wedged run is worse than a
                # gap in the roster, and both are logged.
                log.exception("could not mark step %s failed", sid)
            self.events.put({"ev": "settle_done", "gen": gen, "target": target,
                             "ok": False, "npts": 0, "step_id": sid,
                             "detail": f"capture failed: {e}"})

    # ------------------------------------------------------------- helpers
    def _recenter(self):
        center = self.acc.centroid
        # pct-trimmed: raw min/max let 168 stray cable points (4.7% of the
        # cloud) grow the box until it contained the arm's own pose and every
        # plan died in OMPL's start tree. See CloudAccumulator.aabb.
        mn, mx = self.acc.aabb(pct=BOX_PCT)
        if self.r is None:
            # DERIVED ONCE, from the survey cloud, then frozen for the run.
            # Frozen because cells are the keys for coverage and evidence: a
            # shell that tightened as the cloud grew would silently re-point
            # addresses that have already been visited and photographed.
            # rig.intr is the IR-left/depth intrinsics; the colour frame the
            # VLM reads differs by ~1.6% in fy on the D405 — below the noise
            # in the extent itself.
            extent = object_extent(mn, mx)
            self.r = radius_for_extent(extent, self.rig.intr["fy"],
                                       self.rig.intr["height"])
            self.pub.log("info", f"shell r={self.r:.3f} m derived from a "
                                 f"{extent * 1000:.0f} mm object")
            # The shell radius is a property of the RUN's view method, and it
            # only exists once the survey has seen the object — so the method
            # params are completed here, where it is derived, rather than
            # guessed at run creation.
            self.writer.set_view_method_params(VIEW_METHOD, r=float(self.r))
        self.sphere = ViewSphere(center, r=self.r)
        dims = np.maximum(mx - mn + 0.04, 0.05)         # 2 cm margin each side
        mid = (mn + mx) / 2
        self.world.set_object("object", dims.tolist(),
                              [*mid.tolist(), 0.0, 0.0, 0.0], parent="base")

    def _write(self, what, fn, *a, **kw):
        """Run one writer verb from the DISPATCHER; log and carry on if it
        raises.

        Best effort by design, and only on this thread: a run that loses a
        line of its record is bad, a dispatcher that aborts mid-handler and
        leaves the machine stuck in `fusing` with a robot on the table is
        worse. The worker side is deliberately not like this — there a failed
        write marks the step `failed` and the terminal event still goes out.
        """
        try:
            fn(*a, **kw)
        except Exception:
            log.exception("record write failed (%s)", what)

    def _geometry(self, res):
        """This step's fusion accounting, as `GeometryStats` (worker side).

        Built where the numbers are — `res` and the accumulator are both the
        exec worker's — and carried to the dispatcher on `settle_done`, so
        the fused record is written by the one thread that owns run state.
        """
        mn_mx = self.acc.aabb(pct=BOX_PCT)
        return {
            "offered": int(res.npts),
            "kept": int(len(self.acc.last_kept)),
            "dropped": int(res.dropped),
            "fused_points": int(len(self.acc.points)),
            "source": "mask" if res.seg is not None else "depth",
            "fallback_reason": res.fallback,
            "extent_mm": None if mn_mx is None else
            [round(float(v) * 1000, 1) for v in (mn_mx[1] - mn_mx[0])],
            "plane": _floats(res.view["plane"]) if res.view is not None
            and res.view.get("plane") is not None else None,
        }

    def _view_state(self, step_id, target):
        """The candidate view set as it stands at the end of this step.

        PHYSICAL truth only (schema.py:ViewState): every cell of the shell
        with the status the loop would colour it, its camera pose, and the
        roll that made it reachable. `chosen`/`decider` say which of them this
        step went to and who picked it — the operator's button, the
        orchestrator, or a sweep.
        """
        cands = []
        if self.sphere is not None:
            for cell in self.sphere.cells():
                roll = self._reach.get(cell)
                if roll is None:
                    status = "unreachable"
                elif cell == self.current:
                    status = "current"
                elif cell in self.visited:
                    status = "visited"
                elif cell in self.blocked:
                    status = "blocked"
                else:
                    status = "available"
                cands.append(ViewCandidate(
                    address=list(cell), status=status,
                    pose=_mat(self.sphere.cam_pose(*cell, roll or 0.0)),
                    roll=None if roll is None else float(roll)))
        mn_mx = self.acc.aabb(pct=BOX_PCT)
        centroid = self.acc.centroid
        return ViewState(
            step_id=step_id, t=time.time(), candidates=cands,
            centroid=None if centroid is None else _floats(centroid),
            extent=None if mn_mx is None else _floats(mn_mx[1] - mn_mx[0]),
            r=None if self.r is None else float(self.r),
            chosen=self._target_json(target), decider=self._decider)

    def _set_phase(self, phase, target=None):
        self.phase, self.target = phase, target
        total = sum(1 for r in self._reach.values() if r is not None)
        self.pub.status(phase=phase, target=self._target_json(target),
                        visited=len(self.visited), total=total)
        if phase in ("idle", "fault"):
            self._publish_real_pose()

    def _publish_real_pose(self):
        """Publish where the arm ACTUALLY is. Dispatcher-side, idle/fault only.

        Poses otherwise have exactly two producers — the preview worker while
        `previewing`, the pose streamer while `executing` — and neither runs
        here, so this cannot race them (the same phase-exclusivity that makes
        the publisher's cached pinocchio data safe).

        Without this the retained `scene/poses` topic is either EMPTY at
        startup — every visual mesh then renders at its identity placement,
        stacked on the base — or left showing the last frame of a preview that
        was cancelled, i.e. an arm position that never happened. FR1 says a
        window opened at any moment shows the current world; the arm's real
        configuration is part of that.
        """
        try:
            self.pub.publish_pose(self.world, self.rig.q())
        except Exception:
            # A rig that cannot report its pose must never break the
            # dispatcher — the UI just keeps the previous frame.
            log.exception("publish_real_pose failed")

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
            if self._preview_thread.is_alive():
                # The whole point of the join is that the outgoing preview is
                # provably done touching `pub` before a planner or `_recenter`
                # runs. If it timed out, that proof is gone — say so instead
                # of proceeding silently on an assumption that just failed.
                log.warning("preview thread did not exit within 1 s — "
                            "proceeding without the join guarantee")
                self.pub.log("warn", "preview thread slow to cancel")
                # Keep the handle: `join_workers` at teardown is the last
                # chance to catch this straggler before `rig.close()`, and
                # dropping the reference here is what would make it
                # unreachable. `_preview_cancel` is already set, so a new
                # preview overwriting these two fields loses nothing.
                return
        self._preview_cancel = None
        self._preview_thread = None

    def _shutdown(self):
        self._cancel_preview()
        self.pose_active.clear()
        self._set_phase("done")

    @staticmethod
    def _normalize_target(raw):
        """Command target -> "survey" | (h, v) | None.

        None means "malformed": the caller rejects it with a `pub.log` line
        the operator can see (M5). Commands arrive from the bus and are
        untrusted, so `tuple(raw)` on a number or a 5-element list used to
        raise out of the handler and into the dispatcher's blanket
        `except Exception`, which logs a traceback and looks like a bug in
        the state machine rather than a bad message.
        """
        if raw == "survey":
            return "survey"
        # Marker id, as published in the scene: "h07v1" (ui.publisher.cell_key).
        # Clicking a sphere in the 3D view hands back the id the BACKEND chose,
        # so the viewsphere's naming stays on this side of the bus and the
        # copied scene panel never learns what a cell is. Strict on purpose —
        # a near-miss id is a bad command, not a cell to guess at.
        if isinstance(raw, str):
            m = re.fullmatch(r"h(\d{2})v(\d)", raw)
            return (int(m.group(1)), int(m.group(2))) if m else None
        try:
            h, v = raw                      # TypeError / ValueError on anything else
            hi, vi = int(h), int(v)
        except (TypeError, ValueError):
            return None
        # `int()` is for numpy scalars off the viewsphere, not for coercion:
        # a target that is not exactly an integer pair is a bad command, and
        # silently flooring 3.7 to cell 3 would move the arm somewhere the
        # operator did not click.
        if (hi, vi) != (h, v):
            return None
        return (hi, vi)

    @staticmethod
    def _target_json(target):
        return list(target) if isinstance(target, tuple) else target


# --- record shaping ---------------------------------------------------------
# numpy is what the loop computes in; JSON is what the record is. These three
# are the whole conversion, in one place, so no write site does it by hand.

def _floats(v) -> list[float]:
    return [float(x) for x in np.asarray(v, float).ravel()]


def _mat(T) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.asarray(T, float)]


def _segmentation(seg) -> dict | None:
    """`run.segmenter.Segmentation` -> the schema's trimmed record.

    `box` is the PROMPT box in the raw image frame (schema.py:Conventions),
    `px` the mask the model actually returned — score/box/px only, by the
    schema review's decision; the mask itself stays a binary on disk.
    """
    if seg is None:
        return None
    return {"score": float(seg.score), "box": [int(v) for v in seg.box],
            "px": int(np.asarray(seg.mask).sum())}
