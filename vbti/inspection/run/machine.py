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
directly, and `_reach` (the cell accessibility map — one resolved ray per
cell, expensive to compute) is refreshed by `_plan_worker`/`_exec_worker`
(inside `_recenter`) rather than by the dispatcher.

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

MOVES ARE TRANSACTIONS (Stage B, design 2026-09-17-ray-views-design.md
§Move transaction). A cell or face move is never one plan any more: it is
`motion/transaction.py`'s three legs — slide-out along the ray the camera
stands on, hop through open space, slide-in along the target ray. The
dispatcher keeps the ray the arm sits on (`_ray_state`, leg 1's `current`),
the exec worker runs the legs one `rig.move` at a time under the phase each
one earns (`adjusting` for a slide, `executing` for the hop) and records the
waypoints it ACTUALLY travelled (`_arrival_path`) — which is what the
retreat's retrace tier reverses when a move is stopped half-way.

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

from inspection.cell.geometry import (BOX_PCT, CloudAccumulator, object_box,
                                      pose_box_aabb)
from inspection.cell.world import DEFAULT_STEP, RobotCell
from inspection.motion.direct import direct_move
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.motion.transaction import RayState, plan_transaction
from inspection.record.schema import ViewCandidate, ViewState
from inspection.run.settle import _plane_error, settle_capture
from inspection.view.grid import DEFAULT_R
from inspection.view.rays import (Ray, belief_bounds, cell_map,
                                  resolve_face)
from inspection.view.viewsphere import T_FLANGE_CAM, ViewSphere

log = logging.getLogger(__name__)

# Phases in which the arm is either moving or unrecoverable without operator
# action at the pendant — no new request may be accepted while in one of these.
# `adjusting` is a slide leg of a move transaction: the arm is travelling
# radially along a ray, so it is every bit as busy as it is during the hop.
_BUSY_PHASES = ("adjusting", "executing", "capturing", "fusing", "fault")

#: Phases in which the arm is under power on a planned path — the ones a
#: `run/stop` must interrupt and a `run/exit` must wait out.
_MOVING_PHASES = ("adjusting", "executing")

#: The move transaction's legs in execution order, with the phase each runs
#: under (plan §Task 3). A slide is a delicate radial approach that cannot
#: laterally sweep the object; the hop is the ordinary free-space move, and
#: keeps the phase name the loop has always used for it.
_LEG_PHASES = (("out", "adjusting"), ("hop", "executing"),
               ("slide_in", "adjusting"))

#: The capture-refit self-trap (run 1409-lan-ai_bench1): `_recenter()` refits
#: the object box from the grown cloud while the arm still STANDS at the
#: viewpoint that produced the points; a stance accepted with millimetres of
#: margin (padding is the only standoff) goes into collision, and from an
#: invalid start every plan refuses — tier 1 fails at config 0, OMPL cannot
#: initialize its start tree. The refusals blame the targets, so nothing
#: upstream can tell the stance is the problem. After this many consecutive
#: refusals from an unmoved stance, the machine retreats radially instead of
#: refusing forever (Anton 2026-09-14).
RETREAT_AFTER_FAILS = 3
#: How far to back the flange out, away from the sphere center [m]. 3 cm
#: already cleared the recorded trap (margin deficit ~10 mm); 5 buys headroom
#: against bigger refits for one extra beat of travel.
RETREAT_M = 0.05

#: The id of the view method this machine addresses cells with, as declared in
#: the run's `view_methods` (`RunWriter.create`). Every step's `view.method`
#: carries it, and `validate_run` refuses a step whose method the run never
#: declared — so the two must be written from one constant.
VIEW_METHOD = "vs"

#: `_face_states()` status -> `ViewCandidate.status` (schema.py:280, a closed
#: Literal that predates faces and has no "ready"/"settling" of its own).
#: "settling" candidates are never recorded (`_view_state` skips them — an
#: unstable box has no honest pose to log); "ready" maps onto "available",
#: the cell vocabulary's word for exactly the same fact (reachable, not yet
#: visited). "unreachable"/"visited" already match verbatim.
_FACE_STATUS_TO_CANDIDATE = {"ready": "available", "unreachable": "unreachable",
                             "visited": "visited"}

# `_plane_error` now lives in `run/settle.py` (the plane gate is part of the
# extracted settle pipeline); re-exported here so any external reference to
# `machine._plane_error` keeps working.


class Supervisor:
    def __init__(self, rig, pub, writer, q_survey, world=None, ik=None,
                 r=None, seed=0, segmenter=None, fit=None):
        self.rig, self.pub = rig, pub
        #: Name of the per-step primitive fitter (`fit/apply.py:METHODS`), or
        #: None for no live fitting. Default None: composition roots opt in
        #: explicitly (`app.py`/`mock.py` pass "boxguard"), the same way the
        #: segmenter below is injected rather than assumed — a bare unit-test
        #: Supervisor must not drag open3d onto every fused step. When set,
        #: each fused step gets the SAME `decomposition.json` the offline
        #: backfill would write: same registry, same record builder, same
        #: writer (`_fit_step`).
        self.fit = fit
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
        self.seed = seed
        if r is not None:
            # Cells are RAYS now (Stage B): each one's distance is resolved
            # per cell against the live world, so there is no one shell
            # radius left to pin. Accepted (app.py still offers `--r`) and
            # said out loud rather than silently ignored.
            log.warning("--r %.3f ignored: cells are rays, each cell's "
                        "distance is resolved on its own ray", float(r))
            self.pub.log("warn", f"--r {float(r):.3f} ignored: cells are "
                                 f"rays now, each resolves its own distance")
        #: Refit counter — every `_recenter()` box install bumps it, and
        #: `_last_box_rev`/`_stance_box_rev` ride along with their boxes.
        #: Pure diagnostics (1509-cam-ai_bench1: four silent failed retreats,
        #: and the record could not say WHICH box the retreat validated
        #: against). rev equality stance==live at retreat time is the
        #: smoking gun for a poisoned stance box.
        self._box_rev = 0
        self._last_box_rev = 0
        self._stance_box_rev = 0
        #: The waypoints the arm ACTUALLY travelled on its way to the stance
        #: it is in — the retreat's retrace tier reverses them when radial-out
        #: has no IK (far-side cells stand at the workspace boundary:
        #: cam_bench2's [5,0] flange sat at 0.751 m and even +1 cm out had 0
        #: branches). Planned under exactly the stance-box world, so reversing
        #: it is IK-valid by construction and consistent with the 2026-09-14
        #: validation semantics. EXECUTED, not planned (Stage B): a move
        #: stopped half-way through its slide-in has a hop and a slide-out
        #: behind it and nothing in front, and retracing the planned path
        #: would jump the arm to the end of a leg it never finished. Written
        #: by `_on_exec_done` from the worker's own count of what moved; None
        #: until the first move; cleared after any retreat motion (the arm is
        #: no longer where the path ends).
        self._arrival_path = None
        #: The ray the camera stands on — `motion/transaction.RayState`, and
        #: leg 1's `current`. Set on arrival from the resolution that was
        #: actually executed, cleared whenever the arm leaves that ray without
        #: arriving at another (stop, fault, retreat) or whenever the view it
        #: was standing for turned out to be unusable (failed settle). None is
        #: the honest "no current ray": the transaction then has no slide-out.
        self._ray_state = None
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
        #: The `Legs` behind `self._path`, or None for a plan that is one leg
        #: (the survey). Handed to the exec worker at `view/confirm`, which is
        #: the only thing that needs the legs apart; everything else — the
        #: preview replay, the record, the retrace — reads the concatenation.
        self._legs = None
        #: Where each leg ENDS in `self._path`: `[out, hop, slide_in]`, so
        #: `path[:b0+1]`, `path[b0:b1+1]`, `path[b1:b2+1]` are the three legs
        #: and an empty slide-out is simply `b0 == 0`. Published on the status
        #: strip beside the phase (plan §Task 3, additive payload field).
        self._leg_bounds = None
        #: The `RayResolution` `self._path` was planned to, or None when the
        #: target is not a ray (the survey). It becomes `_ray_state` on
        #: arrival, so the ray the arm sits on is the one it was actually
        #: sent to — never re-read from the cell map, which the arrival's own
        #: refit has already moved.
        self._target_res = None
        #: The cell accessibility map — `{(h, v): RayResolution | None}`,
        #: `None` = no provably-safe sample on that cell's ray. Worker-
        #: refreshed (see the module docstring's second exception).
        self._reach = {}
        #: Axis-aligned bounds of the box `_recenter` installed as the world
        #: obstacle. The cell rays' FALLBACK belief bound, used only until a
        #: boxguard fit gives an unpadded one. None until survey.
        self._object_aabb = None
        #: Mean resolved cell depth [m] from the last refit, or None when no
        #: cell resolved — the Stage B stopgap `ViewState.r` carries.
        self._r_mean = None
        self._exit_after = False    # run/exit arrived mid-action, shut down at its end
        self._shutting_down = False  # set by request_shutdown (signal thread)
        #: `run/finish`: the operator declared the run DONE, not abandoned.
        #: The machine only remembers it — the app teardown reads it to close
        #: the record "completed" where a bare exit closes "aborted".
        self.finish_requested = False
        self._last_cap_dir = None   # capture dir of the most recent capture()
        self._pending_request = None  # target deferred while a plan is in flight
        self._plan_fails = 0        # consecutive refusals from an unmoved stance
        self._last_box = None       # (dims, pose) currently installed by _recenter
        self._stance_box = None     # (dims, pose) the CURRENT stance was approved under
        self._face_prim = None      # freshest boxguard Primitive (or None)
        self._face_stable = 0       # consecutive small normal deltas (gate: >=2)
        #: What the last plan's `resolve_face` found, per face:
        #: `{"theta", "shift", "depth", "pos", "pose"}` — the achieved
        #: off-angle [deg], the fronto-parallel shift [m], the resolved depth
        #: [m], the camera endpoint [x, y, z] and the camera pose it came
        #: from. ONE dict, because one `RayResolution` produces all of it and
        #: two dicts could disagree; `pos` is `pose`'s translation, kept
        #: ready-made because the views payload publishes exactly that list.
        self._face_find = {}
        self._face_unreachable = set()  # faces whose ray family resolved None
        self.face_visited = set()   # face handles captured this run
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
                if self._exit_after and ev.get("ev") in ("plan_done", "settle_done",
                                                         "retreat_done"):
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
        elif kind == "retreat_done":
            self._on_retreat_done(ev)
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
        elif isinstance(target, tuple) and target[0] == "face":
            st = self._face_states()[target[1]]["status"]
            if st != "ready":
                log.info("view/request face %s dropped: %s", target[1], st)
                self.pub.log("warn", f"face {target[1]} dropped: {st}")
                return
        else:
            # The cell map IS the gate (Stage B): `None` means the cell's ray
            # carried no provably-safe sample. Say WHICH of the three facts
            # refused the request — "dropped" alone left the operator (and
            # the mover reading `pub.log`) guessing at a fact the machine
            # already knows.
            if self.sphere is None:
                reason = "no object yet"
            elif target not in self._reach:
                reason = "not a cell on this grid"
            elif self._reach[target] is None:
                reason = "inaccessible: no safe sample on its ray"
            elif target in self.visited:
                reason = "already visited"
            else:
                reason = None
            if reason is not None:
                log.info("view/request %s dropped: %s", target, reason)
                self.pub.log("warn", f"view/request {target} dropped — "
                                     f"{reason}")
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
        # The brain client records its own requested/awaiting/approved rows
        # (brain/live.py). Operator-originated requests had NO writer at all —
        # 1509-cam-ai_bench1's step 3 appeared out of thin air in the record.
        # The decider gates the write so brain rows are never duplicated.
        if decider == "operator":
            self._write("requested event", self.writer.event, "requested",
                        detail=f"cell={self._target_json(target)} "
                               f"decider=operator")
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
        # The stance this move ends at was validated against the world as it
        # stands NOW; no refit can run between here and arrival (refits live
        # in this move's own settle leg). Ordering is safe: `_last_box` is
        # written by the exec worker before it queues `settle_done`, and the
        # queue is processed before any next confirm can exist.
        self._stance_box = self._last_box
        self._stance_box_rev = self._last_box_rev
        # `_arrival_path` is NOT written here any more: the retrace needs the
        # waypoints that were EXECUTED, and only the worker knows how far the
        # transaction got. `_on_exec_done` writes it from the worker's count.
        if self._decider == "operator":    # brain writes its own row
            self._write("approved event", self.writer.event, "approved",
                        detail=f"cell={self._target_json(self.target)} "
                               f"decider=operator")
        self._set_phase("executing", self.target)
        self._action_thread = threading.Thread(
            target=self._exec_worker,
            args=(self.gen, self.target, self._path, self._legs),
            name=f"exec-{self._target_json(self.target)}", daemon=True)
        self._action_thread.start()

    def _on_run_stop(self):
        if self.phase in _MOVING_PHASES:
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
        if self.phase in ("planning", "capturing", "fusing") + _MOVING_PHASES:
            if self.phase in _MOVING_PHASES:
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
        # I3: `_plan_worker` runs on a worker thread and may not touch run
        # state (module docstring, "only the dispatcher mutates run state").
        # It carries a face plan's gate-state findings on the event instead;
        # applied HERE, AFTER the exit/supersede checks above, so a
        # superseded plan's face_find/face_unreachable never lands even
        # though its OWN result was just discarded.
        face_find = ev.get("face_find")
        if face_find is not None:
            name, find = face_find
            self._face_find[name] = find
        face_unreachable = ev.get("face_unreachable")
        if face_unreachable is not None:
            self._face_unreachable.add(face_unreachable)
        if path is None:
            if target == "survey":
                self.survey_state = "available"
            elif isinstance(target, tuple) and target[0] == "face":
                # I1: a face refusal already lives in `_face_unreachable`/
                # `_face_find` (applied above) — `blocked` stays cell-only
                # so `_menu_input`'s `sorted(map(list, v))` over it never
                # meets a face tuple, and the menu never shows a face BOTH
                # "blocked" and "ready" at once.
                pass
            else:
                self.blocked.add(target)
            self.pub.log("warn", f"plan refused for {target}: "
                          f"{ev.get('detail', '')}")
            self._plan_fails += 1
            if (self._plan_fails >= RETREAT_AFTER_FAILS
                    and self.sphere is not None and not self._shutting_down):
                self._start_retreat()
                return
            self._set_phase("idle", None)
            self._publish_views()
            return
        # A found path disproves the trap: the stance can still reach
        # somewhere, so the refusals were about targets, not the start.
        self._plan_fails = 0
        self._path = path
        # The transaction rides along with its concatenation: the legs for the
        # executor, the boundaries for the status strip, the resolution for
        # the ray state the arrival will install. All three are None for a
        # survey, which is one plain leg and no ray at all.
        self._legs = ev.get("legs")
        self._leg_bounds = ev.get("bounds")
        self._target_res = ev.get("resolution")
        if self._decider == "operator":    # brain writes its own row
            self._write("awaiting event", self.writer.event,
                        "awaiting_approval",
                        detail=f"cell={self._target_json(target)} "
                               f"decider=operator")
        self._set_phase("previewing", target)
        self._publish_views()
        self._preview_cancel = threading.Event()
        self._preview_thread = threading.Thread(
            target=self._preview_worker, args=(path, self._preview_cancel),
            name=f"preview-{self._target_json(target)}", daemon=True)
        self._preview_thread.start()

    def _on_worker_phase(self, ev):
        phase = ev.get("phase")
        # `adjusting`/`executing` are the move transaction's own legs: the
        # worker announces each one as it starts it, so the phase strip says
        # which of the three the arm is on (`executing` is re-announced for
        # the hop — the dispatcher set it at `view/confirm`, and a slide-out
        # before it has since moved on).
        if phase not in ("adjusting", "executing", "capturing", "fusing"):
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
        # What the arm TRAVELLED, whatever the outcome — the retrace tier's
        # input, and the only record of where a half-executed transaction
        # left it.
        self._arrival_path = ev["executed"]
        # Cleared FIRST, in every outcome, and only then re-earned on
        # arrival: if `_ray_state_of` ever threw, an assignment written the
        # other way round would leave the PREVIOUS move's ray in place and
        # the next transaction would slide out along a ray the camera has
        # left. Stopped or faulted needs no re-earning at all — the arm is
        # parked between ray depths (or between rays), which is the same
        # reason `current` is cleared below.
        self._ray_state = None
        if outcome == "done":
            # Arrival: the ray the executed resolution sits on becomes the
            # ray the camera stands on, and leg 1's `current`. A survey
            # resolves no ray, so arriving there says so with None.
            if self._target_res is not None:
                self._ray_state = _ray_state_of(self._target_res)
        if outcome == "stopped":
            # No step exists yet: the arm is stopped BEFORE `begin_step`, so
            # this turn leaves a run-level event and no step directory —
            # which is the truth (nothing was captured anywhere).
            self._write("stopped event", self.writer.event, "stopped",
                        detail=f"{self._target_json(target)}: software stop "
                               f"— arm halted mid-move")
            self.current = None
            self.blocked.clear()
            self._plan_fails = 0      # the arm moved — earlier refusals are moot
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
                if isinstance(target, tuple) and target[0] == "face":
                    self.face_visited.add(target[1])
                else:
                    self.visited.add(target)
                self.captures[target] = {"step": sid,
                                         "dir": self._last_cap_dir}
                self.current = target
            self.blocked.clear()
            self._plan_fails = 0    # the arm stands somewhere new
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
            # The accumulator's arrays ride along: `_exec_worker` fused them
            # BEFORE queueing settle_done and nothing touches `acc` until the
            # next move is commanded, so this is the exact post-fuse state —
            # recorded, not left to replay (StepRecord v2, Anton 2026-09-10).
            self._write("fused step", lambda: self.writer.write_fused(
                sid, ev["geometry"], self._view_state(sid, target),
                points=self.acc.points, colors=self.acc.colors))
            # Immediately after the fused cloud hits disk, and on the SAME
            # array `write_fused` just saved — so the record's `n_points` is
            # the byte-for-byte length of `fused/cloud.npy`, exactly as the
            # offline backfill would compute it. Guarded like every other
            # dispatcher write: a fit (or its write) that dies costs this one
            # slot, never the machine's way back to idle.
            self._write("step decomposition", self._fit_step, sid)
            self._write("captured event", self.writer.event, "captured",
                        step_id=sid)
            self._publish_object()
        else:
            # The move DID happen — only the capture/fuse failed (M3). So the
            # arm is standing somewhere new: every earlier plan refusal was
            # judged from a pose that no longer holds, and `current` would
            # otherwise keep pointing at a cell the camera has left.
            self.blocked.clear()
            self._plan_fails = 0
            self.current = None
            # The view the arm stood for was refused, so the address it was
            # standing at is not one this run has served — the next move
            # plans its slide-out from nothing rather than from a ray whose
            # own view never counted.
            self._ray_state = None
            self.pub.log("error", ev.get("detail", ""))
            # The step itself was already marked (rejected/failed) by the
            # worker that owns it — nothing to write here.
        if self._exit_after:
            self._shutdown()
        else:
            self._set_phase("idle", None)
        self._publish_views()   # recover the retained topic either way (F3)

    def _start_retreat(self):
        """Escape the capture-refit self-trap: back the flange out radially.

        Reached from `_on_plan_done` after RETREAT_AFTER_FAILS consecutive
        refusals from an unmoved stance. The move is validated in the worker
        against the box the stance was APPROVED under (`_stance_box`), on a
        throwaway RobotCell — never against the live world, whose refit is
        exactly what swallowed the stance; and never ON the live world, so
        no pose handoff is needed and nothing races the streamer.

        A FAULTED move can never reach here, and that is load-bearing:
        `_arrival_path` then holds a half-executed transaction, and the
        retrace tier would drive the arm back through legs whose executor
        just refused. What prevents it is `"fault" in _BUSY_PHASES` — the
        request gate drops every `view/request` from `fault`, so
        `_plan_fails` can never climb to RETREAT_AFTER_FAILS once the run
        is halted. Said out loud because nothing here would stop it.
        """
        self.gen += 1               # any stale worker event dies at the gate
        self.stop_event.clear()
        self.pose_active.set()      # the streamer shows the arm moving
        # A retreat is not a transaction: drop the previewed plan's legs,
        # boundaries and resolution together, so the status strip cannot
        # describe this move with them and no arrival can install a ray state
        # out of a plan this move is not executing.
        self._legs = self._leg_bounds = self._target_res = None
        self._set_phase("executing", None)
        self.pub.log("warn", f"{RETREAT_AFTER_FAILS} plans refused from this "
                             f"stance — retreating {RETREAT_M * 100:.0f} cm "
                             f"to re-plan")
        # In the RECORD, not just the UI bus: 1509-cam-ai_bench1 fired four
        # retreats and the run kept no trace (pub.log reaches neither the
        # terminal nor events.jsonl). The box revs are the stance-identity
        # diagnostic: stance_rev == live rev means the retreat is about to
        # validate against the very box that swallowed the stance.
        self._write("retreat event", self.writer.event, "retreat",
                    detail=f"start after {RETREAT_AFTER_FAILS} refusals — "
                           f"stance_box rev {self._stance_box_rev}, "
                           f"live box rev {self._box_rev}")
        self._action_thread = threading.Thread(
            target=self._retreat_worker,
            args=(self.gen, self._stance_box, self._arrival_path),
            name="retreat", daemon=True)
        self._action_thread.start()

    def _retreat_worker(self, gen, stance_box, arrival_path=None):
        try:
            q_now = self.rig.q()
            w = RobotCell()         # same cell.yaml the live world was built from
            T = np.asarray(self.ik.fk(q_now))
            out = T[:3, 3] - np.asarray(self.sphere.center, dtype=float)
            n = float(np.linalg.norm(out))
            if n < 1e-6:
                raise RuntimeError("flange is at the sphere center")
            T_r = T.copy()
            T_r[:3, 3] = T[:3, 3] + out / n * RETREAT_M
            branches = self.ik.branches(T_r, q_near=q_now)
            # Two validation tiers (Anton 2026-09-15, after cam_bench1's four
            # no-motion retreats while the escape provably existed):
            #   1. the stance box — the world that approved this stance
            #      (the 2026-09-14 semantics, unchanged);
            #   2. env-only — the object box removed. The move is radially
            #      AWAY from the sphere center the object sits at, so object
            #      clearance grows by construction; if the stance box is
            #      poisoned (holds a post-refit box that already swallows
            #      q_now), that box is exactly what must be ignored to leave.
            # Cell keep-outs, table, and self-collision guard both tiers.
            tiers = ([("stance-box", stance_box), ("env-only", None)]
                     if stance_box is not None else [("env-only", None)])
            tried = []
            for tag, box in tiers:
                if box is not None:
                    dims, pose = box
                    w.set_object("object", dims, pose, parent="base")
                else:
                    w.remove_object("object")
                free = [q for q in branches if not w.is_colliding(q)]
                free.sort(key=lambda q: float(np.abs(q - q_now).sum()))
                path = None
                for q_goal in free:
                    path = direct_move(w, q_now, q_goal)
                    if path is not None:
                        break
                if path is None:
                    tried.append(f"{tag}: {len(free)}/{len(branches)} "
                                 f"branch(es) free, no direct path")
                    continue
                # w carries this tier's box state — the rig's final gate must
                # judge the motion by the world it was planned under, not the
                # live world whose refit is what the arm is escaping.
                rep = self.rig.move(path, world=w)
                detail = f"via {tag}" if tag != "stance-box" else ""
                if rep.get("stopped"):
                    detail = (detail + " — " if detail else "") + \
                        "stopped mid-retreat"
                self.events.put({"ev": "retreat_done", "gen": gen,
                                 "moved": True, "detail": detail})
                return
            # Retrace tier (Anton 2026-09-15, cam_bench2): far-side stances
            # sit at the workspace boundary — radial-out can have ZERO IK
            # branches ([5,0] flange at 0.751 m; +1 cm out: none). The path
            # that brought the arm here was planned under the stance-box
            # world and executed, so its reverse is IK-valid by construction;
            # re-validated per tier on the throwaway cell all the same.
            if arrival_path is not None and len(arrival_path) >= 2:
                rev = [np.asarray(q, dtype=float)
                       for q in reversed(arrival_path)]
                rev[0] = q_now          # ≈ arrival end; honest start anyway
                for tag, box in tiers:
                    if box is not None:
                        dims, pose = box
                        w.set_object("object", dims, pose, parent="base")
                    else:
                        w.remove_object("object")
                    ok, _info = w.path_valid(rev)
                    if not ok:
                        tried.append(f"retrace({tag}): path invalid")
                        continue
                    rep = self.rig.move(rev, world=w)
                    detail = f"via retrace({tag})"
                    if rep.get("stopped"):
                        detail += " — stopped mid-retreat"
                    self.events.put({"ev": "retreat_done", "gen": gen,
                                     "moved": True, "detail": detail})
                    return
            else:
                tried.append("retrace: no arrival path stored")
            self.events.put({"ev": "retreat_done", "gen": gen,
                             "moved": False,
                             "detail": "no valid retreat move ("
                                       + "; ".join(tried) + ")"})
        except Exception as e:
            log.exception("retreat worker failed")
            self.events.put({"ev": "retreat_done", "gen": gen, "moved": False,
                             "detail": f"retreat error: {e}"})

    def _on_retreat_done(self, ev):
        self.pose_active.clear()
        self._pose_ack.set()        # parity with exec: nobody waits, no harm
        self.stop_event.clear()     # between actions, like the exec-stop path
        # Reset in EVERY outcome: after a move the old refusals are moot, and
        # after a failed attempt the next retreat must be re-earned by three
        # fresh refusals — never re-triggered per-refusal into a thrash loop.
        self._plan_fails = 0
        self._write("retreat event", self.writer.event, "retreat",
                    detail=(f"moved: {ev.get('detail') or 'ok'}" if ev["moved"]
                            else f"FAILED: {ev.get('detail', '')}"))
        if ev["moved"]:
            # The stance the refusals were judged from is gone, and so is the
            # cell the camera was standing at.
            self.blocked.clear()
            self.current = None
            self._arrival_path = None   # the arm no longer stands at its end
            self._ray_state = None      # nor on the ray that end sat on
            detail = ev.get("detail", "")
            self.pub.log("warn" if detail else "info",
                         f"retreat: {detail}" if detail
                         else "retreat complete — cells re-plannable")
        else:
            self.pub.log("error", f"retreat failed: {ev.get('detail', '')} — "
                                  f"the arm could not leave this stance")
        if self._exit_after:
            self._shutdown()
        else:
            self._set_phase("idle", None)
        self._publish_views()

    # ------------------------------------------------------------- workers
    def _plan_worker(self, gen, target):
        # I3: this runs on a worker thread. `face_find`/`face_unreachable`
        # ride the `plan_done` event instead of touching `self._face_find`/
        # `self._face_unreachable` directly — those are gate state, and the
        # dispatcher is the only thread allowed to mutate it (module
        # docstring). `_on_plan_done` applies them, after its own
        # exit/supersede checks. `self._ray_state`/`self._face_prim` are read
        # the other way round — plain reference reads of dispatcher-owned
        # fields, the same shape as the `_reach` exception.
        face_find = face_unreachable = None
        legs = bounds = res = None
        try:
            if self._object_aabb is not None:
                self._reach = self._cell_map()
            q_now = self.rig.q()
            if target == "survey":
                # The one target that is not a ray: the survey pose is taught,
                # not resolved, so it stays a single planned leg.
                path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                           self.ik.fk(self.q_survey), seed=self.seed)
                detail = "" if path else str(rep.get("reason", rep))
            elif isinstance(target, tuple) and target[0] == "face":
                face = target[1]
                res = resolve_face(self.world, self.ik, T_FLANGE_CAM,
                                   self._face_prim, face,
                                   self.rig.intr["fy"], self.rig.intr["height"],
                                   q_near=q_now)
                if res is None:
                    face_unreachable = face
                    path, detail = None, ("face inaccessible: no safe sample "
                                          "in its ray family")
                else:
                    # What the family ACTUALLY resolved — the shift rung and
                    # the cone put the camera in different places, and the
                    # marker, the record and (Stage C) the prompt all snap to
                    # this one answer.
                    face_find = (face, {
                        "theta": res.theta_deg, "shift": res.shift,
                        "depth": res.depth,
                        "pos": [float(x) for x in res.T_cam[:3, 3]],
                        "pose": np.asarray(res.T_cam, dtype=float)})
                    legs, path, bounds, detail = self._transaction(res, q_now)
                    if path is not None:
                        detail = (f"shift {res.shift:.2f} m, "
                                  f"{res.theta_deg:.0f}deg off, {detail}")
            else:
                # The cell map refreshed at the top of this worker IS
                # `resolve_cell` against the live planner world, run once for
                # the whole grid — read it rather than resolve the one cell a
                # second time, so the accessibility GATE and the plan can
                # never disagree about the same ray.
                res = self._reach.get(target)
                if res is None:
                    path, detail = None, ("cell inaccessible: no safe sample "
                                          "on its ray")
                else:
                    legs, path, bounds, detail = self._transaction(res, q_now)
        except Exception as e:
            legs = bounds = res = None
            path, detail = None, f"planner error: {e}"
        self.events.put({"ev": "plan_done", "gen": gen, "target": target,
                         "path": path, "detail": detail, "legs": legs,
                         "bounds": bounds, "resolution": res,
                         "face_find": face_find,
                         "face_unreachable": face_unreachable})

    def _transaction(self, res, q_now):
        """Plan the three-leg move to `res`. Returns (legs, path, bounds,
        detail); `legs`/`path`/`bounds` are None when the address is
        transition-inaccessible.

        The legs are planned against the LIVE planner world (ruling
        2026-09-17) and carry it home in `Legs.world`, which is the world the
        executor re-validates each leg against. `self._ray_state` is leg 1's
        `current`: the ray the camera stands on, or None after a survey, a
        retreat or a halted move — then there is nothing to slide out along
        and the transaction is legs 2-3.
        """
        legs = plan_transaction(self.world, self.ik, T_FLANGE_CAM,
                                self._ray_state, res, q_now, seed=self.seed)
        if legs is None:
            # The honest class, named (design §Failure semantics): the ray
            # itself resolved — there is just no out-depth from which the hop
            # plans. No fallback substitution.
            return None, None, None, ("transition-inaccessible: no out-depth "
                                      "from which the hop plans")
        path, bounds = _concat_legs(legs, q_now)
        return legs, path, bounds, (f"{len(legs.out)}/{len(legs.hop)}/"
                                    f"{len(legs.slide_in)} leg waypoints at "
                                    f"{res.depth:.3f} m")

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

    def _exec_worker(self, gen, target, path, legs=None):
        # What the arm has actually travelled so far. `path[0]` is where it
        # started (every planner returns its own start config first), and the
        # legs extend it as they complete — so a transaction halted half-way
        # leaves an honest record of where the arm IS, which is what the
        # retreat's retrace tier reverses.
        executed = [np.asarray(path[0], dtype=float)]
        try:
            if legs is None:
                # The survey: one leg, and the rig's own live-world gate,
                # exactly as before rays.
                rep = self.rig.move(path)
                _extend_path(executed, path[1:1 + _done(rep, path)])
                stopped = bool(rep.get("stopped"))
            else:
                stopped = False
                for name, phase in _LEG_PHASES:
                    leg = list(getattr(legs, name))
                    # The slide-out is the one leg that does not carry its own
                    # start: `Legs.out` is the waypoints AFTER q_now, while the
                    # hop and the slide-in each begin on the pose the previous
                    # leg ended at (transaction.py joins them that way).
                    move = ([executed[-1]] + leg) if name == "out" else leg
                    if len(move) < 2:
                        continue        # nothing to travel on this leg
                    # The joins are a CONTRACT, not a convention: a planner
                    # that stopped returning its own start config verbatim
                    # would silently drop the waypoint between two legs —
                    # `_extend_path` would keep both, but the rig would be
                    # commanded from a pose it is not in, and the executed
                    # record would claim a segment nothing travelled. Fail
                    # here, loudly, rather than there.
                    assert np.allclose(move[0], executed[-1]), (
                        f"leg {name!r} starts at {np.asarray(move[0])}, but "
                        f"the arm is at {executed[-1]} — the legs do not join")
                    self.events.put({"ev": "phase", "gen": gen, "phase": phase})
                    # Each leg is judged by the world it was PLANNED under —
                    # the retreat idiom (`rigs.py:move`), which is the one
                    # place in this file that already names its own gate
                    # world instead of taking the rig's "live".
                    #
                    # Today that buys no protection: `legs.world` IS
                    # `self.world` IS `rig.world` (app.py wires one
                    # `RobotCell` to all three), and `_recenter` refits it in
                    # place, so a refit landing between plan and motion would
                    # veto here exactly as the live gate would. What this
                    # does buy is the SEAM: the leg carries the identity of
                    # its own validation world, so giving the transaction a
                    # world snapshot later is a change in one place rather
                    # than a change to every move site. Snapshotting is a
                    # design question, not this task's.
                    rep = self.rig.move(move, world=legs.world)
                    _extend_path(executed, move[1:1 + _done(rep, move)])
                    if rep.get("stopped"):
                        stopped = True
                        break
        except Exception as e:
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "fault", "detail": str(e),
                             "executed": executed})
            return
        if stopped:
            self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                             "outcome": "stopped", "detail": "",
                             "executed": executed})
            return
        self.events.put({"ev": "exec_done", "gen": gen, "target": target,
                         "outcome": "done", "detail": "",
                         "executed": executed})
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
                {"method": "face" if isinstance(target, tuple)
                                     and target[0] == "face" else VIEW_METHOD,
                 # ["face", name] — ONE identity for a face step, matching
                 # the ViewState candidate address (`_view_state` below) and
                 # what a mover/agent sends to request/confirm it (Anton
                 # 2026-09-14 ruling). Was `[target[1]]` (just the name),
                 # which could never round-trip through `Run.at()` against a
                 # candidate address of `["face", name]`.
                 "address": (["face", target[1]] if isinstance(target, tuple)
                             and target[0] == "face"
                             else None if target == "survey" else list(target))})
            cap = self.rig.capture(sid, sdir)
            # `settle_capture` (run/settle.py) is the object-identity + plane-
            # gate + fuse pipeline, verbatim out of this worker: it rejects a
            # bad view BEFORE anything from it is kept (no accumulator touch,
            # no capture publish, no cell marked visited) and fuses into
            # `self.acc` on success. `settle_done ok=False` is the same path
            # a failed capture takes below — the run carries on either way.
            # Pose confirmation for the plane-gate waiver (settle.py): the
            # stamped joints match the path's planned endpoint -> the arm
            # stood where we sent it, so the stamp cannot be the frozen-RTDE
            # failure the gate exists for. 0.02 rad (~1.1 deg) is far above
            # controller settle wobble and far below the >=30 deg between
            # cells a stale q would show. For a transaction the endpoint is
            # the SLIDE-IN's last waypoint — the view pose the whole three
            # legs exist to reach (it is also `path[-1]`, said out loud so a
            # future leg order cannot silently re-point the check).
            q_end = legs.slide_in[-1] if legs is not None else (
                path[-1] if path is not None and len(path) else None)
            arrived = bool(q_end is not None and
                           float(np.max(np.abs(np.asarray(cap["q"], float) -
                                               np.asarray(q_end, float))))
                           < 0.02)
            res = settle_capture(cap, self.rig, self.segmenter, self.acc,
                                 is_survey=(target == "survey"),
                                 on_warn=lambda msg: self.pub.log("warn", msg),
                                 floor_z=self._floor_z,
                                 pose_confirmed=arrived)
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
            # `_recenter` refits the object AND refreshes the cell map on the
            # new belief box — it has to, the marker sphere's radius is the
            # mean of what those rays resolved. This is the post-fuse
            # refresh the module docstring's `_reach` exception covers.
            self._recenter()
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
    def _cell_map(self):
        """Resolve every cell's ray against the LIVE planner world.

        Anton's ruling (2026-09-17): rays validate against the same world
        the planner uses — the one carrying `geometry.object_box` — so an
        accessible cell is accessible to the thing that will actually move
        the arm.

        The BELIEF the rays measure (surface run + containment) is the
        freshest boxguard fit when there is one: both the hull and the eight
        corners come from that one prim. `object_box`'s hull is the
        COLLISION box — 2 cm of margin per side, 5 cm floor — and feeding
        those margins to the resolver would push every cell's `clear_m`, and
        so its resolved depth, outward from what the bench measured
        (`ray_replay.box_aabb`: "over-padding it would push every cell ray's
        sensor floor outward"). With no fit yet, the padded hull is the only
        bound there is and is used as the conservative fallback. Reading
        `_face_prim` here is a plain reference read of a dispatcher-owned
        field, the same shape as the `_reach` exception.

        `q_near` is deliberately NOT the arm's current configuration: the
        map is a fact about the CELL, and a stance-dependent one would make
        a cell's resolved depth flicker with wherever the arm happens to be
        standing when the map is refreshed. The gate it feeds is therefore a
        WEAKER promise than a plan is: "some safe q exists for this cell",
        not "the arm can get there from where it stands". Reaching
        `res.T_flange` from `q_now` is the planner's job, and the refusals
        that promise gap produces are what `_plan_fails` counts (the input
        `RETREAT_AFTER_FAILS` watches).
        """
        if self._object_aabb is None or self.acc.centroid is None:
            return {}
        aabb, corners = belief_bounds(self._face_prim, self._object_aabb)
        return cell_map(self.world, self.ik, T_FLANGE_CAM, self.acc.centroid,
                        aabb, self.rig.intr["fy"], self.rig.intr["height"],
                        corners=corners)

    def _mean_depth(self):
        """Mean resolved depth over accessible cells [m], or None.

        The Stage B honesty stopgap (plan Global Constraints): the schema
        keeps ONE radius float and the UI draws ONE marker shell, while
        cells now each resolve their own depth — so both carry the mean of
        what was actually resolved instead of a frozen fill-formula radius.

        None when NOTHING resolved: there is no measurement to report, and
        `ViewState.r` is Optional precisely so that case can say so. The
        marker sphere still needs some radius to be drawn — `_recenter`
        falls back to `DEFAULT_R` for the helper alone and says so in the
        log; the record gets None.
        """
        d = [res.depth for res in self._reach.values() if res is not None]
        return float(np.mean(d)) if d else None

    def _recenter(self):
        center = self.acc.centroid
        # Min-yaw box FIRST, one source of truth for margins — see
        # `geometry.object_box`. The pose carries the yaw the UI also draws.
        # Installed before the cell map below because the rays are resolved
        # against the LIVE planner world (Anton 2026-09-17) — this box is
        # both the obstacle they must clear and the belief they must frame.
        dims, pose = object_box(self.acc.points)
        self.world.set_object("object", dims.tolist(), pose, parent="base")
        self._object_aabb = pose_box_aabb(dims, pose)
        self._reach = self._cell_map()
        # No shell radius is DERIVED any more: a cell is a ray and carries
        # its own resolved depth. The fill formula and the per-run shell
        # floor that used to answer here are gone from the package entirely
        # (Stage D). `self.sphere` survives purely as the cell-
        # frame helper (center, cell_dir, cam_pose) the UI draws markers
        # with, and its radius is the honest Stage B stopgap — the mean
        # depth the rays actually resolved.
        #
        # SUCCESSOR to the invariant that died with the frozen shell: the
        # old radius was derived once and frozen because "a shell that
        # tightened as the cloud grew would silently re-point addresses that
        # have already been visited and photographed". Cell POSES now do
        # move on every refit — both the resolved `res.T_cam` and the
        # nominal `cam_pose` at the new mean radius. Accepted for Stage B:
        # a cell's IDENTITY is its (h, v) ray, which is fixed by the
        # centroid and the grid, not by any radius; what moves is the depth
        # along that ray, and each step's `ViewState` records the pose IT
        # was chosen from, so coverage read across steps stays honest. What
        # a reader must NOT do is compare a cell's pose between two steps
        # and conclude the address changed.
        self._r_mean = self._mean_depth()
        self.sphere = ViewSphere(
            center, r=DEFAULT_R if self._r_mean is None else self._r_mean)
        n_ok = sum(1 for res in self._reach.values() if res is not None)
        if self._r_mean is None:
            self.pub.log("warn", f"0/{len(self._reach)} cells accessible — "
                                 f"no cell resolved, so there is no mean "
                                 f"depth; markers drawn at the default "
                                 f"{DEFAULT_R:.3f} m")
        else:
            self.pub.log("info", f"{n_ok}/{len(self._reach)} cells "
                                 f"accessible, mean resolved depth "
                                 f"{self._r_mean:.3f} m")
        # A property of the RUN's view method, refreshed with the belief it
        # is derived from. The schema still keeps one float (Stage C gives
        # rays their own fields), so the stopgap is what it carries — and
        # None when nothing resolved, never a constant dressed as a mean.
        self.writer.set_view_method_params(
            VIEW_METHOD, r=None if self._r_mean is None else float(self._r_mean))
        # What this refit installed. `_on_view_confirm` copies it into
        # `_stance_box` when a move starts: the box a stance was APPROVED
        # under is the last one installed before its move executed — the
        # world a trapped arm may legally retreat through even after a newer
        # refit has swallowed its stance (see RETREAT_AFTER_FAILS).
        self._box_rev += 1
        self._last_box = (dims.tolist(), pose)
        self._last_box_rev = self._box_rev

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

    def _fit_step(self, sid):
        """Fit primitives to the step's just-saved cloud; record + publish.

        The LIVE half of the one decomposition path (`fit/apply.py` is the
        offline half): the method name comes from the same registry
        (`METHODS`), the record from the same builder (`record_of`), the
        bytes from the same writer (`write_step_decomposition`, via the
        RunWriter door) — so a live run and a backfilled one are
        byte-compatible by construction, and re-running the backfill over a
        live run is a no-op, not a diff.

        Imports are lazy on purpose: `fit/apply.py` pulls in the fit stack
        (open3d behind `boxguard_fit`), and a Supervisor with `fit=None` —
        every unit test, any rig without the fit extra installed — must
        never pay that import.

        Failure policy mirrors `apply_run`, with one live-loop softening: a
        fitter exception is caught, logged LOUDLY, and recorded as an
        abstain whose reason names the error — the slot records failure
        honestly, and one bad fit never kills a run with a robot in it.
        Registry-name honesty (`d.method == self.fit`) is checked the same
        way, but a mismatch logs + abstains here where the backfill crashes:
        the backfill can afford to die on a lying fitter, a live run cannot.

        The publish is best-effort and `getattr`-guarded: the publisher's
        `publish_decomposition` is landing separately, and a pub without it
        (or a stub) just means no live box in the 3D scene — never a lost
        record, which is written first.
        """
        if self.fit is None:
            return
        from inspection.fit.apply import METHODS, record_of
        from inspection.fit.contract import Decomposition
        P = np.asarray(self.acc.points, dtype=float)
        try:
            d = METHODS[self.fit](P)
        except Exception as e:  # noqa: BLE001 — recorded, screamed, loop lives
            log.error("!! step %03d: %s fitter raised %s: %s",
                      sid, self.fit, type(e).__name__, e)
            d = Decomposition(method=self.fit, version="error", abstain=True,
                              reason=f"fitter error: {type(e).__name__}: {e}")
        else:
            if d.method != self.fit:
                log.error("!! step %03d: registry name %r != produced "
                          "Decomposition.method %r — recording an abstain",
                          sid, self.fit, d.method)
                d = Decomposition(
                    method=self.fit, version="error", abstain=True,
                    reason=f"method mismatch: registry {self.fit!r} produced "
                           f"{d.method!r}")
        rec = record_of(sid, len(P), d)
        if not d.abstain and d.method == "boxguard" and d.primitives:
            self._update_face_gate(d.primitives[0])
        self.writer.write_decomposition(rec)
        publish = getattr(self.pub, "publish_decomposition", None)
        if publish is not None:
            try:
                publish(rec.model_dump(mode="json"))
            except Exception:
                log.exception("decomposition publish failed for step %s", sid)

    def _update_face_gate(self, prim):
        """Feed one boxguard fit to the stability gate (design §gate).

        Called from `_fit_step` on the dispatcher for every non-abstain
        boxguard fit. Two consecutive refit deltas within tolerance open
        the gate; any jump closes it and voids cached resolutions — the
        box moved, so every achieved theta is stale.
        """
        from inspection.view.faceview import (FACE_STABLE_TOL_DEG_DEFAULT,
                                              normals_delta_deg)
        if (self._face_prim is not None
                and normals_delta_deg(self._face_prim, prim)
                <= FACE_STABLE_TOL_DEG_DEFAULT):
            self._face_stable = min(self._face_stable + 1, 2)
        else:
            self._face_stable = 0
            self._face_find.clear()
            self._face_unreachable.clear()
        self._face_prim = prim

    def _face_nominal(self, face):
        """Where a face's camera would stand if nothing were in its way: the
        closest depth on the face's fronto-parallel ray from which the WHOLE
        face stays in frame (`face_family`'s theta=0 member and its own
        containment floor).

        NOT a standoff formula — the per-face framing radius and the fill
        formula under it died with the rest of that family (design §Dies,
        deleted in Stage D). This is one of
        the ray sampler's own stop conditions evaluated alone, with no IK and
        no collision check, so it is a LOWER BOUND on where the arm could
        stand. That is exactly why the marker it feeds is published
        `resolved: false` and drawn dim: it is the geometry's promise, not
        the sampler's answer, and it is replaced by `_face_find` the moment a
        plan resolves the face for real.

        None when the family is empty — a geometry the sampler has nothing
        to say about. This runs on the DISPATCHER for every non-settling
        face on every publish, so it costs that face its marker and its
        recorded pose rather than raising out of `_publish_views` and
        wedging the loop with the arm on the table.
        """
        from inspection.view.faceview import face_cam_pose
        from inspection.view.rays import face_family
        family = face_family(self._face_prim, face, self.rig.intr["fy"],
                             self.rig.intr["height"])
        if not family:
            return None
        ray, _shift, _theta, d_min, rolls = next(
            (m for m in family if m[2] == 0.0), family[0])
        # the member's own base roll, so the top face's dim marker previews
        # the yaw-aligned camera the sampler will actually resolve
        return face_cam_pose(np.asarray(ray.anchor),
                             np.asarray(ray.direction), d_min, rolls[0])

    def _face_states(self):
        """Face menu block: {face: {status, off_angle_deg}} (design §menu)."""
        from inspection.view.faceview import FACES
        out = {}
        gate_open = self._face_stable >= 2
        for f in FACES:
            if f in self.face_visited:
                status = "visited"
            elif not gate_open:
                status = "settling"
            elif f in self._face_unreachable:
                status = "unreachable"
            else:
                status = "ready"
            find = self._face_find.get(f)
            out[f] = {"status": status,
                      "off_angle_deg": None if find is None else find["theta"]}
        return out

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

        An accessible cell's pose is now the RAY's resolved camera pose —
        where the arm would actually stand, at that cell's own depth — not a
        point on a shell of one radius. An inaccessible cell has no resolved
        pose to record, so it keeps the nominal cell frame (marker radius),
        flagged `unreachable` exactly as before.
        """
        cands = []
        if self.sphere is not None:
            for cell in self.sphere.cells():
                res = self._reach.get(cell)
                if res is None:
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
                    pose=_mat(self.sphere.cam_pose(*cell, 0.0)
                              if res is None else res.T_cam),
                    roll=None if res is None else float(res.roll)))
        if self._face_prim is not None:
            for f, st in self._face_states().items():
                if st["status"] == "settling":
                    continue
                # The pose a face candidate records is the one the last plan
                # RESOLVED for it; before any plan has run it is the ray's
                # containment floor (`_face_nominal`), which is a bound, not
                # a promise — the same distinction the UI draws dim.
                find = self._face_find.get(f)
                T_f = self._face_nominal(f) if find is None else find["pose"]
                theta = st["off_angle_deg"]
                # `pose` is Optional in the schema precisely so an address
                # with no pose to report can say so (a face whose ray family
                # is empty) instead of recording an invented one.
                cands.append(ViewCandidate(
                    address=["face", f],
                    # `_face_states()` speaks "ready"/"unreachable"/"visited";
                    # ViewCandidate.status is the closed cell vocabulary
                    # (available/visited/blocked/unreachable/current) — a
                    # "ready" face IS an "available" candidate by that
                    # vocabulary's own meaning (reachable, not yet visited),
                    # so it is recorded under that name rather than widening
                    # the schema for one new synonym.
                    status=_FACE_STATUS_TO_CANDIDATE[st["status"]],
                    pose=None if T_f is None else _mat(T_f),
                    scores={} if theta is None
                    else {"off_angle_deg": float(theta)}))
        mn_mx = self.acc.aabb(pct=BOX_PCT)
        centroid = self.acc.centroid
        return ViewState(
            step_id=step_id, t=time.time(), candidates=cands,
            centroid=None if centroid is None else _floats(centroid),
            extent=None if mn_mx is None else _floats(mn_mx[1] - mn_mx[0]),
            # Stage B stopgap: the schema's one radius float carries the mean
            # resolved cell depth (plan Global Constraints). None when the
            # last refit resolved nothing — `sphere.r` would be the marker
            # fallback there, which is not a measurement.
            r=None if self._r_mean is None else float(self._r_mean),
            chosen=self._target_json(target), decider=self._decider)

    def _set_phase(self, phase, target=None):
        self.phase, self.target = phase, target
        if phase in ("idle", "fault", "done"):
            # Nothing is in flight, so there is no plan to describe — and a
            # retained strip still carrying the last move's boundaries would
            # describe the NEXT thing the UI drew with them. `_target_res`
            # rides with them rather than outliving them: it is what installs
            # `_ray_state` on arrival, and a plan's three products must go
            # stale together or a reader may assume the survivor is fresh.
            self._legs = self._leg_bounds = self._target_res = None
        total = sum(1 for r in self._reach.values() if r is not None)
        self.pub.status(phase=phase, target=self._target_json(target),
                        visited=len(self.visited), total=total,
                        legs=self._leg_bounds)
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
        is_face = isinstance(self.target, tuple) and self.target[0] == "face"
        cell_target = self.target if (isinstance(self.target, tuple)
                                      and not is_face) else None
        pending = cell_target if self.phase == "planning" else None
        previewing = cell_target if self.phase == "previewing" else None
        self.pub.publish_views(
            self.sphere, self._reach, visited=self.visited, blocked=self.blocked,
            current=self.current, captures=self.captures, pending=pending,
            previewing=previewing, survey=survey_state,
            faces=self._faces_display())

    def _faces_display(self):
        """`_face_states()` enriched for the UI: the in-flight face target
        overlaid as pending/previewing (phase-derived, exactly like cells —
        the gate vocabulary itself never holds transient states), and the
        camera endpoint attached to every non-settling face so the operator
        sees WHERE the arm would go before approving. Once a plan has
        resolved the face's ray family, `pos` is its resolved endpoint
        (`resolved: true` — the exact point, cleared with θ on gate
        regression); before that it is the ray's containment floor
        (`_face_nominal`, `resolved: false`, drawn dim — Anton, bench: never
        present a hypothetical point as the truth).
        """
        faces = self._face_states()
        if (isinstance(self.target, tuple) and self.target[0] == "face"
                and self.phase in ("planning", "previewing")):
            faces[self.target[1]]["status"] = (
                "pending" if self.phase == "planning" else "previewing")
        if self._face_prim is not None:
            for f, st in faces.items():
                if st["status"] == "settling":
                    continue
                find = self._face_find.get(f)
                if find is not None:
                    st["pos"] = list(find["pos"])
                    st["resolved"] = True
                    continue
                T_f = self._face_nominal(f)
                if T_f is None:     # no family, no point — and so no marker
                    continue
                st["pos"] = [float(x) for x in T_f[:3, 3]]
                st["resolved"] = False
        return faces

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
        # The SAME box the planner was told (`_recenter`) — this used to be a
        # third copy of the math, and an untrimmed one: the UI drew a raw-
        # AABB box while the planner avoided a pct-trimmed one.
        dims, pose = object_box(self.acc.points)
        self.pub.publish_object(self.acc.points, dims.tolist(), pose[:3],
                                yaw=pose[5])

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
        # Face target: ["face", "x+"] from the bus (UI's face panel), or
        # "face:x+" from a marker id (parity with the "h07v1" cell markers
        # below) — both name one of faceview.FACES, never the cell grid.
        from inspection.view.faceview import FACES
        if (isinstance(raw, (list, tuple)) and len(raw) == 2
                and raw[0] == "face" and raw[1] in FACES):
            return ("face", raw[1])
        if isinstance(raw, str) and raw.startswith("face:") \
                and raw[5:] in FACES:
            return ("face", raw[5:])
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


# --- the move transaction, in the machine's terms ---------------------------

def _done(rep, move) -> int:
    """How many of `move`'s waypoints the rig reports as reached.

    A missing count is a FAULT, not a default. Both rigs in tree report it
    (`rigs.py`, `motion/execute.py`), and there is no safe way to guess at
    one that does not: this number is where the executed record — and so the
    retreat's retrace — believes the arm is. Guess LOW and the retrace opens
    with a segment from a configuration the arm has already left; guess HIGH
    and it drives forward through the waypoint the stop existed to prevent.
    Raising instead puts the run in `fault` with the reason on the record,
    which is the outcome an operator can act on.
    """
    if "waypoints_done" not in rep:
        raise RuntimeError("rig report carries no waypoints_done — cannot "
                           "say where the arm stopped")
    return int(rep["waypoints_done"])


def _extend_path(path, waypoints):
    """Append `waypoints` to `path`, dropping any that repeat its last one."""
    for q in waypoints:
        q = np.asarray(q, dtype=float)
        if not np.array_equal(q, path[-1]):
            path.append(q)
    return path


def _concat_legs(legs, q_now):
    """The transaction as ONE waypoint list, plus where each leg ends in it.

    `transaction.py` joins its legs by repeating a waypoint: the hop starts
    on the pose the slide-out ended at and the slide-in starts on the pose
    the hop landed on. The repeats are dropped here, because this list is
    what the preview replays and what the retreat retraces in reverse, and
    in both a duplicated waypoint reads as the arm standing still.

    Bounds are the INDEX OF EACH LEG'S LAST WAYPOINT, `[out, hop, slide_in]`,
    so an empty slide-out is `bounds[0] == 0` and the legs slice back out as
    `path[:b0+1]`, `path[b0:b1+1]`, `path[b1:b2+1]`.
    """
    path, bounds = [np.asarray(q_now, dtype=float)], []
    for leg in (legs.out, legs.hop, legs.slide_in):
        _extend_path(path, leg)
        bounds.append(len(path) - 1)
    return path, bounds


def _ray_state_of(res):
    """The `RayState` for the ray a `RayResolution` was resolved on.

    The ray is read back off the resolution rather than rebuilt from the
    grid: a cell ray's anchor is the centroid AT PLAN TIME, and by the time
    this runs the arrival's own refit has moved it. `face_cam_pose` points a
    camera's +X at the anchor, so the direction is `-T_cam[:3, 0]` and the
    anchor is `depth` back along it — one reconstruction for cell rays and
    face-family rays alike, because both are posed by that one helper.
    """
    T = np.asarray(res.T_cam, dtype=float)
    direction = -T[:3, 0]
    return RayState(Ray(tuple(T[:3, 3] - res.depth * direction),
                        tuple(direction)),
                    res.depth, res.roll, res.samples)


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
