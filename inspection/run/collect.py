#!/usr/bin/env python3
"""Flow B: the data-collection sweep driver.

`SweepDriver` walks a viewsphere the way an operator would from the UI —
one `SupervisorMover.request` at a time, through the SAME human approval
gate `Supervisor._on_view_confirm` enforces for everyone else (machine.py's
module docstring: `view/request` is the only door in). What is different is
WHICH cell it asks for and in what order: every open candidate is costed by
`planner` and the cheapest is requested first, so a sweep visits the whole
shell in roughly the least total joint travel rather than in cell-id order.
`--random` swaps that order for a seeded shuffle (`DECIDER_RANDOM` on every
step): nearest-first grows the cloud from one side outward, which is its own
special step ordering — random order is the baseline for reasoning about how
per-step estimates converge in a real run.

The driver never touches `sup.world`: that `RobotCell` is the dispatcher
thread's alone (machine.py's docstring — "exactly one action worker alive
at a time" is a property of the SAME object, and this thread is not one of
the workers the dispatcher already serialises against). So `SweepDriver`
carries its OWN `RobotCell`/`UR5eIK` purely for costing candidates before it
asks; the actual move is planned for real, on the real world, inside the
Supervisor once `mover.request(cell)` sends `view/request` — this thread's
own plan is thrown away, it exists only to rank.
"""
from __future__ import annotations

import logging
import random
import threading

import numpy as np

from inspection.brain.live import SupervisorMover
from inspection.cell.geometry import object_box
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.motion.transaction import plan_transaction
from inspection.view.rays import belief_bounds, resolve_cell
from inspection.view.viewsphere import T_FLANGE_CAM

#: Who this driver's mover tells the record chose the view. Rides every
#: `view/request` via `SupervisorMover.decider` and lands in the step's
#: `ViewState.decider` — the only place on disk that says a sweep, not a
#: human or the AI orchestrator, picked this cell.
DECIDER = "sweep-shortest"

#: `--random`'s decider string. A separate name, not a tag on the run: the
#: per-step `ViewState.decider` is the on-disk statement of WHICH policy
#: picked each cell, and convergence analysis over these datasets dispatches
#: on it (Anton, 2026-09-14: shortest-path step order flatters the box
#: estimate's convergence — random order is the honest baseline).
DECIDER_RANDOM = "sweep-random"

#: `--auto`'s dwell between "previewing" and the driver's own confirm: the
#: 3D view shows the chosen cell and the planned path for this long before
#: the arm moves, so the operator watching the cell still sees every move
#: coming — the approval is automatic, the transparency is not.
AUTO_DWELL_S = 1.0


def joint_l1(path) -> float:
    """Cost of a planned path: sum of |dq| over consecutive waypoints."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.abs(np.diff(path, axis=0)).sum())


def rank_candidates(cands, planner):
    """`cands` ordered cheapest-first by `planner(cell) -> cost | None`.

    Pure — no I/O, no motion, nothing about the machine. A candidate whose
    planner returns None never ranks at all: it is dropped, not sorted to
    the back, because "unplannable" is not merely "the worst option".
    """
    scored = [(cost, c) for c in cands for cost in (planner(c),)
             if cost is not None]
    scored.sort(key=lambda sc: sc[0])
    return [c for _, c in scored]


def _mirror_object_box(world, acc) -> None:
    """Copy the object's collision box from the accumulator onto `world`.

    Read-only on `acc`, and `world` is the driver's OWN `RobotCell`, never
    `sup.world`. Uses `geometry.object_box` — the SAME function
    `machine._recenter` calls — so this thread's ranking plans agree with
    what the Supervisor's real planner will see when it plans for real.
    """
    pts = acc.points
    if pts is None or not len(pts):
        return
    dims, pose = object_box(pts)
    world.set_object("object", dims.tolist(), pose, parent="base")


def _default_planner(sup, seed: int = 0, hop_planner=plan_viewpoint):
    """The default `(world, ik, q_now, cell) -> (path | None, cost)`.

    Ranks a candidate the way the REAL move will be planned: `resolve_cell`
    finds the cell's closest safe sample on THIS thread's own world (never
    `sup.world`), then `plan_transaction` plans the three-leg move to it
    from `q_now` — slide-out included, `current=sup._ray_state`, the same
    ray the dispatcher hands `machine._transaction` at request time. That
    docstring claim used to be false: passing `current=None` ranked a
    DIFFERENT search (candidates `entries x 1` instead of
    `entries x (1 + out_chain)`) at a cost missing the slide-out leg, and a
    cell that only transacts once the arm has slid outward ranked
    unplannable — enough to call a coverable shell stuck.
    Cost is `joint_l1` over the whole trip, `q_now` followed by every leg's
    waypoints in order (`plan_transaction`'s own smoke test concatenates
    a transaction the same way) — `hop`/`slide_in` each repeat the seam
    they continue from, a zero-length edge that costs nothing.

    Replaced the KNOWN-DEGRADED shell ranker (Stage B task 4, ray-views;
    deleted in Stage D): that one planned a ladder move to a point on a
    shell radius that is merely the MEAN of the rays' resolved depths,
    disagreeing by construction with the accessibility gate now that cells
    are rays. A
    cell whose ray does not resolve, or whose transition does not plan,
    costs `(None, None)` — dropped by `rank_candidates`, never crashed on.

    Reads `sup.acc.centroid`/`sup._object_aabb`/`sup._face_prim`/
    `sup._ray_state`/`sup.rig.intr` only — never `sup.world`/`sup.ik` — and
    derives the belief through `rays.belief_bounds`, the ONE reduction
    `machine._cell_map` also calls: the freshest boxguard fit's own
    UNPADDED hull when there is one, the padded `_object_aabb` fallback
    with `corners=None` when there is not.
    `corners` feeds `resolve_cell`'s containment floor for a grid CELL
    unconditionally — a fit tightens exactly the belief this ranker
    measures, not merely a face's. Skipping it (as this used to) costs
    every candidate the fit's whole margin of standoff it would not
    actually have to pay — real ranking/execution mismatch, the same
    shape as the shell-vs-ray one this task replaces.

    All five fields are replaced wholesale (`machine._recenter`'s
    `_object_aabb` assignment, `_update_face_gate`'s `self._face_prim =
    prim`, `_on_exec_done`'s `self._ray_state = ...` — a `RayState`
    namedtuple, immutable, never edited in place — and
    `CloudAccumulator.add`'s `_points` reassignment on every fuse),
    never mutated in place, so a snapshot taken here cannot change
    underneath one `resolve_cell`/`plan_transaction` call. Reading
    `_face_prim` off the dispatcher from this thread is sanctioned the
    same way `_reach` is (machine.py's own `_plan_worker` reads it the
    identical way, its docstring: "a plain reference read of a
    dispatcher-owned field, the same shape as the `_reach` exception").
    `hop_planner` rides straight through to `plan_transaction`'s own
    `planner=` (default `plan_viewpoint`, the real ladder) — injectable
    so a caller can stub the hop the way `test_transaction.py`'s
    `_StubPlanner` does, without faking `plan_transaction` itself.
    """
    def planner(world, ik, q_now, cell):
        center, fallback = sup.acc.centroid, sup._object_aabb
        if center is None or fallback is None:
            return None, None
        aabb, corners = belief_bounds(sup._face_prim, fallback)
        h, v = cell
        target = resolve_cell(world, ik, T_FLANGE_CAM, center, aabb, h, v,
                              sup.rig.intr["fy"], sup.rig.intr["height"],
                              corners=corners)
        if target is None:
            return None, None
        legs = plan_transaction(world, ik, T_FLANGE_CAM, sup._ray_state,
                                target, q_now, seed=seed,
                                planner=hop_planner)
        if legs is None:
            return None, None
        path = [np.asarray(q_now, dtype=float), *legs.out, *legs.hop,
               *legs.slide_in]
        return path, joint_l1(path)
    return planner


class SweepDriver(threading.Thread):
    """Flow B: survey, then work the whole shell cheapest-path-first.

    Survey first (`mover.request("survey")` — the object has to exist
    before there is a shell to sweep). Then, until no open candidate is
    left: rank every reachable/unvisited/unblocked cell by `planner`,
    request the cheapest (or, with `random_order`, shuffle them and request
    in that order — same pass structure, same refusal handling, no ranking). A refusal (unreachable, unplannable, not
    approved) tries the next-cheapest candidate from the SAME ranked pass —
    replanning every candidate again would be wasted work when only one of
    them was refused. A redirect (the operator approved a different cell
    instead) is not a refusal to route around: it is a correction, so the
    driver abandons the rest of this ranked pass and starts over from
    wherever the arm now is. The pass also restarts, naturally, once a
    request succeeds — `visited`/`blocked` have changed and the candidate
    set needs recomputing anyway.

    `self.finished` is True only once the candidate set is genuinely empty
    — the honest signal the composition root closes the run on
    ("completed" vs "aborted", see `run/app.py:collect`). A stuck pass
    (every candidate's plan refused - `ranked` comes back empty though
    `cands` is not) or a failed survey leave it False: something stopped
    this run short of covering the shell, and pretending otherwise would
    lose that from the record.

    ENDING. The driver is not the run's owner and cannot be joined by one
    that has already torn its hardware down, so it watches for two ways out
    and checks BOTH before every request: `stop()` (the composition root's
    teardown, `run/app.py:collect`) and a Supervisor that has reached
    `fault`/`done`. Without them, a faulted run leaves this thread ranking
    an unchanged candidate set forever — every request refused instantly by
    `SupervisorMover`'s own busy check — hammering `resolve_cell`/
    `plan_transaction` and `rig.q()` on hardware the teardown is in the
    middle of closing.
    """

    def __init__(self, sup, run_dir, on_event=None, planner=None,
                 auto: bool = False, random_order: bool = False,
                 cells=None):
        super().__init__(name="sweep-driver", daemon=True)
        self.sup = sup
        #: Fixed cell list (calib/depth_eval.py): survey, then exactly these
        #: cells in this order, then `run/finish`. Cells, not joint configs —
        #: the object moves between eval runs and a cell follows it.
        self.cells = [tuple(c) for c in cells] if cells else None
        self.run_dir = run_dir
        self.on_event = on_event or (lambda *a, **k: None)
        self.finished = False
        self._stop_flag = threading.Event()
        #: This thread's OWN collision world/IK — see the module docstring.
        self._world = RobotCell()
        self._ik = UR5eIK()
        self.planner = planner or _default_planner(sup, seed=getattr(sup, "seed", 0))
        #: `--random`: request cells in seeded-shuffled order instead of
        #: cheapest-first. The ranking planner is skipped entirely — the
        #: local plan exists only to rank (module docstring), and a shuffle
        #: needs no rank; plannability is judged where it always is, by the
        #: Supervisor's real planner behind `mover.request`.
        self.random_order = random_order
        self._rng = random.Random(getattr(sup, "seed", 0))
        self._redirected = False

        def _on_mover_event(state, **kw):
            # `SupervisorMover.request` reports a redirect through this same
            # channel — catching it here, rather than pattern-matching the
            # refusal string `request` returns, is what lets the run loop
            # below tell "try the next candidate" apart from "start over".
            if state == "redirected":
                self._redirected = True
            self.on_event(state, **kw)

        self.mover = SupervisorMover(sup, run_dir, on_event=_on_mover_event,
                                     decider=DECIDER_RANDOM if random_order
                                     else DECIDER,
                                     auto_confirm_s=AUTO_DWELL_S if auto
                                     else None)

    def _candidates(self):
        """Reachable, unvisited, unblocked — straight off the Supervisor.

        Read-only, like `SupervisorMover.nav()` reads the same three sets
        from this same thread's counterpart in `brain/live.py` — the
        dispatcher only ever reassigns/mutates these in place, never holds
        a lock over them, so this is the established pattern for a second
        thread to look, not a new race.
        """
        return [c for c, r in self.sup._reach.items()
                if r is not None and c not in self.sup.visited
                and c not in self.sup.blocked]

    def stop(self):
        """Ask the sweep to end at its next check. Idempotent, never blocks.

        The composition root calls this FIRST in its teardown, before the rig
        is closed: the driver reads `rig.q()` and plans against its own world
        on every pass, and neither should be happening while the RTDE
        interface and the camera are being shut down.
        """
        self._stop_flag.set()

    def _run_fixed(self):
        """Request each fixed cell once, in order; a refusal is recorded and
        skipped (the eval scores what was captured). Then end the run."""
        for cell in self.cells:
            if self._should_stop():
                return
            ok, reason = self.mover.request(cell)
            if not ok:
                logging.getLogger(__name__).warning("cell %s refused: %s", cell, reason)
                self.on_event("cell_refused", cell=list(cell), reason=reason)
        self.finished = True
        self.sup.events.put({"cmd": "run/finish"})

    def _should_stop(self) -> bool:
        """Both ways out. `fault`/`done` are terminal for the whole run —
        no request can ever be accepted again — so continuing to rank from
        here is pure spin."""
        return self._stop_flag.is_set() or self.sup.phase in ("fault", "done")

    def run(self):
        if self._should_stop():
            return
        ok, reason = self.mover.request("survey")
        if not ok:
            # LOUD, in the terminal, not only on the bus log topic: a sweep
            # that dies here leaves a UI sitting at "idle 0/?" with a healthy
            # camera, which reads as "nothing started" instead of "the survey
            # was refused" (box7, 2026-09-08: arm parked in an unplannable
            # configuration — the operator had nothing to go on).
            logging.getLogger(__name__).error(
                "SWEEP ABORTED before the survey: %s — the run will sit idle. "
                "Most likely the arm's current pose cannot be planned to the "
                "survey pose (move the arm to a normal front-of-cell pose "
                "and restart).", reason)
            self.on_event("sweep_aborted", reason=reason)
            return
        if self.cells is not None:
            self._run_fixed()
            return
        while True:
            if self._should_stop():
                return                # finished stays False: cut short
            cands = self._candidates()
            if not cands:
                self.finished = True
                return
            if self.random_order:
                # No mirror, no costing, no rank: the local plan exists only
                # to order candidates, and this mode's order is a seeded
                # shuffle. The Supervisor still plans every request for real.
                ranked = list(cands)
                self._rng.shuffle(ranked)
            else:
                _mirror_object_box(self._world, self.sup.acc)
                q_now = np.asarray(self.sup.rig.q(), dtype=float)

                def cost_fn(cell):
                    return self.planner(self._world, self._ik, q_now, cell)[1]

                ranked = rank_candidates(cands, cost_fn)
            if not ranked:
                # Every open candidate's plan refused this pass — nothing
                # left this driver can usefully ask for from here.
                self.on_event("sweep_stuck", candidates=len(cands))
                return
            self._redirected = False
            for cell in ranked:
                # Checked BETWEEN candidates too, not just per pass: a fault
                # or a shutdown mid-pass would otherwise walk the whole ranked
                # list collecting instant refusals before anyone noticed.
                if self._should_stop():
                    return
                ok, reason = self.mover.request(cell)
                if ok or self._redirected:
                    break
            # A pass where every request was refused retries from the top —
            # in BOTH modes. Refusals can be transient (an approval window
            # missed, a mover momentarily busy), and a first cut of --random
            # (2026-09-14) that ended the sweep on an unchanged candidate
            # set died on exactly that race. Hopeless runs are ended by the
            # `_should_stop` checks above, not by second-guessing refusals.
