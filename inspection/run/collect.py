#!/usr/bin/env python3
"""Flow B: the data-collection sweep driver.

`SweepDriver` walks a viewsphere the way an operator would from the UI —
one `SupervisorMover.request` at a time, through the SAME human approval
gate `Supervisor._on_view_confirm` enforces for everyone else (machine.py's
module docstring: `view/request` is the only door in). What is different is
WHICH cell it asks for and in what order: every open candidate is costed by
`planner` and the cheapest is requested first, so a sweep visits the whole
shell in roughly the least total joint travel rather than in cell-id order.

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

import threading

import numpy as np

from inspection.brain.live import SupervisorMover
from inspection.cell.geometry import BOX_PCT
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK

#: Who this driver's mover tells the record chose the view. Rides every
#: `view/request` via `SupervisorMover.decider` and lands in the step's
#: `ViewState.decider` — the only place on disk that says a sweep, not a
#: human or the AI orchestrator, picked this cell.
DECIDER = "sweep-shortest"


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

    Read-only on `acc` — `aabb()` returns fresh numpy arrays, nothing here
    mutates the accumulator — and `world` is the driver's OWN `RobotCell`,
    never `sup.world`. Mirrors `machine.py:_recenter`'s own box math (same
    2 cm margin, same 5 cm floor) so this thread's ranking plans agree with
    what the Supervisor's real planner will see when it plans for real.
    """
    aabb = acc.aabb(pct=BOX_PCT)
    if aabb is None:
        return
    mn, mx = aabb
    dims = np.maximum(mx - mn + 0.04, 0.05)
    mid = (mn + mx) / 2
    world.set_object("object", dims.tolist(), [*mid.tolist(), 0.0, 0.0, 0.0],
                     parent="base")


def _default_planner(sup, seed: int = 0):
    """The default `(world, ik, q_now, cell) -> (path | None, cost)`.

    Reads `sup.sphere` only — never `sup.world`/`sup.ik` — and only to ask
    it for cell geometry. `sup.sphere` is safe to read from this thread
    without a lock: `machine.py:_recenter` always REPLACES it wholesale
    (`self.sphere = ViewSphere(...)`), never mutates one in place, so a
    reference taken here cannot change underneath a single `plan_to_cell`
    call. Planning itself runs entirely against the caller's `world`/`ik`.
    """
    def planner(world, ik, q_now, cell):
        sphere = sup.sphere
        if sphere is None:
            return None, None
        h, v = cell
        path, _rep, _roll = sphere.plan_to_cell(world, ik, q_now, h, v,
                                                 seed=seed)
        return path, (None if path is None else joint_l1(path))
    return planner


class SweepDriver(threading.Thread):
    """Flow B: survey, then work the whole shell cheapest-path-first.

    Survey first (`mover.request("survey")` — the object has to exist
    before there is a shell to sweep). Then, until no open candidate is
    left: rank every reachable/unvisited/unblocked cell by `planner`,
    request the cheapest. A refusal (unreachable, unplannable, not
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
    `SupervisorMover`'s own busy check — hammering `plan_to_cell` and
    `rig.q()` on hardware the teardown is in the middle of closing.
    """

    def __init__(self, sup, run_dir, on_event=None, planner=None):
        super().__init__(name="sweep-driver", daemon=True)
        self.sup = sup
        self.run_dir = run_dir
        self.on_event = on_event or (lambda *a, **k: None)
        self.finished = False
        self._stop_flag = threading.Event()
        #: This thread's OWN collision world/IK — see the module docstring.
        self._world = RobotCell()
        self._ik = UR5eIK()
        self.planner = planner or _default_planner(sup, seed=getattr(sup, "seed", 0))
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
                                     decider=DECIDER)

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
            self.on_event("sweep_aborted", reason=reason)
            return
        while True:
            if self._should_stop():
                return                # finished stays False: cut short
            cands = self._candidates()
            if not cands:
                self.finished = True
                return
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
