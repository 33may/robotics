#!/usr/bin/env python3
"""Live execution: the orchestrator as a second mouse.

The `brain` does not get its own motion path. It puts the SAME command on the
SAME queue the browser's buttons put theirs on (`run/app.py`'s command pump →
`Supervisor.events`), and then waits on the phase machine. Anton, 2026-08-25.

**The approval gate needs no new code.** `Supervisor._on_view_confirm`
(machine.py:314) drops any confirm unless `phase == "previewing"` and the
target matches, and nothing here ever sends one. So:

    brain may press `view/request`.   Only a human may press `view/confirm`.

That is the whole gate. It stays true for as long as no code here sends a
confirm, which is a property you can check by grepping this file. When the
chain has earned authority, the change is to send a confirm from here — and at
that moment `view/confirm` needs the generation counter it currently does not
carry, because there would finally be a second confirm source.

The generation counter already protects the other direction: every
`view/request` bumps `Supervisor.gen` and workers carry the gen they were
spawned with, so a stale agent request cannot move the arm — the same
mechanism that stops a stale button press.

Phases are polled rather than hooked. The dispatcher owns every bit of run
state and publishes transitions itself; reaching in with a callback would add
a second writer to the thing whose single-writer property is the reason the
state machine is trustworthy.
"""
import threading
import time

from inspection.record.run import Run

POLL_S = 0.05
#: How long to wait for the planner before giving up on a request. Approval
#: itself is NOT timed out — a human thinking is not a failure.
PLAN_TIMEOUT_S = 90.0

#: EXACTLY `Supervisor._BUSY_PHASES` (machine.py:63). `planning` and
#: `previewing` are deliberately NOT here: the state machine accepts a request
#: in both — a new one supersedes the pending plan — so refusing to send during
#: them is a restriction we invented, and it made the orchestrator spin
#: retrying a move it was never allowed to make.
_BUSY = ("executing", "capturing", "fusing", "fault")

#: How long the dispatcher gets to pick a command off the queue before silence
#: counts as a drop. The request lands on a queue read by another thread, so
#: sampling the phase immediately after `put()` samples the state BEFORE the
#: command was seen — which read as "refused" while a planner was in fact
#: starting, and stranded a preview nobody was waiting on.
ACK_GRACE_S = 3.0


class Refused(Exception):
    """The request never became a preview. Carries a categorical reason."""


class SupervisorMover:
    """`move(cell)` against a real Supervisor, gated on operator approval.

    Returns a categorical outcome rather than raising, because a refusal is
    information the orchestrator should reason about, not an error that ends
    the run (LLM3 `2403.11552`: naming the failure class raised success 40% ->
    60% AND cut retries).
    """

    def __init__(self, sup, run_dir, on_event=None, plan_timeout=PLAN_TIMEOUT_S):
        self.sup = sup
        self.run_dir = run_dir
        self.on_event = on_event or (lambda *a, **k: None)
        self.plan_timeout = plan_timeout

    # ------------------------------------------------------------ helpers
    def _wait(self, predicate, timeout=None):
        """Poll until `predicate(phase, target)` is true. None on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            phase, target = self.sup.phase, self.sup.target
            if predicate(phase, target):
                return phase, target
            if deadline is not None and time.monotonic() > deadline:
                return None
            time.sleep(POLL_S)

    def nav(self):
        """What the arm can actually do from here, straight off the Supervisor.

        These are the SAME three sets the 3D markers are coloured from
        (`ui/publisher.py:_cell_state`): reachability cached by the plan/exec
        workers, cells already visited, and cells the planner refused from the
        current stance. Before this existed the orchestrator guessed, and in
        the first live run it spent two of its three moves being refused.

        `blocked` is deliberately soft — the Supervisor clears it on every real
        move, because "no path from where the arm is standing" can become
        reachable one move later.
        """
        return {
            "reachable": {c for c, r in self.sup._reach.items() if r is not None},
            "visited": set(self.sup.visited),
            "blocked": set(self.sup.blocked),
        }

    def _captures(self):
        """How many frames exist on disk. Zero before the first one.

        Deliberately tolerant: `Run.load` needs `run.json`, which does not
        exist until the survey has been captured — and the survey is now the
        first thing this class is asked to do.
        """
        try:
            return len(Run.load(self.run_dir).captured)
        except Exception:                            # noqa: BLE001
            return 0

    # --------------------------------------------------------------- move
    def request(self, cell):
        """Ask for a viewpoint and block until it is reached, refused, or
        superseded. Returns `(ok: bool, reason: str)`."""
        cell = "survey" if cell == "survey" else tuple(cell)

        if self.sup.phase in _BUSY:
            return False, (f"the arm is {self.sup.phase} right now — the "
                           f"request was not sent. Ask again once it settles.")

        before = self._captures()
        self.on_event("requested", cell=_json(cell))
        # Exactly the message the browser's button sends. No private API, no
        # second path into the robot.
        self.sup.events.put({"cmd": "view/request", "target": _json(cell)})

        # 1a. Wait for the dispatcher to ACK by leaving idle. Silence for the
        #     whole grace means the request was dropped on the floor (bad
        #     target, unreachable, off-sphere, already visited) — the
        #     Supervisor logs the reason on `log/events`.
        if self._wait(lambda p, t: p != "idle" or t is not None,
                      timeout=ACK_GRACE_S) is None:
            return False, (f"{_json(cell)} was refused — unreachable, "
                           f"off-sphere, or already visited. The arm has not "
                           f"moved.")

        # 1b. planning -> previewing on OUR cell, or back to idle (the plan
        #     failed and the dispatcher gave up).
        got = self._wait(
            lambda p, t: (p == "previewing" and t == cell) or
                         (p == "idle" and t is None),
            timeout=self.plan_timeout)
        if got is None:
            return False, "the planner did not answer in time; nothing moved."
        if got[0] != "previewing":
            # The honest statement is the class of failure, not a guess.
            return False, (f"{_json(cell)} could not be planned — no collision-"
                           f"free path was found. The arm has not moved.")

        # 2. the human gate. Deliberately un-timed.
        self.on_event("awaiting_approval", cell=_json(cell))
        got = self._wait(lambda p, t: not (p == "previewing" and t == cell))
        phase, target = got

        if phase == "previewing" and target != cell:
            # Anton pressed a different cell. That is not an error, it is a
            # correction — and telling the orchestrator so puts the operator's
            # judgement into its context where it can act on it.
            self.on_event("redirected", cell=_json(cell), to=_json(target))
            return False, (f"the operator did not approve {_json(cell)} and "
                           f"selected {_json(target)} instead. Consider going "
                           f"there, or explain why {_json(cell)} matters.")
        if phase != "executing":
            self.on_event("cancelled", cell=_json(cell), phase=phase)
            return False, (f"{_json(cell)} was not approved (run went to "
                           f"{phase}). The arm has not moved.")

        # 3. executing -> capture -> fuse -> idle
        self.on_event("approved", cell=_json(cell))
        got = self._wait(lambda p, t: p in ("idle", "fault", "done"))
        if got[0] == "fault":
            return False, f"the move to {_json(cell)} faulted; the run is halted."

        if self._captures() <= before:
            return False, (f"the arm reached {_json(cell)} but no frame was "
                           f"captured. Nothing to inspect there.")
        self.on_event("captured", cell=_json(cell))
        return True, "reached and captured"


def make_ask_handler(sup, pub, run_dir, cognition):
    """Build the `brain/ask` handler shared by the real run and the mock.

    One implementation on purpose: the mock is how the approval gate gets
    rehearsed without hardware, and a rehearsal against different wiring
    rehearses nothing. `cognition` is therefore the ONLY thing that differs
    between the two — `real_cognition()` at `run/app.py:107`, `stub_cognition()`
    at `ui/mock.py:93` — and this handler chooses no model of its own. Until
    2026-08-26 it picked one from `GEMINI_API_KEY`, so an unset key turned the
    demo into a stub run that was indistinguishable from the real thing.
    """
    import asyncio
    import threading

    running = threading.Event()

    def ask(question):
        from inspection.brain.loop import Brain

        question = str(question or "").strip()
        if not question:
            pub.log("warn", "brain/ask with no question")
            return
        if running.is_set():
            pub.log("warn", "a question is already running")
            return
        running.set()

        # The trace is opened HERE, not inside Brain, because the survey is
        # part of the run and has to be visible while it happens: the question
        # appears in the panel the moment it is asked, and the survey's
        # approval beats land under it. Brain appends to this same writer.
        from inspection.brain.loop import DEFAULT_MODEL
        from inspection.brain.trace import TraceWriter

        trace = TraceWriter(run_dir)
        # `cognition` is on the header event because a mock run writes a trace
        # of exactly the same shape as a real one — same wiring, by design —
        # so without the label a screenshot of one is a screenshot of either.
        trace.event("run", question=question, model=DEFAULT_MODEL,
                    cognition=cognition.label, run_dir=str(run_dir),
                    live=True, captured=0)

        def on_event(state, **kw):
            trace.event("approval", state=state, **kw)
            pub.log("info", f"brain: {state} {kw.get('cell', '')}")

        mover = SupervisorMover(sup, run_dir, on_event=on_event)
        TraceWatcher(pub, run_dir).start()

        def go():
            try:
                # THE SURVEY IS STEP ONE OF ASKING (Anton 2026-08-25). The
                # operator should not have to remember a prerequisite: the
                # question starts the whole run. It is still a motion, so it
                # goes through the same approval gate as every other move.
                if not has_survey(run_dir):
                    trace.event("text", text="Surveying the scene first.")
                    ok, reason = mover.request("survey")
                    if not ok:
                        trace.event("text", text=f"no survey: {reason}")
                        pub.log("warn", f"survey not completed: {reason}")
                        return
                brain = Brain(run_dir, question, vlm=cognition.vlm,
                              verbs=cognition.verbs, mover=mover, trace=trace)
                asyncio.run(brain.run())
            except Exception:                        # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("brain run failed")
                trace.event("text", text="the run failed — see the console")
                pub.log("error", "brain run failed — see the console")
            finally:
                running.clear()

        threading.Thread(target=go, name="brain", daemon=True).start()
        # Labelled here too: the trace header renders question/model/captured
        # only (`ui/src/panels/TracePanel.tsx:768`), so the log line is the one
        # place the operator can see WHICH eyes this run is using today.
        pub.log("info", f"brain[{cognition.label}]: {question}")

    return ask


class TraceWatcher(threading.Thread):
    """Republish `<run>/eyes/trace.jsonl` whenever it grows.

    The same mechanism `ui/app.py trace` uses, so a live run and a replayed
    one reach the panel by exactly one path: the file is the interface.
    """

    def __init__(self, pub, run_dir, poll=0.4):
        super().__init__(name="trace-watch", daemon=True)
        self.pub, self.run_dir, self.poll = pub, run_dir, poll
        self.stop = threading.Event()

    def run(self):
        from inspection.brain.trace import TRACE_NAME, read_trace
        path = self.run_dir / "eyes" / TRACE_NAME
        stamp = None
        while not self.stop.wait(self.poll):
            try:
                if not path.exists():
                    continue
                now = path.stat().st_mtime_ns
                if now != stamp:
                    stamp = now
                    self.pub.publish_trace(read_trace(self.run_dir))
            except Exception:                       # noqa: BLE001
                pass          # a broken trace must never disturb a live run


def _json(target):
    return list(target) if isinstance(target, tuple) else target


def has_survey(run_dir):
    """True once a survey frame exists on disk.

    The survey is the brain's opening view — it is what the first prompt
    describes — so a run without one has nothing to reason from.
    """
    try:
        return Run.load(run_dir).survey is not None
    except FileNotFoundError:
        return False
