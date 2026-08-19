#!/usr/bin/env python3
"""Headless dry-run of the whole loop: real world+IK+planner, fake arm and
camera, scripted decider. Run: p inspection/tests/test_loop_fake.py"""
import json
import tempfile
import threading
from pathlib import Path

import numpy as np

from inspection.motion.plan import DEMO_PARK
from inspection.run.decider import Console, Look, Answer
from inspection.run.loop import Loop
from inspection.tests.synth import synth_capture
from inspection.tests.test_decider import FeedStream


class FakeRig:
    """Arm+camera stand-in honouring the five-member rig contract."""

    def __init__(self, q0):
        self._q = np.asarray(q0, dtype=float)
        depth, intr, scale, T_bc, _ = synth_capture()
        self._depth, self._T_bc = depth, T_bc
        self.intr, self.depth_scale = intr, scale
        self.moves = []

    def q(self):
        return self._q.copy()

    def preview(self, path):
        pass                                    # meshcat replay in real rig

    def move(self, path):
        self.moves.append(len(path))
        self._q = np.asarray(path[-1], dtype=float)
        return {"waypoints_done": len(path) - 1, "s": 0.0,
                "final_err_deg": 0.0, "stopped": False}

    def capture(self, pose_id):
        return {"dir": f"(fake {pose_id:03d})", "rgb": None,
                "depth_raw": self._depth, "T_base_cam": self._T_bc,
                "q": self._q.copy()}


class SyncConsole(Console):
    """Test-only synchronization for scripted loop.run() drives.

    loop.py now calls Console.drain() immediately before every approval
    readline() (Finding 3: stale type-ahead must never approve a motion).
    That means a test that feeds all its answers up front — the old style
    — races drain(): the background reader thread typically queues every
    fed line during boot's OMPL planning, well before the first approval
    prompt is reached, so drain() wipes them all and readline() blocks
    forever.

    Fix: track a monotonically increasing count of readline() calls behind
    a condition variable. A driving thread waits for "count >= N" (a
    resettable Event here would itself race — the driver could observe a
    stale "set" left over from the PREVIOUS prompt before this one clears
    it) before feeding the Nth answer, guaranteeing drain() for that
    prompt has already run — the same "only after seeing the prompt"
    timing a real operator has. No sleeps, no flakiness.
    """

    def __init__(self, stream):
        super().__init__(stream=stream)
        self._cv = threading.Condition()
        self.prompt_count = 0

    def readline(self, prompt=""):
        with self._cv:
            self.prompt_count += 1
            self._cv.notify_all()
        return super().readline(prompt)


class ScriptedFeeder:
    """Feeds `con` one line per call, each only once `con` has actually
    reached that many readline() calls (see SyncConsole)."""

    def __init__(self, con, fs):
        self.con, self.fs = con, fs
        self.n = 0

    def feed(self, line, timeout=10.0):
        self.n += 1
        with self.con._cv:
            ok = self.con._cv.wait_for(
                lambda: self.con.prompt_count >= self.n, timeout)
        assert ok, f"console never reached prompt #{self.n}"
        self.fs.feed(line)


def _run_in_background(loop, timeout=30.0):
    th = threading.Thread(target=loop.run, daemon=True)
    th.start()
    return th


def _join(th, timeout=30.0):
    th.join(timeout)
    assert not th.is_alive(), "loop.run() did not finish in time"


class ScriptDecider:
    """READ: canned comment. DECIDE: same menu cell twice (refused, then
    approved), then answer."""

    def __init__(self):
        self.decisions = 0

    def read(self, cap):
        return f"scripted comment {self.decisions}"

    def decide(self, ctx):
        self.decisions += 1
        if self.decisions in (1, 2):
            assert ctx.menu, "no reachable cells offered"
            return Look(ctx.menu[0].h, ctx.menu[0].v)
        return Answer("scripted: no logo")


def test_full_fake_run():
    q_survey = DEMO_PARK.copy()
    q_start = q_survey.copy()
    q_start[5] += np.radians(8)             # boot must plan+move to survey
    fs = FeedStream()
    con = SyncConsole(stream=fs)
    outdir = Path(tempfile.mkdtemp()) / "run1"
    loop = Loop(FakeRig(q_start), ScriptDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")

    th = _run_in_background(loop)
    feeder = ScriptedFeeder(con, fs)
    feeder.feed("y")                        # boot move: approved
    feeder.feed("n")                        # first look at the cell: refused
    feeder.feed("y")                        # second look, same cell: approved
    _join(th)

    rec = json.loads((outdir / "run.json").read_text())
    assert rec["question"] == "logo?"
    acts = [t["action"] for t in rec["turns"]]
    assert any(a.startswith("look") for a in acts)
    assert acts[-1].startswith("answer")
    assert rec["turns"][0]["comment"] == "scripted comment 0"
    assert loop.acc.centroid is not None    # cloud fused

    # the refused look must already be on disk (crash-safe: every turn saved)
    look_turns = [t for t in rec["turns"] if t["action"].startswith("look")]
    refused = [t for t in look_turns if t.get("approved") is False]
    assert len(refused) == 1, "expected exactly one refused look turn"
    assert refused[0]["stopped"] is False
    approved = [t for t in look_turns if t.get("approved") is True]
    assert len(approved) == 1, "expected exactly one approved look turn"
    assert approved[0]["stopped"] is False


class CaptureFailRig(FakeRig):
    """Boot capture (pose 0) is fine; the first post-move capture (pose 1)
    raises, like a bad real-world view — the loop must not crash."""

    def capture(self, pose_id):
        if pose_id == 1:
            raise RuntimeError("synthetic bad view")
        return super().capture(pose_id)


class LookThenAnswerDecider:
    """READ: canned comment. DECIDE: one look, then answer regardless of
    what the previous turn's result was."""

    def __init__(self):
        self.decisions = 0

    def read(self, cap):
        return f"comment {self.decisions}"

    def decide(self, ctx):
        self.decisions += 1
        if self.decisions == 1:
            assert ctx.menu, "no reachable cells offered"
            return Look(ctx.menu[0].h, ctx.menu[0].v)
        return Answer("scripted: capture-fail path")


def test_capture_fail_returns_to_menu():
    q_survey = DEMO_PARK.copy()             # start already at survey: no boot move
    fs = FeedStream()
    con = SyncConsole(stream=fs)
    outdir = Path(tempfile.mkdtemp()) / "run2"
    loop = Loop(CaptureFailRig(q_survey), LookThenAnswerDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")

    th = _run_in_background(loop)
    ScriptedFeeder(con, fs).feed("y")       # approve the one look move
    _join(th)                               # must not raise / crash the run

    rec = json.loads((outdir / "run.json").read_text())
    acts = [t["action"] for t in rec["turns"]]
    assert any("capture failed" in t.get("result", "") for t in rec["turns"])
    assert acts[-1].startswith("answer")    # loop continued past the failure


class BootStopOnceRig(FakeRig):
    """move() reports a software stop on the very first call (mid-boot,
    zero waypoints done — q never actually changes) and succeeds normally
    afterwards. boot() must re-plan from wherever it stopped rather than
    exit the loop."""

    def __init__(self, q0):
        super().__init__(q0)
        self._stopped_once = False

    def move(self, path):
        if not self._stopped_once:
            self._stopped_once = True
            self.moves.append(len(path))
            return {"waypoints_done": 0, "s": 0.0,
                    "final_err_deg": 0.0, "stopped": True}
        return super().move(path)


def test_boot_stop_retries():
    q_survey = DEMO_PARK.copy()
    q_start = q_survey.copy()
    q_start[5] += np.radians(8)             # boot must plan+move to survey
    fs = FeedStream()
    con = SyncConsole(stream=fs)
    outdir = Path(tempfile.mkdtemp()) / "run3"
    loop = Loop(BootStopOnceRig(q_start), LookThenAnswerDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")

    th = _run_in_background(loop)
    feeder = ScriptedFeeder(con, fs)
    feeder.feed("y")                        # boot move 1: approved -> stopped
    feeder.feed("y")                        # boot move 2 (re-plan): approved -> moves
    feeder.feed("y")                        # first look: approved
    _join(th)                               # must not raise / crash the run

    rec = json.loads((outdir / "run.json").read_text())
    assert rec["turns"], "boot must have succeeded — no turns recorded"
    acts = [t["action"] for t in rec["turns"]]
    assert any(a.startswith("look") for a in acts), "expected >=1 look turn"


class ExecutorRefusalRig(FakeRig):
    """Boot move succeeds; the first look move raises like a real executor
    halt/refusal (protective stop, safety mode changed mid-path, ...)."""

    def __init__(self, q0):
        super().__init__(q0)
        self._boot_done = False

    def move(self, path):
        if not self._boot_done:
            self._boot_done = True
            return super().move(path)
        raise RuntimeError("safety mode changed mid-path")


def test_executor_refusal_recorded():
    q_survey = DEMO_PARK.copy()
    q_start = q_survey.copy()
    q_start[5] += np.radians(8)             # boot must plan+move to survey
    fs = FeedStream()
    con = SyncConsole(stream=fs)
    outdir = Path(tempfile.mkdtemp()) / "run4"
    loop = Loop(ExecutorRefusalRig(q_start), LookThenAnswerDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")

    th = _run_in_background(loop)
    feeder = ScriptedFeeder(con, fs)
    feeder.feed("y")                        # boot move: approved, succeeds
    feeder.feed("y")                        # look move: approved, executor raises
    _join(th)                               # must not raise / crash the run

    rec = json.loads((outdir / "run.json").read_text())
    look_turns = [t for t in rec["turns"] if t["action"].startswith("look")]
    assert len(look_turns) == 1, "expected exactly one look turn (then exit)"
    assert look_turns[0]["reason"] == "executor"


def main():
    test_full_fake_run()
    test_capture_fail_returns_to_menu()
    test_boot_stop_retries()
    test_executor_refusal_recorded()
    print("OK test_loop_fake")


if __name__ == "__main__":
    main()
