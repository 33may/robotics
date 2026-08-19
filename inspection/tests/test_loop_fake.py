#!/usr/bin/env python3
"""Headless dry-run of the whole loop: real world+IK+planner, fake arm and
camera, scripted decider. Run: p inspection/tests/test_loop_fake.py"""
import json
import tempfile
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
    con = Console(stream=fs)
    fs.feed("y")                            # boot move: approved
    fs.feed("n")                            # first look at the cell: refused
    fs.feed("y")                            # second look, same cell: approved
    outdir = Path(tempfile.mkdtemp()) / "run1"
    loop = Loop(FakeRig(q_start), ScriptDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")
    loop.run()

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
    what the capture-fail turn's result was."""

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
    con = Console(stream=fs)
    fs.feed("y")                            # approve the one look move
    outdir = Path(tempfile.mkdtemp()) / "run2"
    loop = Loop(CaptureFailRig(q_survey), LookThenAnswerDecider(), con, outdir,
                q_survey=q_survey, max_turns=5, question="logo?")
    loop.run()                              # must not raise / crash the run

    rec = json.loads((outdir / "run.json").read_text())
    acts = [t["action"] for t in rec["turns"]]
    assert any("capture failed" in t.get("result", "") for t in rec["turns"])
    assert acts[-1].startswith("answer")    # loop continued past the failure


def main():
    test_full_fake_run()
    test_capture_fail_returns_to_menu()
    print("OK test_loop_fake")


if __name__ == "__main__":
    main()
