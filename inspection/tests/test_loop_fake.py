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
    """READ: canned comment. DECIDE: first menu cell once, then answer."""

    def __init__(self):
        self.decisions = 0

    def read(self, cap):
        return f"scripted comment {self.decisions}"

    def decide(self, ctx):
        self.decisions += 1
        if self.decisions == 1:
            assert ctx.menu, "no reachable cells offered"
            return Look(ctx.menu[0].h, ctx.menu[0].v)
        return Answer("scripted: no logo")


def test_full_fake_run():
    q_survey = DEMO_PARK.copy()
    q_start = q_survey.copy()
    q_start[5] += np.radians(8)             # boot must plan+move to survey
    fs = FeedStream()
    con = Console(stream=fs)
    for _ in range(4):                      # approvals: boot move + one look
        fs.feed("y")
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


def main():
    test_full_fake_run()
    print("OK test_loop_fake")


if __name__ == "__main__":
    main()
