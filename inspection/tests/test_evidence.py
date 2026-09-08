#!/usr/bin/env python3
"""The evidence agent and the answer-time hunt flow.
Run: p inspection/tests/test_evidence.py
"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.agents.evidence_agent import (EVIDENCE, hunt_evidence,
                                                   normalise_evidence_image)
from inspection.eyes.models import StubVlm
from inspection.eyes.store import FindingWriter, RunStore
from inspection.eyes.tools import ViewTools
from inspection.eyes.verbs_local import LocalVerbs, StubBackend
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run

T = np.eye(4)


def _rig(tmp):
    root = Path(tmp) / "run"
    img = np.zeros((480, 848, 3), np.uint8); img[:, :, 0] = 70
    make_legacy_run(root, [{"cell": (3, 1), "pose_id": 1, "t": 1.0,
                           "rgb": img, "T_base_cam": T}])
    return Run.load(root)


def _hunt(tmp, script, find="the ACME wordmark"):
    run = _rig(tmp)
    store = RunStore.create(Path(tmp) / "run" / "notes", h_bins=12,
                            v_elevs=(10.0, 40.0, 70.0), r=0.35)
    tools = ViewTools(run, writer=FindingWriter(store))
    verbs = LocalVerbs(StubBackend(
        lines=[([[110, 110], [200, 110], [200, 140], [110, 140]],
                "ACME", 0.92)]))
    return store, hunt_evidence(tools, verbs, FindingWriter(store),
                                StubVlm(script), (3, 1), find)


def test_citation_is_parsed_leniently():
    # 2026-08-26 standing decision: lenient always. The model echoes a number
    # the loop announced; every reasonable spelling of it resolves.
    assert normalise_evidence_image(2) == 2
    assert normalise_evidence_image("2") == 2
    assert normalise_evidence_image("image 2") == 2
    assert normalise_evidence_image(2.0) == 2
    for garbage in (None, "", "the crop", True, [2]):
        assert normalise_evidence_image(garbage) is None


def test_found_hunt_resolves_the_cited_crop():
    with tempfile.TemporaryDirectory() as tmp:
        store, rec = _hunt(tmp, [
            {"tool": "read_text", "args": {}},
            {"tool": "crop", "args": {"box": [100, 100, 260, 180]}},
            {"evidence": ["read 'ACME' verbatim inside the crop"],
             "reasoning": "the wordmark is verified by OCR",
             "answer": "found — the ACME wordmark, whole and unclipped",
             "evidence_image": 1}])
        assert rec["found"] is True
        # Both live under the AI session's own artifacts/ now — that is
        # where an answer may cite them from (schema: EvidenceImage).
        assert rec["crop"] and "artifacts/" in rec["crop"]   # the cited crop
        assert rec["frame"] and rec["frame"].endswith("_frame.png")
        assert rec["report"].startswith("found")
        assert rec["transcript"]                           # audit trail
        assert (store.path / rec["transcript"]).exists()


def test_honest_miss_ships_with_frame_receipt_and_no_crop():
    with tempfile.TemporaryDirectory() as tmp:
        store, rec = _hunt(tmp, [
            {"evidence": ["a plain blue surface, no printing"],
             "reasoning": "nothing matching the target is visible",
             "answer": "not found — only a plain panel shows here"}])
        assert rec["found"] is False
        assert rec["crop"] is None
        assert rec["frame"]                # the "we looked here" receipt


def test_garbled_citation_falls_back_to_the_last_image():
    with tempfile.TemporaryDirectory() as tmp:
        store, rec = _hunt(tmp, [
            {"tool": "crop", "args": {"box": [100, 100, 260, 180]}},
            {"evidence": ["e"], "reasoning": "r",
             "answer": "found — the mark", "evidence_image": "the last crop"}])
        assert rec["found"] is True
        assert rec["crop"] and "artifacts/" in rec["crop"]  # last image taken


def test_the_hunter_has_no_note_and_no_view():
    # `note` feeds the planner's ledger; a hunt reports to a verdict. And no
    # `view` appendix: a hunt happens at answer time, on a run that is ending.
    assert "note" not in EVIDENCE.tool_names
    assert "view" not in EVIDENCE.emit
    assert EVIDENCE.emit[-1] == "evidence_image"


def _brain_rig(root):
    """A legacy-format run dir (run.json + numbered capture dirs), which is
    what Brain's `Run.load` expects — the RunStore alone is not a run."""
    import json
    import cv2
    turns = [{"step": 1, "target": "survey", "t": 1.0, "result": "ok"},
             {"step": 2, "target": [3, 1], "t": 2.0, "result": "+100 pts"}]
    (root / "run.json").write_text(json.dumps(
        {"q_survey": [0.0] * 6, "r": 0.35, "turns": turns}))
    img = np.zeros((480, 848, 3), np.uint8); img[:, :, 0] = 70
    for pose_id, name in [(0, "000"), (1, "001")]:
        d = root / name; d.mkdir()
        (d / "meta.json").write_text(json.dumps(
            {"pose_id": pose_id, "timestamp": 10.0 + pose_id,
             "joints_rad": [0.0] * 6, "T_base_flange": T.tolist(),
             "T_base_cam": T.tolist()}))
        cv2.imwrite(str(d / "rgb.png"), img)


def test_bounce_on_fresh_miss_then_recite_ships_the_gap():
    from inspection.brain.loop import Brain
    with tempfile.TemporaryDirectory() as tmp:
        _brain_rig(Path(tmp))
        vlm = StubVlm([{"evidence": ["e"], "reasoning": "r",
                        "answer": "not found — plain panel"}])
        brain = Brain(tmp, "is there a logo?", vlm=vlm,
                      verbs=LocalVerbs(StubBackend()))
        cites = [{"cell": [3, 1], "find": "the logo"}]
        bounce, recs = brain._collect_evidence(cites)
        assert bounce and "not over" in bounce             # fresh miss bounces
        assert recs[0]["found"] is False
        # Re-citing the same failed hunt = explicit acknowledgment: cache hit,
        # no new VLM call (the stub is exhausted), no bounce — ships with gap.
        bounce2, recs2 = brain._collect_evidence(cites)
        assert bounce2 is None and recs2[0] is recs[0]
        # A cell that was never captured is refused categorically.
        bounce3, _ = brain._collect_evidence([{"cell": [9, 9], "find": "x"}])
        assert bounce3 and "no capture" in bounce3


def main():
    test_citation_is_parsed_leniently()
    test_found_hunt_resolves_the_cited_crop()
    test_honest_miss_ships_with_frame_receipt_and_no_crop()
    test_garbled_citation_falls_back_to_the_last_image()
    test_the_hunter_has_no_note_and_no_view()
    test_bounce_on_fresh_miss_then_recite_ships_the_gap()
    print("OK test_evidence")


if __name__ == "__main__":
    main()
