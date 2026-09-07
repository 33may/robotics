#!/usr/bin/env python3
"""The inspection subagent's run loop. Run: p inspection/tests/test_eyes_inspect.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.agents.inspect_agent import SCHEMA_ORDER, inspect_view
from inspection.eyes.models import StubVlm
from inspection.eyes.store import FindingWriter, RunStore
from inspection.eyes.tools import ViewTools
from inspection.eyes.verbs_local import LocalVerbs, StubBackend
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run

T = np.eye(4)


def _rig(tmp):
    root = Path(tmp) / "run"
    img = np.zeros((480, 848, 3), np.uint8); img[:, :, 1] = 90
    make_legacy_run(root, [{"cell": (3, 1), "pose_id": 1, "t": 1.0,
                           "rgb": img, "T_base_cam": T}])
    return Run.load(root)


def _run(tmp, script):
    run = _rig(tmp)
    store = RunStore.create(Path(tmp) / "notes", h_bins=12,
                            v_elevs=(10.0, 40.0, 70.0), r=0.35)
    tools = ViewTools(run, writer=FindingWriter(store))
    verbs = LocalVerbs(StubBackend(boxes=[((100, 100, 300, 300), 0.9, "cup")],
                                   lines=[([[110, 110], [200, 110], [200, 140],
                                            [110, 140]], "ACME", 0.92)]))
    finding = inspect_view(tools, verbs, FindingWriter(store), StubVlm(script),
                           cell=(3, 1), task="is there a logo on this cup?")
    return store, finding


def test_single_turn_answer_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [{"evidence": ["a green cup fills the frame"],
                               "reasoning": "no marking is visible on this side",
                               "answer": "no"}])
        assert f.answer == "no" and f.cell == (3, 1)
        assert store.findings()[0]["summary"].startswith("no")
        assert len(store.findings()) == 1


def test_tool_turns_run_then_the_answer_lands():
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [
            {"tool": "detect", "args": {"phrase": "cup"}},
            {"tool": "read_text", "args": {}},
            {"evidence": ["text ACME on the body"],
             "reasoning": "the OCR read sits inside the cup box",
             "answer": "yes"}])
        assert f.answer == "yes"
        assert [t["tool"] for t in f.transcript["turns"] if "tool" in t] == \
            ["detect", "read_text"]


def test_transcript_is_written_verbatim_but_only_the_summary_returns():
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [
            {"tool": "detect", "args": {"phrase": "cup"}},
            {"evidence": ["e"], "reasoning": "r", "answer": "no"}])
        rel = store.findings()[0]["transcript"]
        disk = json.loads((store.path / rel).read_text())
        assert len(disk["turns"]) == len(f.transcript["turns"]) >= 3
        assert "prompt" in disk and "cell [3, 1]" in disk["prompt"]
        assert store.findings()[0]["summary"] == f.summary
        assert len(f.summary) < len(json.dumps(disk))     # distilled, not verbatim


def test_uncaptured_neighbour_becomes_a_note_not_a_move():
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [
            {"tool": "note", "args": {"text": "want the far side, not captured"}},
            {"evidence": ["e"], "reasoning": "r", "answer": "unknown"}])
        texts = [n["text"] for n in store.notes()]
        assert any("far side" in t for t in texts)
        assert all(n["who"] == "finding" for n in store.notes())


def test_schema_puts_reasoning_before_answer():
    # Answer-first erases the CoT gain entirely (LaMDA GSM8K 14.3 -> 6.1), and
    # JSON mode is what causes it: 100% of GPT-3.5 responses emitted `answer`
    # before `reason`. Field ORDER is the contract, not decoration.
    assert SCHEMA_ORDER.index("reasoning") < SCHEMA_ORDER.index("answer")
    assert SCHEMA_ORDER.index("evidence") < SCHEMA_ORDER.index("reasoning")
    assert SCHEMA_ORDER.index("answer") < SCHEMA_ORDER.index("view")
    assert "confidence" not in SCHEMA_ORDER      # verbalized confidence ~ chance


def test_view_block_is_normalised_and_kept():
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [
            {"evidence": ["e"], "reasoning": "r", "answer": "no",
             "view": {"saw": "plain green side, the mark turned away right",
                      "recommendation": "around right"}}])
        assert f.view == {"saw": "plain green side, the mark turned away right",
                          "recommendation": "around right"}
        rel = store.findings()[0]["transcript"]
        disk = json.loads((store.path / rel).read_text())
        assert disk["view"] == f.view


def test_view_none_recommendation_is_dropped_saw_survives():
    # "none" means nothing to recommend: absence, not a row saying "none".
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [
            {"evidence": ["e"], "reasoning": "r", "answer": "yes",
             "view": {"saw": "logo square to the camera",
                      "recommendation": "none"}}])
        assert f.view == {"saw": "logo square to the camera"}


def test_view_survives_malformed_shapes():
    # A bare string becomes `saw`; garbage becomes an empty dict. An inspection
    # that saw the object clearly must not die over its appendix.
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [{"evidence": ["e"], "reasoning": "r",
                               "answer": "no", "view": "only the rim shows"}])
        assert f.view == {"saw": "only the rim shows"}
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, [{"evidence": ["e"], "reasoning": "r",
                               "answer": "no", "view": 42}])
        assert f.view == {}


def test_the_subagent_is_never_given_a_way_to_move():
    from inspection.eyes.agents import inspect_agent
    src = Path(inspect_agent.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.run" not in src
    for verb in ("view_at", "views_near"):
        assert verb not in inspect_agent.TOOLS        # no cross-view retrieval
    assert set(inspect_agent.TOOLS) == {"detect", "segment", "read_text",
                                        "crop", "note"}


def main():
    test_single_turn_answer_is_recorded()
    test_tool_turns_run_then_the_answer_lands()
    test_transcript_is_written_verbatim_but_only_the_summary_returns()
    test_uncaptured_neighbour_becomes_a_note_not_a_move()
    test_schema_puts_reasoning_before_answer()
    test_view_block_is_normalised_and_kept()
    test_view_none_recommendation_is_dropped_saw_survives()
    test_view_survives_malformed_shapes()
    test_the_subagent_is_never_given_a_way_to_move()
    print("OK test_eyes_inspect")


if __name__ == "__main__":
    main()
