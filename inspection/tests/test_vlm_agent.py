#!/usr/bin/env python3
"""The shared VLM-agent loop. Run: p inspection/tests/test_vlm_agent.py

The structural guarantees moved here with the loop (from test_eyes_inspect,
which keeps pinning the inspect variant's own surface): the tier cannot move,
cannot fetch other views, and no spec can reorder the emit or invent tools.
"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.agents import vlm_agent
from inspection.eyes.models import StubVlm
from inspection.eyes.store import FindingWriter, RunStore
from inspection.eyes.tools import ViewTools
from inspection.eyes.verbs_local import LocalVerbs, StubBackend
from inspection.eyes.agents.vlm_agent import EMIT_CORE, TOOLS, VlmAgent
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run

T = np.eye(4)


def _rig(tmp):
    root = Path(tmp) / "run"
    img = np.zeros((480, 848, 3), np.uint8); img[:, :, 1] = 90
    make_legacy_run(root, [{"cell": (3, 1), "pose_id": 1, "t": 1.0,
                           "rgb": img, "T_base_cam": T}])
    return Run.load(root)


def _run(tmp, agent, script):
    run = _rig(tmp)
    store = RunStore.create(Path(tmp) / "notes", h_bins=12,
                            v_elevs=(10.0, 40.0, 70.0), r=0.35)
    tools = ViewTools(run, writer=FindingWriter(store))
    verbs = LocalVerbs(StubBackend(boxes=[((100, 100, 300, 300), 0.9, "cup")]))
    f = agent.run(tools, verbs, FindingWriter(store), StubVlm(script),
                  task="t", cell=(3, 1))
    return store, f


def test_the_loop_is_never_given_a_way_to_move():
    src = Path(vlm_agent.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.run" not in src
    for verb in ("view_at", "views_near"):
        assert verb not in TOOLS                  # no cross-view retrieval
    assert set(TOOLS) == {"detect", "segment", "read_text", "crop", "note"}


def test_a_broken_spec_dies_at_import_time_not_at_2am():
    for bad in (dict(emit=("answer", "reasoning", "evidence")),   # CoT order
                dict(emit=EMIT_CORE + ("confidence",),
                     extras={"confidence": str}),                 # chance-level
                dict(emit=EMIT_CORE, extras={"view": dict}),      # undeclared
                dict(tool_names=frozenset({"detect", "teleport"}))):
        try:
            VlmAgent(rules="r", **bad)
            raise RuntimeError(f"spec {bad} should not construct")
        except AssertionError:
            pass


def test_a_pruned_tool_is_refused_categorically_and_the_loop_continues():
    lean = VlmAgent(rules="r", tool_names=frozenset({"detect", "crop"}))
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, lean, [
            {"tool": "note", "args": {"text": "x"}},
            {"evidence": ["e"], "reasoning": "r", "answer": "ok"}])
        assert f.answer == "ok"
        assert store.notes() == []                # note never ran
        refusal = [t for t in f.transcript["turns"] if "result" in t][0]
        assert "not a tool this agent has" in refusal["result"]


def test_images_are_numbered_as_handed_over():
    # The citation mechanism for the evidence agent, inert text for everyone
    # else: the model can only cite a number the loop itself announced, and
    # the numbering counts SAVED images so it aligns with the transcript.
    plain = VlmAgent(rules="r")
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, plain, [
            {"tool": "crop", "args": {"box": [10, 10, 200, 200]}},
            {"tool": "crop", "args": {"box": [20, 20, 150, 150]}},
            {"evidence": ["e"], "reasoning": "r", "answer": "ok"}])
        crops = [t for t in f.transcript["turns"]
                 if "result" in t and t.get("image")]
        assert "this is image 1" in crops[0]["result"]
        assert "this is image 2" in crops[1]["result"]
        assert f.transcript["image"]              # frame saved -> image 0


def test_extras_are_normalised_written_to_disk_and_exposed():
    import json
    spec = VlmAgent(rules="r", emit=EMIT_CORE + ("mark",),
                    extras={"mark": lambda raw: str(raw or "").upper()})
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, spec, [{"evidence": ["e"], "reasoning": "r",
                                     "answer": "ok", "mark": "here"}])
        assert f.extras["mark"] == "HERE"
        disk = json.loads((store.path / store.findings()[0]["transcript"])
                          .read_text())
        assert disk["mark"] == "HERE"             # disk and Finding agree
        assert f.view == {}                       # undeclared appendix -> {}


class _YFirstVlm(StubVlm):
    """A model that speaks ER-2's convention — the real GeminiVlm codecs, no
    API key. Non-identity conversion is the point: it is what exposes any
    asymmetry at the box boundary."""
    from inspection.eyes.models import GeminiVlm as _G
    BOX_CONVENTION = _G.BOX_CONVENTION
    # re-wrap: class-body assignment of an already-resolved function would
    # rebind it as an instance method and feed it `self` as the box.
    to_pixel_box = staticmethod(_G.to_pixel_box)
    to_model_box = staticmethod(_G.to_model_box)


def test_boxes_round_trip_through_the_model_boundary():
    # 2708-aicam f04: detect printed pixels, the model echoed them into crop,
    # the adapter decoded 0-1000 y-first, every crop landed on background —
    # 16 turns burned. The invariant: a box the model READS, echoed back
    # verbatim, must land on the SAME pixels it described.
    from inspection.eyes.models import GeminiVlm
    px = (100, 100, 300, 300)                       # StubBackend's detect box
    echoed = GeminiVlm.to_model_box(px, 848, 480)
    spec = VlmAgent(rules="r")
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp)
        store = RunStore.create(Path(tmp) / "notes", h_bins=12,
                                v_elevs=(10.0, 40.0, 70.0), r=0.35)
        tools = ViewTools(run, writer=FindingWriter(store))
        verbs = LocalVerbs(StubBackend(boxes=[(px, 0.9, "cup")]))
        f = spec.run(tools, verbs, FindingWriter(store),
                     _YFirstVlm([{"tool": "detect", "args": {"phrase": "cup"}},
                                 {"tool": "crop", "args": {"box": echoed}},
                                 {"evidence": ["e"], "reasoning": "r",
                                  "answer": "ok"}]),
                     task="t", cell=(3, 1))
        results = [t["result"] for t in f.transcript["turns"] if "result" in t]
        assert str(echoed) in results[0]            # detect speaks MODEL boxes
        assert "crop 100,100-300,300" in results[1]  # echo lands on the pixels
        assert f"region {echoed}" in results[1]      # echoed back model-side


def test_read_text_takes_a_region_and_each_box_is_its_own_call():
    # Full-frame OCR on 848x480 cannot resolve label text, and the argless
    # call was one-shot behind the dedup (2708-aicam f04 got garbage once and
    # could never retry). A region read is sharper and each box is distinct.
    spec = VlmAgent(rules="r")
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp)
        store = RunStore.create(Path(tmp) / "notes", h_bins=12,
                                v_elevs=(10.0, 40.0, 70.0), r=0.35)
        tools = ViewTools(run, writer=FindingWriter(store))
        verbs = LocalVerbs(StubBackend(
            lines=[([[110, 110], [200, 110], [200, 140], [110, 140]],
                    "ACME", 0.92)]))
        f = spec.run(tools, verbs, FindingWriter(store),
                     StubVlm([{"tool": "read_text", "args": {}},
                              {"tool": "read_text",
                               "args": {"box": [50, 50, 400, 400]}},
                              {"tool": "read_text",
                               "args": {"box": [60, 60, 400, 400]}},
                              {"evidence": ["e"], "reasoning": "r",
                               "answer": "ok"}]),
                     task="t", cell=(3, 1))
        results = [t["result"] for t in f.transcript["turns"] if "result" in t]
        assert all("ACME" in r for r in results[:3])   # none hit the dedup
        assert "in [50, 50, 400, 400]" in results[1]   # region echoed


def test_failure_messages_speak_the_models_convention():
    spec = VlmAgent(rules="r")
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp)
        store = RunStore.create(Path(tmp) / "notes", h_bins=12,
                                v_elevs=(10.0, 40.0, 70.0), r=0.35)
        tools = ViewTools(run, writer=FindingWriter(store))
        f = spec.run(tools, LocalVerbs(StubBackend()), FindingWriter(store),
                     _YFirstVlm([{"tool": "crop", "args": {"box": [0, 0, 0, 0]}},
                                 {"evidence": ["e"], "reasoning": "r",
                                  "answer": "ok"}]),
                     task="t", cell=(3, 1))
        fail = [t["result"] for t in f.transcript["turns"] if "result" in t][0]
        assert "y first" in fail                     # the model's OWN grammar
        assert "x0, y0" not in fail                  # never a foreign one


def test_exhaustion_is_a_flag_not_a_string_match():
    spec = VlmAgent(rules="r", max_turns=2)
    with tempfile.TemporaryDirectory() as tmp:
        store, f = _run(tmp, spec, [
            {"tool": "detect", "args": {"phrase": "a"}},
            {"tool": "detect", "args": {"phrase": "b"}}])
        assert f.exhausted and f.answer == "unknown"
        assert f.transcript_rel and (store.path / f.transcript_rel).exists()


def main():
    test_the_loop_is_never_given_a_way_to_move()
    test_a_broken_spec_dies_at_import_time_not_at_2am()
    test_a_pruned_tool_is_refused_categorically_and_the_loop_continues()
    test_images_are_numbered_as_handed_over()
    test_extras_are_normalised_written_to_disk_and_exposed()
    test_boxes_round_trip_through_the_model_boundary()
    test_read_text_takes_a_region_and_each_box_is_its_own_call()
    test_failure_messages_speak_the_models_convention()
    test_exhaustion_is_a_flag_not_a_string_match()
    print("OK test_vlm_agent")


if __name__ == "__main__":
    main()
