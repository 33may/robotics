#!/usr/bin/env python3
"""AIRunWriter — the brain thread's own pen, ai/<seq>/ subtree only.
Run: p inspection/tests/test_ai_writer.py
"""
import json

import pytest

from inspection.record.ai_writer import AIRunWriter, next_seq
from inspection.record.run import Run
from inspection.record.schema import MenuDef, MenuInput, TranscriptRecord
from inspection.tests.record_fixtures import make_run


def _transcript(step_id=1, **over):
    kw = dict(transcript_id="t000", kind="inspect", step_id=step_id, t=1.0,
              model="vlm-x", task="is there a logo?",
              prompt="describe what you see")
    kw.update(over)
    return TranscriptRecord(**kw)


# --- create ---------------------------------------------------------------

def test_create_writes_airun_json_and_subdirs(tmp_path):
    make_run(tmp_path, run_id="0709-aiw")
    a = AIRunWriter.create(tmp_path / "0709-aiw", orchestrator_model="opus",
                           menu_id="viewsphere-menu", menu_hash="a" * 64)
    assert a.dir == tmp_path / "0709-aiw" / "ai" / "000"
    assert (a.dir / "transcripts").is_dir() and (a.dir / "menu").is_dir()
    raw = json.loads((a.dir / "airun.json").read_text())
    assert raw["seq"] == 0 and raw["mode"] == "live"
    assert raw["orchestrator_model"] == "opus"


def test_create_refuses_existing_seq_dir(tmp_path):
    make_run(tmp_path, run_id="0709-aiw2")
    AIRunWriter.create(tmp_path / "0709-aiw2", orchestrator_model="opus",
                       menu_id="m", menu_hash="a" * 64)
    with pytest.raises(FileExistsError):
        AIRunWriter.create(tmp_path / "0709-aiw2", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)


def test_create_honors_explicit_seq(tmp_path):
    make_run(tmp_path, run_id="0709-aiw3")
    a = AIRunWriter.create(tmp_path / "0709-aiw3", seq=2,
                           orchestrator_model="opus", menu_id="m",
                           menu_hash="a" * 64)
    assert a.dir.name == "002"


# --- next_seq ---------------------------------------------------------------

def test_next_seq_zero_when_no_ai_dir(tmp_path):
    make_run(tmp_path, run_id="0709-ns")
    assert next_seq(tmp_path / "0709-ns") == 0


def test_next_seq_is_max_plus_one(tmp_path):
    make_run(tmp_path, run_id="0709-ns2")
    run_dir = tmp_path / "0709-ns2"
    AIRunWriter.create(run_dir, seq=0, orchestrator_model="opus",
                       menu_id="m", menu_hash="a" * 64)
    AIRunWriter.create(run_dir, seq=3, orchestrator_model="opus",
                       menu_id="m", menu_hash="a" * 64)
    assert next_seq(run_dir) == 4


# --- transcripts / menu / answer / usage / events --------------------------

def test_transcript_writes_collision_proof_counter(tmp_path):
    make_run(tmp_path, run_id="0709-t")
    a = AIRunWriter.create(tmp_path / "0709-t", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    p0 = a.transcript(_transcript(step_id=1))
    p1 = a.transcript(_transcript(step_id=1, transcript_id="t001"))
    assert p0.name == "t000.json" and p1.name == "t001.json"
    assert TranscriptRecord.model_validate_json(p1.read_text()).transcript_id == "t001"


def test_menu_def_written_once(tmp_path):
    make_run(tmp_path, run_id="0709-md")
    a = AIRunWriter.create(tmp_path / "0709-md", orchestrator_model="opus",
                           menu_id="m", menu_hash="d" * 64)
    a.menu_def(MenuDef(menu_id="m", version="1", content_hash="d" * 64,
                       code_sha="deadbeef", verbs=["move", "answer"]))
    on_disk = MenuDef.model_validate_json((a.dir / "menu" / "menu_def.json").read_text())
    assert on_disk.menu_id == "m" and on_disk.verbs == ["move", "answer"]


def test_menu_input_appends_jsonl(tmp_path):
    make_run(tmp_path, run_id="0709-mi")
    a = AIRunWriter.create(tmp_path / "0709-mi", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    a.menu_input(MenuInput(turn=0, step_id=1, t=1.0))
    a.menu_input(MenuInput(turn=1, step_id=1, t=2.0))
    lines = (a.dir / "menu" / "inputs.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["turn"] == 1


def test_answer_accepts_dict_and_validates(tmp_path):
    make_run(tmp_path, run_id="0709-ans")
    a = AIRunWriter.create(tmp_path / "0709-ans", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    a.answer({"verdict": "yes", "reasoning": "r", "evidence": ["saw it"],
             "views_inspected": [[3, 0]], "coverage": "1/48 cells seen"})
    on_disk = json.loads((a.dir / "answer.json").read_text())
    assert on_disk["verdict"] == "yes" and on_disk["views_inspected"] == [[3, 0]]


def test_answer_rejects_unknown_field(tmp_path):
    make_run(tmp_path, run_id="0709-ans2")
    a = AIRunWriter.create(tmp_path / "0709-ans2", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    with pytest.raises(ValueError):
        a.answer({"verdict": "yes", "reasoning": "r", "typo_field": 1})


def test_usage_merges_into_airun_json(tmp_path):
    make_run(tmp_path, run_id="0709-us")
    a = AIRunWriter.create(tmp_path / "0709-us", orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    a.usage(output_tokens=4231)
    a.usage(cost_usd=0.51)
    raw = json.loads((a.dir / "airun.json").read_text())
    assert raw["usage"] == {"output_tokens": 4231, "cost_usd": 0.51}


def test_event_lands_in_shared_events_jsonl(tmp_path):
    run_dir = make_run(tmp_path, run_id="0709-ev")
    a = AIRunWriter.create(run_dir, orchestrator_model="opus",
                           menu_id="m", menu_hash="a" * 64)
    a.event("approved", step_id=1)
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["kind"] == "approved"


# --- full round trip through Run.load ---------------------------------------

def test_ai_writer_full_round_trip(tmp_path):
    make_run(tmp_path, run_id="0709-aiw")
    a = AIRunWriter.create(tmp_path / "0709-aiw", orchestrator_model="opus",
                           menu_id="viewsphere-menu", menu_hash="a" * 64)
    a.transcript(_transcript(step_id=1))
    a.answer({"verdict": "yes", "reasoning": "r", "evidence": []})
    a.event("approved", step_id=1)
    run = Run.load(tmp_path / "0709-aiw")
    assert run.ai[0].answer.verdict == "yes"
    assert run.ai[0].transcripts[0].step_id == 1
    assert run.events[-1].kind == "approved"  # landed in the SHARED events.jsonl


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
