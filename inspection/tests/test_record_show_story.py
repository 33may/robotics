#!/usr/bin/env python3
"""show (Rerun projection) + story (AI trace reader) + CLI surface.
Run: p inspection/tests/test_record_show_story.py
"""
import json

import numpy as np
import pytest

from inspection.record.schema import (
    AIRunRecord,
    AnswerRecord,
    TranscriptRecord,
)
from inspection.record.story import story
from inspection.tests.record_fixtures import make_run
from inspection.tests.test_record_legacy import _legacy_run


def _add_airun(run_dir):
    adir = run_dir / "ai" / "0"
    (adir / "transcripts").mkdir(parents=True)
    (adir / "airun.json").write_text(AIRunRecord(
        seq=0, mode="live", orchestrator_model="claude-opus-5",
        menu_id="moves-v1", menu_hash="d" * 64).model_dump_json())
    (adir / "transcripts" / "t000.json").write_text(TranscriptRecord(
        transcript_id="t000", kind="inspect", step_id=0, t=10.0,
        model="gemini-er-2", task="is there a logo?",
        prompt="...", answer={"found": True}).model_dump_json())
    (run_dir / "answer.json").write_text(AnswerRecord(
        verdict="yes", reasoning="logo on survey").model_dump_json())


def test_story_renders_new_schema_run(tmp_path):
    d = make_run(tmp_path)
    _add_airun(d)
    text = story(d)
    assert "is there a logo?" in text  # the task
    assert "inspect" in text and "step 0" in text
    assert "verdict: yes" in text


def test_story_renders_legacy_run(tmp_path):
    d = _legacy_run(tmp_path, with_answer=True)
    text = story(d)
    assert "verdict: yes" in text


def test_show_writes_rrd(tmp_path):
    pytest.importorskip("rerun")
    from inspection.record.show import show_run
    d = make_run(tmp_path)
    np.save(d / "fused_cloud.npy", np.random.rand(50, 3))
    out = show_run(d, tmp_path / "out.rrd")
    assert out.exists() and out.stat().st_size > 1000


def test_show_handles_legacy_run(tmp_path):
    pytest.importorskip("rerun")
    from inspection.record.show import show_run
    d = _legacy_run(tmp_path)
    np.save(d / "fused_cloud.npy", np.random.rand(50, 3))
    out = show_run(d, tmp_path / "legacy.rrd")
    assert out.exists() and out.stat().st_size > 1000


def test_cli_ls_and_card(tmp_path, capsys):
    from inspection.record.__main__ import main
    make_run(tmp_path, run_id="0309-cup1", object="cup",
             status="completed", with_answer=True)
    main(["ls", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "0309-cup1" in out and "cup" in out
    main(["validate", "0309-cup1", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "ok" in out.lower()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
