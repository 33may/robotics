#!/usr/bin/env python3
"""catalog — cards + filters over run.json files; files are the truth.
Run: p inspection/tests/test_record_catalog.py
"""
import json
import os

import pytest

from inspection.record.catalog import card, runs
from inspection.tests.record_fixtures import make_run


def test_runs_lists_all_with_cards(tmp_path):
    make_run(tmp_path, run_id="0309-boxA", object="box")
    make_run(tmp_path, run_id="0309-cup1", object="cup")
    cards = runs(tmp_path)
    assert {c.id for c in cards} == {"0309-boxA", "0309-cup1"}


def test_filter_by_object_and_status(tmp_path):
    make_run(tmp_path, run_id="0309-boxA", object="box")
    make_run(tmp_path, run_id="0309-cup1", object="cup",
             status="completed", with_answer=True)
    assert [c.id for c in runs(tmp_path, object="cup")] == ["0309-cup1"]
    assert [c.id for c in runs(tmp_path, status="aborted")] == ["0309-boxA"]


def test_card_carries_summary_fields(tmp_path):
    make_run(tmp_path, run_id="0309-cup1", object="cup",
             status="completed", with_answer=True)
    c = card(tmp_path, "0309-cup1")
    assert c.object == "cup" and c.n_steps == 2
    assert c.size_bytes > 0  # summed from manifest
    assert c.verdict == "yes"  # pulled from answer.json
    assert c.question == "is there a logo?"


def test_crashed_shown_as_effective_status(tmp_path):
    d = make_run(tmp_path, run_id="0309-dead")
    raw = json.loads((d / "run.json").read_text())
    raw["status"] = "running"
    raw["closed_at"] = None
    (d / "run.json").write_text(json.dumps(raw))
    for p in d.rglob("*"):
        os.utime(p, (1000.0, 1000.0))
    c = card(tmp_path, "0309-dead", stale_s=3600)
    assert c.status == "running" and c.effective_status == "crashed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
