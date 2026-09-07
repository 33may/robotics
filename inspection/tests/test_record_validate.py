#!/usr/bin/env python3
"""validate_run — per-file schema + cross-refs + manifest + crash-on-read.
Run: p inspection/tests/test_record_validate.py
"""
import json
import os

import pytest

from inspection.record.validate import validate_archive, validate_run
from inspection.tests.record_fixtures import make_run


def test_writer_output_validates_clean(tmp_path):
    d = make_run(tmp_path)
    rep = validate_run(d)
    assert rep.ok, [p.what for p in rep.problems]


def test_completed_run_requires_answer(tmp_path):
    d = make_run(tmp_path, status="completed", with_answer=False)
    rep = validate_run(d)
    assert not rep.ok
    assert any("answer" in p.what for p in rep.problems)


def test_completed_with_answer_is_clean(tmp_path):
    d = make_run(tmp_path, status="completed", with_answer=True)
    assert validate_run(d).ok


def test_unresolved_method_id_flagged(tmp_path):
    d = make_run(tmp_path)
    sj = d / "steps" / "000" / "step.json"
    raw = json.loads(sj.read_text())
    raw["view"]["method"] = "ghost-method"
    sj.write_text(json.dumps(raw))
    rep = validate_run(d)
    assert any("ghost-method" in p.what for p in rep.problems)


def test_step_dir_missing_from_roster_flagged(tmp_path):
    d = make_run(tmp_path)
    extra = d / "steps" / "007"
    extra.mkdir()
    (extra / "step.json").write_text(
        (d / "steps" / "001" / "step.json").read_text().replace(
            '"step_id": 1', '"step_id": 7'))
    rep = validate_run(d)
    assert any("roster" in p.what for p in rep.problems)


def test_manifest_missing_binary_flagged(tmp_path):
    d = make_run(tmp_path)
    (d / "steps" / "000" / "rgb.png").unlink()
    rep = validate_run(d)
    assert any("rgb.png" in p.what for p in rep.problems)


def test_same_size_tamper_caught_only_by_deep(tmp_path):
    d = make_run(tmp_path)
    p = d / "steps" / "000" / "rgb.png"
    original = p.read_bytes()
    st = p.stat()
    p.write_bytes(b"X" * len(original))  # same size
    os.utime(p, (st.st_atime, st.st_mtime))  # same mtime
    assert validate_run(d, deep=False).ok  # cheap pass fooled
    rep = validate_run(d, deep=True)
    assert any("sha256" in p_.what for p_ in rep.problems)


def test_stale_running_run_reported_crashed(tmp_path):
    d = make_run(tmp_path)
    raw = json.loads((d / "run.json").read_text())
    raw["status"] = "running"
    raw["closed_at"] = None
    (d / "run.json").write_text(json.dumps(raw))
    old = 1000.0
    for p in d.rglob("*"):
        os.utime(p, (old, old))
    rep = validate_run(d, stale_s=3600)
    assert rep.effective_status == "crashed"


def test_validate_archive_loops_all_runs(tmp_path):
    make_run(tmp_path, run_id="0309-a")
    make_run(tmp_path, run_id="0309-b")
    reports = validate_archive(tmp_path)
    assert len(reports) == 2 and all(r.ok for r in reports)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
