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


# --- AI subtree: menu-hash cross-check on the REAL layout ---------------------

def test_ai_menu_hash_mismatch_flagged_on_new_layout(tmp_path):
    """CROSS-TASK FIX: validate.py used to probe the old `ai/<seq>/menu.json`
    (task-4's binding layout is `ai/<seq>/menu/menu_def.json`), so this check
    silently no-op'd against every real run. Written via AIRunWriter — the
    same writer the brain thread uses — to prove the fix against the real
    on-disk shape, not a hand-rolled stand-in."""
    from inspection.record.ai_writer import AIRunWriter
    from inspection.record.schema import MenuDef

    d = make_run(tmp_path)
    a = AIRunWriter.create(d, orchestrator_model="opus",
                           menu_id="viewsphere-menu", menu_hash="a" * 64)
    a.menu_def(MenuDef(menu_id="viewsphere-menu", version="1",
                       content_hash="b" * 64, code_sha="deadbeef",
                       verbs=["move"]))
    rep = validate_run(d)
    assert not rep.ok
    assert any("menu_hash" in p.what for p in rep.problems)


def test_ai_menu_hash_match_is_clean(tmp_path):
    from inspection.record.ai_writer import AIRunWriter
    from inspection.record.schema import MenuDef

    d = make_run(tmp_path)
    a = AIRunWriter.create(d, orchestrator_model="opus",
                           menu_id="viewsphere-menu", menu_hash="b" * 64)
    a.menu_def(MenuDef(menu_id="viewsphere-menu", version="1",
                       content_hash="b" * 64, code_sha="deadbeef",
                       verbs=["move"]))
    rep = validate_run(d)
    assert rep.ok, [p.what for p in rep.problems]


def test_ai_menu_absent_is_a_noop_warning_not_error(tmp_path):
    """No menu/menu_def.json on disk yet — the cross-check has nothing to
    compare against; it must not hard-error, but it must say so (warn),
    not silently do nothing."""
    from inspection.record.ai_writer import AIRunWriter

    d = make_run(tmp_path)
    AIRunWriter.create(d, orchestrator_model="opus",
                       menu_id="viewsphere-menu", menu_hash="a" * 64)
    rep = validate_run(d)
    assert rep.ok  # a warn never fails the report
    warns = [p for p in rep.problems if p.severity == "warn" and "ai/" in p.where]
    assert warns, [p.what for p in rep.problems]


def test_validate_archive_loops_all_runs(tmp_path):
    make_run(tmp_path, run_id="0309-a")
    make_run(tmp_path, run_id="0309-b")
    reports = validate_archive(tmp_path)
    assert len(reports) == 2 and all(r.ok for r in reports)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
