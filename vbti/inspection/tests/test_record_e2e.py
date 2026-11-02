#!/usr/bin/env python3
"""End-to-end: write -> validate -> catalog -> derive -> tamper-detect.
The step-4 API layer proven as one chain. Run: p inspection/tests/test_record_e2e.py
"""
import pytest

from inspection.record.catalog import card
from inspection.record.derive import run_derivation
from inspection.record.validate import validate_run
from inspection.tests.record_fixtures import make_run


def test_full_chain(tmp_path):
    runs_root = tmp_path / "runs"

    # 1. capture a run through the writer
    run_dir = make_run(runs_root, run_id="0309-cup1", object="cup",
                       status="completed", with_answer=True)

    # 2. it validates clean
    rep = validate_run(run_dir)
    assert rep.ok, [p.what for p in rep.problems]

    # 3. the catalog finds and summarizes it
    c = card(runs_root, "0309-cup1")
    assert c.object == "cup" and c.verdict == "yes" and c.n_steps == 2

    # 4. a derivation runs with provenance, and re-running skips
    calls = {"n": 0}

    def method(run_dir, out_dir, params):
        calls["n"] += 1
        (out_dir / "result.txt").write_text("42")
        return {0: "ok", 1: "failed: rejected step has no cloud"}

    out = run_derivation(run_dir, tmp_path / "derived", method="toy",
                         fn=method, params={}, semver="0.1", code_sha="abc123",
                         sources=["steps/000/rgb.png"], now=1788180200.0)
    run_derivation(run_dir, tmp_path / "derived", method="toy",
                   fn=method, params={}, semver="0.1", code_sha="abc123",
                   sources=["steps/000/rgb.png"], now=1788180300.0)
    assert calls["n"] == 1 and (out / "meta.json").exists()

    # 5. deep validation catches byte tampering the cheap pass can't see
    p = run_dir / "steps" / "000" / "rgb.png"
    original = p.read_bytes()
    st = p.stat()
    p.write_bytes(b"Z" * len(original))
    import os
    os.utime(p, (st.st_atime, st.st_mtime))
    assert validate_run(run_dir, deep=False).ok
    assert not validate_run(run_dir, deep=True).ok


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
