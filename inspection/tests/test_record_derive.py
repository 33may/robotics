#!/usr/bin/env python3
"""derive — idempotent derivations with provenance, explicit per-step failures.
Run: p inspection/tests/test_record_derive.py
"""
import pytest

from inspection.record.derive import run_derivation
from inspection.record.schema import DerivationMeta
from inspection.tests.record_fixtures import make_run

CALLS = {"n": 0}


def toy_method(run_dir, out_dir, params):
    """Counts bytes of each step's rgb.png; 'fails' on steps without one."""
    CALLS["n"] += 1
    statuses = {}
    for sdir in sorted((run_dir / "steps").iterdir()):
        sid = int(sdir.name)
        rgb = sdir / "rgb.png"
        if rgb.exists():
            (out_dir / f"step_{sid:03d}.txt").write_text(str(len(rgb.read_bytes())))
            statuses[sid] = "ok"
        else:
            statuses[sid] = "failed: no rgb"
    return statuses


def _derive(tmp_path, run_dir):
    return run_derivation(
        run_dir, tmp_path / "derived", method="bytecount", fn=toy_method,
        params={"unit": "bytes"}, semver="1.0", code_sha="bf23b3aa",
        sources=["steps/000/rgb.png"], now=1788180100.0)


def test_derivation_writes_output_and_provenance(tmp_path):
    CALLS["n"] = 0
    run_dir = make_run(tmp_path / "runs")
    out = _derive(tmp_path, run_dir)
    meta = DerivationMeta.model_validate_json(
        (out / "meta.json").read_text())
    assert meta.version == "bytecount/1.0+p:bf23b3aa"
    assert meta.steps[0] == "ok"
    assert meta.steps[1] == "failed: no rgb"  # explicit, never a silent gap
    assert "steps/000/rgb.png" in meta.source_hashes
    assert (out / "step_000.txt").exists()


def test_rerun_skips_when_fresh(tmp_path):
    CALLS["n"] = 0
    run_dir = make_run(tmp_path / "runs")
    _derive(tmp_path, run_dir)
    _derive(tmp_path, run_dir)
    assert CALLS["n"] == 1  # second call skipped — same sources, same version


def test_source_change_triggers_recompute(tmp_path):
    CALLS["n"] = 0
    run_dir = make_run(tmp_path / "runs")
    _derive(tmp_path, run_dir)
    (run_dir / "steps" / "000" / "rgb.png").write_bytes(b"different bytes now!")
    _derive(tmp_path, run_dir)
    assert CALLS["n"] == 2  # stale detected via source hash


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
