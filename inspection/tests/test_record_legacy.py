#!/usr/bin/env python3
"""Legacy adapter — 27 pre-schema runs readable through the same surface,
adapt-on-read, never writing a byte. Run: p inspection/tests/test_record_legacy.py
"""
import json

import numpy as np
import pytest

from inspection.record.catalog import runs
from inspection.record.legacy import adapt_run
from inspection.tests.record_fixtures import make_run

T4 = np.eye(4).tolist()


def _legacy_run(root, run_id="2408-cup9", *, with_answer=False,
                with_session=True, old_meta_on_view=1):
    """Mimic the 2408-era layout: run.json{q_survey,r,turns} + NNN/meta.json."""
    d = root / run_id
    d.mkdir(parents=True)
    turns = [
        {"step": 1, "target": "survey", "t": 1.0, "result": "+100 pts, fused 90", "stopped": False},
        {"step": 2, "target": [3, 0], "t": 2.0, "result": "+80 pts, fused 150", "stopped": False},
    ]
    (d / "run.json").write_text(json.dumps(
        {"q_survey": [0.0] * 6, "r": 0.24, "turns": turns}))
    if with_session:
        (d / "session.json").write_text(json.dumps(
            {"serial": "123622270954", "resolution": [848, 480],
             "depth_scale_m_per_unit": 1e-4, "intrinsics": {},
             "extrinsics_ir1_to_ir2": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                       "translation_m": [0.018, 0, 0]}}))
    for pose_id in (0, 1):
        vdir = d / f"{pose_id:03d}"
        vdir.mkdir()
        meta = {"pose_id": pose_id, "timestamp": 100.0 + pose_id,
                "joints_rad": [0.1] * 6, "T_base_flange": T4, "T_base_cam": T4,
                "mask": {"score": 0.9, "box": [1, 2, 3, 4], "px": 55}}
        if pose_id >= old_meta_on_view:
            pass  # old-era view: no rgb_rotation_deg
        else:
            meta["rgb_rotation_deg"] = 0
        (vdir / "meta.json").write_text(json.dumps(meta))
        (vdir / "rgb.png").write_bytes(b"\x89PNG x")
    if with_answer:
        (d / "eyes").mkdir()
        (d / "eyes" / "answer.json").write_text(json.dumps(
            {"verdict": "yes", "reasoning": "r", "evidence": [],
             "coverage": "", "views_inspected": [[3, 0]]}))
    return d


def test_adapt_maps_views_to_steps(tmp_path):
    d = _legacy_run(tmp_path)
    a = adapt_run(d)
    assert a.run.provenance == "legacy"
    assert a.run.steps == [0, 1]
    s1 = a.steps[1]
    assert s1.view.address == [3, 0]  # from the turns join
    assert s1.T_base_cam == T4
    assert s1.segmentation.px == 55


def test_synthesized_fields_are_declared(tmp_path):
    d = _legacy_run(tmp_path, old_meta_on_view=1)
    a = adapt_run(d)
    assert "t_arrived" in a.steps[0].synthesized
    assert "rgb_rotation_deg" in a.steps[1].synthesized  # absent on old views
    assert "rgb_rotation_deg" not in a.steps[0].synthesized  # present there


def test_status_and_rig_inference(tmp_path):
    done = adapt_run(_legacy_run(tmp_path, "2408-a", with_answer=True))
    dead = adapt_run(_legacy_run(tmp_path, "2408-b"))
    mock = adapt_run(_legacy_run(tmp_path, "2408-c", with_session=False))
    assert done.run.status == "completed"
    assert dead.run.status == "aborted"
    assert mock.run.rig == "fake"


def test_adapter_never_writes(tmp_path):
    d = _legacy_run(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}
    adapt_run(d)
    after = {p: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}
    assert before == after


def test_catalog_spans_both_generations(tmp_path):
    make_run(tmp_path, run_id="0309-new", object="box")
    _legacy_run(tmp_path, "2408-old")
    cards = runs(tmp_path)
    ids = {c.id for c in cards}
    assert ids == {"0309-new", "2408-old"}
    old = next(c for c in cards if c.id == "2408-old")
    assert old.n_steps == 2 and old.status == "aborted"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
