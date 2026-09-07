#!/usr/bin/env python3
"""RunWriter — the single writer both the live loop and the data engine use.
Run: p inspection/tests/test_record_writer.py
"""
import json

import numpy as np
import pytest

from inspection.record.schema import (
    ConfigSnapshot,
    GeometryStats,
    Manifest,
    RunRecord,
    SessionRecord,
    StepRecord,
    ViewState,
)
from inspection.record.writer import RunWriter, default_id

T4 = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]

CONFIG = ConfigSnapshot(
    git_sha="76fabc3",
    cell_yaml={"sha256": "b" * 64, "content": "frames: {}\n"},
    calib={"file": "T_flange_cam_2026-08-19.npy", "sha256": "c" * 64},
)

SESSION = SessionRecord(
    serial="123622270954", resolution=[848, 480],
    depth_scale_m_per_unit=1e-4,
)


def _writer(tmp_path, **over):
    kw = dict(run_id="0309-boxA", name="boxA", object="box",
              source="data-engine", rig="real",
              question="is there a logo?",
              view_methods=[{"id": "vs1", "kind": "viewsphere",
                             "params": {"h_bins": 12, "v_elevs": [10.0]}}],
              config=CONFIG, session=SESSION)
    kw.update(over)
    return RunWriter.create(tmp_path, **kw)


def _capture(w, step_id, **over):
    kw = dict(t_captured=2.0, joints_rad=[0.1] * 6, T_base_flange=T4,
              T_base_cam=T4, rgb_rotation_deg=0)
    kw.update(over)
    w.write_capture(step_id, **kw)


GEO = GeometryStats(offered=100, kept=90, dropped=10, fused_points=90,
                    source="mask")


def _vstate(step_id):
    return ViewState(step_id=step_id, candidates=[
        {"address": [3, 0], "pose": T4, "status": "current"}])


# --- create -------------------------------------------------------------------

def test_create_writes_datasheet_config_session(tmp_path):
    w = _writer(tmp_path)
    d = tmp_path / "0309-boxA"
    run = RunRecord.model_validate_json((d / "run.json").read_text())
    assert run.status == "running" and run.source == "data-engine"
    assert (d / "config.json").exists() and (d / "session.json").exists()


def test_create_refuses_existing_dir(tmp_path):
    _writer(tmp_path)
    with pytest.raises(FileExistsError):
        _writer(tmp_path)


def test_fake_rig_run_has_no_session(tmp_path):
    w = _writer(tmp_path, run_id="0309-mock", rig="fake", session=None)
    assert not (tmp_path / "0309-mock" / "session.json").exists()


def test_default_id_uses_ddmm_name():
    rid = default_id("boxA", now=1788180000.0)  # 2026-08-27 UTC
    assert rid.endswith("-boxA") and len(rid.split("-")[0]) == 4


# --- step lifecycle -----------------------------------------------------------

def test_begin_step_allocates_dense_ids_and_dirs(tmp_path):
    w = _writer(tmp_path)
    s0, d0 = w.begin_step({"method": "vs1", "address": None})
    s1, d1 = w.begin_step({"method": "vs1", "address": [3, 0]})
    assert (s0, s1) == (0, 1)
    assert d0.name == "000" and d1.name == "001" and d1.is_dir()


def test_capture_then_fuse_two_phase(tmp_path):
    w = _writer(tmp_path)
    sid, sdir = w.begin_step({"method": "vs1", "address": [3, 0]})
    _capture(w, sid)
    on_disk = StepRecord.model_validate_json((sdir / "step.json").read_text())
    assert on_disk.phase == "captured" and on_disk.geometry is None

    w.write_fused(sid, GEO, _vstate(sid))
    on_disk = StepRecord.model_validate_json((sdir / "step.json").read_text())
    assert on_disk.phase == "fused" and on_disk.geometry.fused_points == 90
    vs = ViewState.model_validate_json((sdir / "view_state.json").read_text())
    assert vs.step_id == sid


def test_roster_flushed_at_first_step_write(tmp_path):
    w = _writer(tmp_path)
    sid, _ = w.begin_step({"method": "vs1", "address": [3, 0]})
    _capture(w, sid)
    run = RunRecord.model_validate_json(
        (tmp_path / "0309-boxA" / "run.json").read_text())
    assert run.steps == [sid]  # crash after this point loses nothing


def test_rejected_step_recorded_without_pose(tmp_path):
    w = _writer(tmp_path)
    sid, sdir = w.begin_step({"method": "vs1", "address": [4, 0]})
    w.mark_step(sid, outcome="rejected", detail="plane gate: tilt 4.2deg")
    on_disk = StepRecord.model_validate_json((sdir / "step.json").read_text())
    assert on_disk.outcome == "rejected" and on_disk.joints_rad is None
    run = RunRecord.model_validate_json(
        (tmp_path / "0309-boxA" / "run.json").read_text())
    assert sid in run.steps  # explicit failure entry, never a silent gap


# --- events -------------------------------------------------------------------

def test_events_append_as_jsonl(tmp_path):
    w = _writer(tmp_path)
    w.event("requested", step_id=1)
    w.event("approved", step_id=1)
    lines = (tmp_path / "0309-boxA" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["kind"] == "approved"


# --- close --------------------------------------------------------------------

def test_close_writes_manifest_over_binaries(tmp_path):
    w = _writer(tmp_path)
    sid, sdir = w.begin_step({"method": "vs1", "address": [3, 0]})
    _capture(w, sid)
    np.save(sdir / "depth_raw.npy", np.zeros((4, 4), dtype=np.uint16))
    (sdir / "rgb.png").write_bytes(b"\x89PNG fake")
    w.write_fused(sid, GEO, _vstate(sid))
    w.close("completed")

    d = tmp_path / "0309-boxA"
    run = RunRecord.model_validate_json((d / "run.json").read_text())
    assert run.status == "completed" and run.closed_at is not None
    man = Manifest.model_validate_json((d / "manifest.json").read_text())
    rel = {p for p in man.files}
    assert "steps/000/depth_raw.npy" in rel and "steps/000/rgb.png" in rel
    assert "run.json" not in rel  # schema JSON validated, not byte-hashed
    entry = man.files["steps/000/rgb.png"]
    assert entry.bytes == 9 and len(entry.sha256) == 64


def test_close_aborted_partial_run_is_valid(tmp_path):
    w = _writer(tmp_path)
    sid, _ = w.begin_step({"method": "vs1", "address": [3, 0]})
    _capture(w, sid)  # never fused
    w.close("aborted")
    run = RunRecord.model_validate_json(
        (tmp_path / "0309-boxA" / "run.json").read_text())
    assert run.status == "aborted" and run.steps == [sid]


def test_no_temp_files_left_behind(tmp_path):
    w = _writer(tmp_path)
    sid, _ = w.begin_step({"method": "vs1", "address": [3, 0]})
    _capture(w, sid)
    w.write_fused(sid, GEO, _vstate(sid))
    w.close("completed")
    strays = [p for p in (tmp_path / "0309-boxA").rglob("*")
              if p.name.startswith(".tmp") or p.suffix == ".tmp"]
    assert strays == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
