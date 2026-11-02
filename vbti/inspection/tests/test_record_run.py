#!/usr/bin/env python3
"""Run — the one read door over a recorded run (flows-design.md §4)."""
import json

import numpy as np
import pytest

from inspection.record.run import Run
from inspection.record.writer import RunWriter
from inspection.tests.record_fixtures import CONFIG, SESSION, T4, make_run


def _legacy_run(root, run_id="2408-cup9", *, with_answer=False,
                with_session=True, old_meta_on_view=1):
    """Mimic the 2408-era layout: run.json{q_survey,r,turns} + NNN/meta.json.

    Copied from inspection/tests/test_record_legacy.py — kept standalone here
    rather than imported, per the task-3 brief.
    """
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


def test_load_exposes_records(tmp_path):
    make_run(tmp_path, run_id="0709-box1", object="box")
    run = Run.load(tmp_path / "0709-box1")
    assert run.record.id == "0709-box1"
    assert run.provenance == "native"
    assert run.session is not None and run.config is not None


def test_steps_and_survey(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    assert [s.id for s in run.steps] == run.record.steps
    assert run.survey is not None and run.survey.id == 0
    assert run.step(run.steps[-1].id).record.step_id == run.steps[-1].id
    with pytest.raises(KeyError):
        run.step(999)


def test_at_excludes_rejected_steps(tmp_path):
    """make_run's non-survey step lands at address [4, 0] but is REJECTED
    (plane gate) — at() only matches captured steps, so it must miss."""
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    assert run.at((4, 0)) is None
    assert run.at((9, 9)) is None


def test_at_finds_latest_captured_step_for_address(tmp_path):
    """make_run only ever captures the survey (view.address=None), so the
    positive-match/"latest wins" behaviour of at() is exercised against a
    small hand-built run with two captures at the same address."""
    w = RunWriter.create(
        tmp_path, run_id="multi", name="multi", object="box",
        source="data-engine", rig="real",
        view_methods=[{"id": "vs1", "kind": "viewsphere",
                       "params": {"h_bins": 12, "v_elevs": [10.0]}}],
        config=CONFIG, session=SESSION, now=1000.0)
    for i in range(2):
        sid, sdir = w.begin_step({"method": "vs1", "address": [3, 0]},
                                 t_arrived=1000.0 + i)
        w.write_capture(sid, t_captured=1000.0 + i, joints_rad=[0.1] * 6,
                        T_base_flange=T4, T_base_cam=T4, rgb_rotation_deg=0)
    w.close("aborted", now=1100.0)

    run = Run.load(w.dir)
    hit = run.at((3, 0))
    assert hit is not None and hit.record.view.address == [3, 0]
    assert hit.id == run.steps[-1].id  # latest capture, not the first
    assert run.at((9, 9)) is None


def test_T_base_cam_is_4x4_array(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    T = run.captured[0].T_base_cam
    assert isinstance(T, np.ndarray) and T.shape == (4, 4)


def test_fused_and_events_and_validate(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run = Run.load(tmp_path / "0709-box1")
    pts, colors = run.fused() or (None, None)
    assert pts is None or pts.ndim == 2          # fixture doesn't write fused/
    assert isinstance(run.events, list)
    assert run.validate().run_id == "0709-box1"


def test_fused_prefers_new_layout_then_falls_back_to_legacy(tmp_path):
    make_run(tmp_path, run_id="0709-box1")
    run_dir = tmp_path / "0709-box1"

    np.save(run_dir / "fused_cloud.npy", np.zeros((5, 3)))
    np.save(run_dir / "fused_colors.npy", np.zeros((5, 3), dtype=np.uint8))
    run = Run.load(run_dir)
    pts, colors = run.fused()
    assert pts.shape == (5, 3) and colors.shape == (5, 3)

    (run_dir / "fused").mkdir()
    np.save(run_dir / "fused" / "cloud.npy", np.zeros((7, 3)))
    run = Run.load(run_dir)  # new layout wins over the legacy files
    pts, colors = run.fused()
    assert pts.shape == (7, 3) and colors is None  # no fused/colors.npy yet


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Run.load(tmp_path / "nope")


def test_rgb_upright_and_raw_differ_only_when_rotated(tmp_path):
    make_run(tmp_path, run_id="0709-rot")
    run = Run.load(tmp_path / "0709-rot")
    step = run.captured[0]
    import cv2
    marked = np.zeros((4, 6, 3), np.uint8); marked[0, 0] = (255, 0, 0)
    cv2.imwrite(str(step.dir / "rgb.png"), cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    up, raw = step.rgb(upright=True), step.rgb(upright=False)
    if step.record.rgb_rotation_deg == 180:
        assert not np.array_equal(up, raw)
        assert np.array_equal(np.rot90(up, 2), raw)
    else:
        assert np.array_equal(up, raw)


def test_depth_and_missing_binaries_are_none(tmp_path):
    make_run(tmp_path, run_id="0709-bin")
    step = Run.load(tmp_path / "0709-bin").captured[0]
    np.save(step.dir / "depth_aligned.npy", np.ones((4, 6), np.uint16))
    assert step.depth().shape == (4, 6)
    assert step.cloud() is None and step.mask() is None


def test_rgb_rotates_back_when_stored_rotated(tmp_path):
    """make_run's fixture defaults to rgb_rotation_deg=0, so the test above
    only ever walks the equal branch — this drives Step.rgb() through the
    actual 180 branch: rgb.png is written UPRIGHT (per convention), and
    upright=False must rotate it back to match the raw pose."""
    make_run(tmp_path, run_id="0709-rot180", rgb_rotation_deg=180)
    step = Run.load(tmp_path / "0709-rot180").captured[0]
    assert step.record.rgb_rotation_deg == 180
    import cv2
    stored = np.zeros((4, 6, 3), np.uint8); stored[0, 0] = (255, 0, 0)
    cv2.imwrite(str(step.dir / "rgb.png"), cv2.cvtColor(stored, cv2.COLOR_RGB2BGR))
    up, raw = step.rgb(upright=True), step.rgb(upright=False)
    assert np.array_equal(up, stored)          # upright=True returns the file as stored
    assert not np.array_equal(up, raw)
    assert np.array_equal(np.rot90(up, 2), raw)


def test_depth_rotates_back_when_stored_rotated(tmp_path):
    """Same one rotation policy applies to depth_aligned.npy (Conventions
    .rotated_artifacts) — reconstruction consumers need it in RAW frame."""
    make_run(tmp_path, run_id="0709-rot180d", rgb_rotation_deg=180)
    step = Run.load(tmp_path / "0709-rot180d").captured[0]
    stored = np.arange(24, dtype=np.uint16).reshape(4, 6)
    np.save(step.dir / "depth_aligned.npy", stored)
    up, raw = step.depth(upright=True), step.depth(upright=False)
    assert np.array_equal(up, stored)
    assert not np.array_equal(up, raw)
    assert np.array_equal(np.rot90(up, 2), raw)


def test_mask_rotates_back_when_stored_rotated(tmp_path):
    """mask.png shares the same rotated_artifacts policy as rgb/depth."""
    make_run(tmp_path, run_id="0709-rot180m", rgb_rotation_deg=180)
    step = Run.load(tmp_path / "0709-rot180m").captured[0]
    import cv2
    stored = np.zeros((4, 6), np.uint8); stored[0, 0] = 255
    cv2.imwrite(str(step.dir / "mask.png"), stored)
    up, raw = step.mask(upright=True), step.mask(upright=False)
    assert np.array_equal(up, stored > 0)
    assert not np.array_equal(up, raw)
    assert np.array_equal(np.rot90(up, 2), raw)


def test_derived_probes_fits_then_flat_layout_then_none(tmp_path):
    """derived/<id> mirrors runs/<id> as a sibling of the run's own root;
    probes the legacy fits/<method> bucket, then the flat derive.py layout."""
    runs_root = tmp_path / "data" / "runs"
    make_run(runs_root, run_id="0709-box1")
    run = Run.load(runs_root / "0709-box1")

    dbase = tmp_path / "data" / "derived" / "0709-box1"
    (dbase / "fits" / "cyl").mkdir(parents=True)
    (dbase / "bytecount").mkdir(parents=True)

    assert run.derived("cyl") == dbase / "fits" / "cyl"
    assert run.derived("bytecount") == dbase / "bytecount"
    assert run.derived("nope") is None


def test_legacy_run_loads_through_the_same_door(tmp_path):
    d = _legacy_run(tmp_path, "2408-old")        # helper copied from test_record_legacy
    run = Run.load(d)
    assert run.provenance == "legacy"
    assert run.survey is not None
    assert run.at((3, 0)) is not None            # the turns-join address
    assert run.captured[0].dir.name.isdigit()    # old layout: <run>/NNN/


def test_refresh_picks_up_new_steps(tmp_path):
    from inspection.record.writer import RunWriter
    from inspection.tests.record_fixtures import make_config
    w = RunWriter.create(tmp_path, run_id="0709-live", name="live",
                         source="live", config=make_config())
    run = Run.load(tmp_path / "0709-live")
    assert run.steps == []
    sid, sdir = w.begin_step({"method": "vs", "address": None})
    w.write_capture(sid, t_captured=1.0, joints_rad=[0.0] * 6,
                    T_base_flange=np.eye(4).tolist(),
                    T_base_cam=np.eye(4).tolist(), rgb_rotation_deg=0)
    run.refresh()
    assert [s.id for s in run.steps] == [0]


def _write_airun(d):
    """Build ai/000/ by hand with real, minimal-valid schema instances."""
    from inspection.record.schema import (AIRunRecord, AnswerRecord, MenuDef,
                                          MenuInput, TranscriptRecord)
    d.mkdir(parents=True)
    (d / "transcripts").mkdir()
    (d / "menu").mkdir()

    airun = AIRunRecord(seq=0, mode="live", orchestrator_model="claude-x",
                        menu_id="menu-1", menu_hash="a" * 64)
    (d / "airun.json").write_text(airun.model_dump_json())

    t = TranscriptRecord(transcript_id="t000", kind="survey", step_id=1, t=1.0,
                         model="vlm-x", task="look for logo",
                         prompt="describe the object")
    (d / "transcripts" / "t000.json").write_text(t.model_dump_json())

    menu = MenuDef(menu_id="menu-1", version="1.0.0", content_hash="b" * 64,
                   code_sha="deadbeef", verbs=["move", "capture"])
    (d / "menu" / "menu_def.json").write_text(menu.model_dump_json())

    mi = MenuInput(turn=0, step_id=1, t=1.0)
    with (d / "menu" / "inputs.jsonl").open("w") as f:
        f.write(mi.model_dump_json() + "\n")

    answer = AnswerRecord(verdict="yes", reasoning="logo visible on survey")
    (d / "answer.json").write_text(answer.model_dump_json())

    (d / "trace.jsonl").write_text("")


def test_ai_runs_scan_and_parse(tmp_path):
    make_run(tmp_path, run_id="0709-ai")
    _write_airun(tmp_path / "0709-ai" / "ai" / "000")   # helper in this test file
    run = Run.load(tmp_path / "0709-ai")
    assert len(run.ai) == 1
    a = run.ai[0]
    assert a.seq == 0 and a.record.orchestrator_model
    assert len(a.transcripts) == 1 and a.transcripts[0].step_id == 1
    assert a.answer is not None and a.menu is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
