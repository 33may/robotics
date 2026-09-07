#!/usr/bin/env python3
"""Run — the one read door over a recorded run (flows-design.md §4)."""
import numpy as np
import pytest

from inspection.record.run import Run
from inspection.record.writer import RunWriter
from inspection.tests.record_fixtures import CONFIG, SESSION, T4, make_run


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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
