#!/usr/bin/env python3
"""The brain's readers stand on Run (flows-design §4 'consumers ported')."""
import pytest

from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run, make_run


def test_viewtools_over_run(tmp_path):
    # A real addressed capture, not the survey (task-5 review, 2026-09-07):
    # `view_at` matches the EXACT recorded address — the survey's is None
    # and must never satisfy a cell tuple — so the hit case needs a genuine
    # on-grid capture, same as test_eyes_tools.py's latest-wins test.
    make_legacy_run(tmp_path / "0709-vt", [{"cell": (3, 1), "pose_id": 1}])
    run = Run.load(tmp_path / "0709-vt")
    from inspection.eyes.tools import ViewTools
    vt = ViewTools(run)
    assert vt.view_at((3, 1)) is not None
    assert vt.view_at((9, 9)) is None


def test_has_survey_over_run(tmp_path):
    from inspection.brain.live import has_survey
    assert has_survey(tmp_path / "absent") is False
    make_run(tmp_path, run_id="0709-hs")
    assert has_survey(tmp_path / "0709-hs") is True


def test_replay_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import inspection.eyes.replay  # noqa: F401
