#!/usr/bin/env python3
"""The brain's readers stand on Run (flows-design §4 'consumers ported')."""
import pytest

from inspection.record.run import Run
from inspection.tests.record_fixtures import make_run


def test_viewtools_over_run(tmp_path):
    make_run(tmp_path, run_id="0709-vt")
    run = Run.load(tmp_path / "0709-vt")
    from inspection.eyes.tools import ViewTools
    vt = ViewTools(run)
    # make_run's only CAPTURED step is the survey (step 0, address None —
    # its sibling step is outcome="rejected", never a view). `view_at`
    # matches on each step's raw recorded address, defaulting a None address
    # to (0, 0) — true only of the survey — so (0, 0) is the one cell this
    # fixture actually has a hit for; (9, 9) is a genuine miss either way.
    cell = tuple(run.captured[-1].record.view.address or (0, 0))
    assert vt.view_at(cell) is not None
    assert vt.view_at((9, 9)) is None


def test_has_survey_over_run(tmp_path):
    from inspection.brain.live import has_survey
    assert has_survey(tmp_path / "absent") is False
    make_run(tmp_path, run_id="0709-hs")
    assert has_survey(tmp_path / "0709-hs") is True


def test_replay_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import inspection.eyes.replay  # noqa: F401
