#!/usr/bin/env python3
"""Live per-step fitting: a fused step records the backfill's exact slot.

`machine.py:_fit_step` is the LIVE half of the one decomposition path —
same registry (`fit/apply.py:METHODS`), same record builder (`record_of`),
same writer — so a run recorded with `fit="boxguard"` must be
indistinguishable from one backfilled offline. These tests drive the real
Supervisor over the real FakeRig (the `test_flow_a` composition: only the
planner and the publisher are stubbed) and read the slot back through the
one read door.

Run: p -m pytest inspection/tests/test_live_fit.py
"""
import numpy as np

from inspection.record.run import Run
from inspection.tests.test_flow_a import (DirectPlanSupervisor, StubBus,
                                          StubPub, survey, wait_for)


def start(tmp_path, monkeypatch, pub=None, name="0709-mockfit"):
    """A running mock Supervisor (fit="boxguard" — the mock's default wiring).

    Same shape as `test_flow_a.start`, with the publisher injectable: the
    guarded-publish test needs a pub that lacks `publish_decomposition`.
    """
    from inspection.ui import mock

    monkeypatch.setattr(mock, "Supervisor", DirectPlanSupervisor)
    run_dir = tmp_path / name
    sup = mock.start_mock(StubBus(), StubPub() if pub is None else pub,
                          run_dir)
    return sup, run_dir


def finish(sup, run_dir):
    """Shut the mock down and wait for its writer's close receipt."""
    sup.request_shutdown()
    wait_for(lambda: sup.phase == "done", msg="shutdown")
    wait_for(lambda: (run_dir / "manifest.json").exists(), msg="writer closed")


def test_a_fused_step_records_the_backfills_decomposition(tmp_path,
                                                          monkeypatch):
    """One approved turn -> steps/000/decomposition.json, byte-honest:
    method from the registry, n_points the exact length of the cloud that
    `write_fused` saved a moment earlier, and the run still validates."""
    pub = StubPub()
    sup, run_dir = start(tmp_path, monkeypatch, pub=pub)
    try:
        survey(sup)
        assert sup.survey_state == "visited"
    finally:
        finish(sup, run_dir)

    step_dir = run_dir / "steps" / "000"
    assert (step_dir / "decomposition.json").exists(), "no live slot written"
    run = Run.load(run_dir)
    rec = run.step(0).decomposition
    assert rec is not None and rec.method == "boxguard"
    assert rec.source == "fused/cloud.npy"
    cloud = np.load(step_dir / "fused" / "cloud.npy")
    assert rec.n_points == len(cloud), (rec.n_points, len(cloud))
    assert run.validate().ok, run.validate().problems

    # The UI hook was offered the record as its JSON dict — the same bytes
    # the writer put on disk, so the scene can never draw a different box.
    pubs = [a for (name, a, kw) in pub.calls if name == "publish_decomposition"]
    assert pubs and pubs[-1][0]["method"] == "boxguard"
    assert pubs[-1][0]["n_points"] == rec.n_points


def test_a_raising_fitter_lands_as_a_loud_abstain(tmp_path, monkeypatch):
    """A fitter exception never kills the turn: the slot records an abstain
    whose reason names the error (apply_run's exact convention), the cell is
    still marked visited, and the run still validates."""
    from inspection.fit import apply as applymod

    def boom(P):
        raise RuntimeError("no walls at all")

    monkeypatch.setitem(applymod.METHODS, "boxguard", boom)
    sup, run_dir = start(tmp_path, monkeypatch)
    try:
        survey(sup)
        assert sup.survey_state == "visited", "a bad fit cost the turn"
    finally:
        finish(sup, run_dir)

    run = Run.load(run_dir)
    rec = run.step(0).decomposition
    assert rec is not None and rec.abstain
    assert rec.reason.startswith("fitter error"), rec.reason
    assert "no walls at all" in rec.reason
    assert rec.primitives == []
    assert run.validate().ok, run.validate().problems


def test_a_lying_fitter_is_recorded_as_a_mismatch_abstain(tmp_path,
                                                          monkeypatch):
    """Registry-name honesty, live-loop flavour: where the backfill crashes
    on a fitter whose Decomposition.method contradicts the registry name,
    the live loop logs + abstains — the loop survives, the slot says why."""
    from inspection.fit import apply as applymod
    from inspection.fit.contract import Decomposition

    monkeypatch.setitem(
        applymod.METHODS, "boxguard",
        lambda P: Decomposition(method="other", version="v0"))
    sup, run_dir = start(tmp_path, monkeypatch)
    try:
        survey(sup)
        assert sup.survey_state == "visited"
    finally:
        finish(sup, run_dir)

    rec = Run.load(run_dir).step(0).decomposition
    assert rec is not None and rec.abstain
    assert rec.method == "boxguard"          # the registry name, honestly
    assert "mismatch" in rec.reason and "other" in rec.reason


class NoDecompPub(StubPub):
    """A publisher that PREDATES `publish_decomposition`.

    The machine's publish is `getattr`-guarded precisely so the UI hook can
    land in a separate change: a pub without the method must cost nothing
    but the live box — never the record, never the turn.
    """

    def __getattr__(self, name):
        if name == "publish_decomposition":
            raise AttributeError(name)
        return super().__getattr__(name)


def test_a_pub_without_the_decomposition_hook_is_harmless(tmp_path,
                                                          monkeypatch):
    sup, run_dir = start(tmp_path, monkeypatch, pub=NoDecompPub())
    try:
        survey(sup)
        assert sup.survey_state == "visited"
    finally:
        finish(sup, run_dir)

    run = Run.load(run_dir)
    assert run.step(0).decomposition is not None, "the record was lost"
    assert run.validate().ok, run.validate().problems


if __name__ == "__main__":
    import subprocess
    import sys

    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
