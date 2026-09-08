#!/usr/bin/env python3
"""View store: facts tier. Run: p inspection/tests/test_eyes_store.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.store import (FactWriter, FindingWriter, PlanWriter,
                                   RunStore, ViewRecord)

T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]


def _transcript():
    """One subagent transcript, as the eyes tier hands it over: a
    `TranscriptRecord` whose id the writer stamps."""
    from inspection.record.schema import TranscriptRecord
    return TranscriptRecord(transcript_id="unassigned", kind="inspect",
                            step_id=1, t=1.0, model="StubVlm", task="t",
                            prompt="p", turns=[])


def _fresh(tmp):
    """`tmp` IS the store directory — in a run that is the orchestrator
    session's own `ai/<seq>/`, handed in by whoever opened it."""
    return RunStore.create(Path(tmp), h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.35)


def test_add_view_and_query():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        facts = FactWriter(store)
        facts.add_view(cell=None, pose_id=0, cap_dir="000", T_base_cam=T, t=1.0)
        facts.add_view(cell=(3, 1), pose_id=1, cap_dir="001", T_base_cam=T, t=2.0)
        assert store.visited() == {(3, 1)}          # survey is not a cell
        assert len(store.views()) == 2
        v = store.views(cell=(3, 1))[0]
        assert isinstance(v, ViewRecord) and v.cap_dir == "001"
        assert v.T_base_cam.shape == (4, 4)
        cov = store.coverage()
        assert cov.shape == (3, 12) and cov[1, 3] and cov.sum() == 1


def test_flush_and_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        FactWriter(store).add_view(cell=(0, 2), pose_id=1, cap_dir="001",
                                   T_base_cam=T, t=1.0)
        again = RunStore.open(Path(tmp))            # fresh object, disk only
        assert again.visited() == {(0, 2)}
        assert np.allclose(again.views()[0].T_base_cam, T)
        assert (Path(tmp) / "store.json").exists()


def test_no_motion_imports():
    import inspection.eyes.store as m
    src = Path(m.__file__).read_text()
    assert "inspection.motion" not in src and "inspection.run" not in src


def test_agent_writers():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh(tmp)
        plan, finder = PlanWriter(store), FindingWriter(store)
        plan.set_hypothesis("logo likely on far wall")
        plan.set_plan("sweep h=3..5 at v=1")
        plan.note("view 12 shows a fragment at right edge")
        rel = finder.add_finding((3, 1), "partial logo, right edge",
                                 _transcript())
        finder.note("wanted neighbour (4,1) — not captured", cell=(3, 1))
        again = RunStore.open(Path(tmp))
        assert again.hypothesis == "logo likely on far wall"
        assert again.plan.startswith("sweep")
        assert [n["who"] for n in again.notes()] == ["plan", "finding"]
        f = again.findings()[0]
        assert f["cell"] == [3, 1] and (Path(tmp) / rel).exists()


def test_trust_levels_by_construction():
    # The model-facing writers must PHYSICALLY lack geometry verbs.
    for cls in (PlanWriter, FindingWriter):
        api = {m for m in dir(cls) if not m.startswith("_")}
        assert "add_view" not in api, cls
    assert {m for m in dir(PlanWriter) if not m.startswith("_")} == \
        {"set_plan", "set_hypothesis", "note"}
    # `artifacts_dir` is where this tier's images go, not a geometry verb —
    # the trust level is unchanged.
    assert {m for m in dir(FindingWriter) if not m.startswith("_")} == \
        {"add_finding", "artifacts_dir", "note"}


def main():
    test_add_view_and_query(); test_flush_and_reopen(); test_no_motion_imports()
    test_agent_writers(); test_trust_levels_by_construction()
    print("OK test_eyes_store")


if __name__ == "__main__":
    main()
