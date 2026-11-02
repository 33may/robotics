#!/usr/bin/env python3
"""settle_capture: capture -> plane gate -> survey check -> fuse.
Run: p inspection/tests/test_settle.py"""
import threading

from inspection.cell.geometry import CloudAccumulator
from inspection.run import settle
from inspection.run.settle import SettleResult, settle_capture


def _fake_cap(tmp_path):
    from inspection.run.rigs import FakeRig
    rig = FakeRig(None, threading.Event())
    # `tmp_path` stands in for the step dir a `RunWriter.begin_step` hands out.
    return rig, rig.capture(0, tmp_path)


def test_settle_accepts_a_good_capture(tmp_path):
    rig, cap = _fake_cap(tmp_path)
    acc = CloudAccumulator()
    res = settle_capture(cap, rig, None, acc, is_survey=True)
    assert res.ok and res.npts > 0 and len(acc.points) > 0
    assert res.detail == ""


def test_settle_rejects_bad_plane(tmp_path, monkeypatch):
    rig, cap = _fake_cap(tmp_path)
    monkeypatch.setattr(settle, "_plane_error", lambda plane: "table at z=90mm")
    res = settle.settle_capture(cap, rig, None, CloudAccumulator(), is_survey=False)
    assert not res.ok and "rejected" in res.detail
    assert res.detail == "view rejected: table at z=90mm"


def test_settle_rejects_survey_with_no_object(tmp_path, monkeypatch):
    rig, cap = _fake_cap(tmp_path)
    acc = CloudAccumulator()
    # Force the object_view result to carry no centroid, as a real capture
    # of an empty table would — without touching the accumulator.
    import inspection.run.settle as settle_mod
    real_object_view = settle_mod.object_view

    def _no_centroid(*a, **kw):
        view, seg = real_object_view(*a, **kw)
        view = dict(view)
        view["centroid"] = None
        return view, seg

    monkeypatch.setattr(settle_mod, "object_view", _no_centroid)
    res = settle_capture(cap, rig, None, acc, is_survey=True)
    assert not res.ok and res.detail == "NO OBJECT above the table"
    assert len(acc.points) == 0


def test_settle_result_is_the_produced_dataclass():
    res = SettleResult(ok=True, detail="", view=None, seg=None, npts=0, dropped=0)
    assert res.ok and res.npts == 0
