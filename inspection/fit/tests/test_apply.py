#!/usr/bin/env python3
"""fit/apply.py — the one backfill/live compute path for decomposition.json."""
import numpy as np
import pytest

from inspection.fit import apply as applymod
from inspection.fit.contract import Decomposition, Primitive, pose_from
from inspection.fit.tests import synth
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_run


@pytest.fixture
def stub_method():
    """A registered fake fitter — tests the mechanism, not boxguard."""
    def stub(P):
        prim = Primitive("box", pose_from(np.eye(3), P.mean(0)),
                         {"extents": [0.1, 0.1, 0.1]}, "box")
        return Decomposition(method="stub", version="v0", primitives=[prim])
    applymod.METHODS["stub"] = stub
    yield "stub"
    del applymod.METHODS["stub"]


def test_apply_fills_the_slot_and_is_idempotent(tmp_path, stub_method):
    d = make_run(tmp_path)
    stats = applymod.apply_run(d, method="stub")
    assert stats["fitted"] == 1          # the fused step
    assert stats["missing"] == 0         # rejected step has no dir entry
    rec = Run.load(d).step(0).decomposition
    assert rec is not None and rec.method == "stub"
    assert rec.n_points == 3             # the fixture's fused cloud
    # idempotent: second pass skips, content untouched
    assert applymod.apply_run(d, method="stub")["skipped"] == 1
    # overwrite deliberately replaces its own product
    assert applymod.apply_run(d, method="stub", overwrite=True)["fitted"] == 1


def test_fitter_error_is_recorded_as_loud_abstain(tmp_path, capsys):
    """A raising fitter must not kill a backfill — it lands as an abstain
    whose reason names the error, and the terminal screams."""
    def boom(P):
        raise RuntimeError("no walls at all")
    applymod.METHODS["boom"] = boom
    try:
        d = make_run(tmp_path)
        stats = applymod.apply_run(d, method="boom")
    finally:
        del applymod.METHODS["boom"]
    assert stats["abstained"] == 1 and stats["fitted"] == 0
    rec = Run.load(d).step(0).decomposition
    assert rec.abstain and rec.reason.startswith("fitter error")
    assert "!!" in capsys.readouterr().out


def test_boxguard_fits_even_a_toy_cloud_without_abstaining(tmp_path):
    """Documented reality, not aspiration: boxguard v3 has NO maturity
    gate — 3 points still produce a 'box'. The slot records it honestly;
    gating is the fitter's future problem, not the mechanism's."""
    d = make_run(tmp_path)
    stats = applymod.apply_run(d, method="boxguard")
    assert stats["fitted"] == 1 and stats["abstained"] == 0


def test_boxguard_end_to_end_on_a_synth_shell(tmp_path):
    d = make_run(tmp_path)
    cloud = synth.box(extents=(0.06, 0.05, 0.09), center=(0.1, -0.35, 0.06))
    np.save(d / "steps" / "000" / "fused" / "cloud.npy", cloud)  # fixture, not evidence
    stats = applymod.apply_run(d, method="boxguard")
    assert stats["fitted"] == 1
    rec = Run.load(d).step(0).decomposition
    assert rec.method == "boxguard" and not rec.abstain
    assert rec.primitives[0].kind == "box"
    ext = rec.primitives[0].params["extents"]
    assert 0.05 < ext[0] < 0.08 and 0.04 < ext[1] < 0.07


if __name__ == "__main__":
    import subprocess
    import sys
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"]).returncode)
