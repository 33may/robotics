#!/usr/bin/env python3
"""decomposition.json — one polymorphic slot per step, method declared inside.
Run: p vbti/inspection/tests/test_record_decomposition.py
"""
import numpy as np
import pytest

from inspection.record.run import Run
from inspection.record.schema import DecompositionRecord
from inspection.record.writer import write_step_decomposition
from inspection.tests.record_fixtures import T4, make_run

BOX = dict(kind="box", pose=T4, params={"extents": [0.06, 0.05, 0.09]},
           label="box")


def _rec(**over):
    base = dict(step_id=0, source="fused/cloud.npy", n_points=1234,
                method="boxguard", method_version="v3", primitives=[BOX])
    base.update(over)
    return DecompositionRecord(**base)


def test_abstaining_record_carries_no_primitives():
    with pytest.raises(ValueError, match="abstain"):
        _rec(abstain=True, reason="too thin")


def test_method_is_free_form_and_kind_is_structural():
    """Polymorphism: the schema must accept any method name / kind — the
    closed vocabulary is enforced by the producer, never by the record."""
    r = _rec(method="cylguard", method_version="v1",
             primitives=[dict(BOX, kind="cylinder",
                              params={"radius": 0.03, "length": 0.1})])
    assert r.method == "cylguard"


def test_write_then_read_through_step(tmp_path):
    d = make_run(tmp_path)
    p = write_step_decomposition(d, _rec())
    assert p == d / "steps" / "000" / "decomposition.json"
    step = Run.load(d).step(0)
    back = step.decomposition
    assert back is not None and back.method == "boxguard"
    assert back.primitives[0].kind == "box"
    assert np.asarray(back.primitives[0].pose).shape == (4, 4)


def test_write_refuses_a_step_that_does_not_exist(tmp_path):
    d = make_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        write_step_decomposition(d, _rec(step_id=7))


def test_step_without_slot_reads_none(tmp_path):
    d = make_run(tmp_path)
    assert Run.load(d).step(0).decomposition is None


if __name__ == "__main__":
    import subprocess
    import sys
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"]).returncode)
