#!/usr/bin/env python3
"""Replay a captured run into a RunStore. Run: p inspection/tests/test_eyes_replay.py"""
import json
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.replay import load_run
from inspection.eyes.store import PlanWriter, RunStore

T = np.eye(4)


def _synth_run(root):
    """run.json with survey + 3 cell turns (one failed -> no dir)."""
    turns = [{"step": 1, "target": "survey", "t": 1.0, "result": "ok"},
             {"step": 2, "target": [3, 1], "t": 2.0, "result": "+100 pts"},
             {"step": 3, "target": [4, 1], "t": 3.0,
              "result": "capture failed: no frames"},          # no dir 002
             {"step": 4, "target": [5, 2], "t": 4.0, "result": "+90 pts"}]
    (root / "run.json").write_text(json.dumps(
        {"q_survey": [0.0] * 6, "r": 0.35, "turns": turns}))
    for pose_id, name in [(0, "000"), (1, "001"), (3, "003")]:
        d = root / name; d.mkdir()
        (d / "meta.json").write_text(json.dumps(
            {"pose_id": pose_id, "timestamp": 10.0 + pose_id,
             "joints_rad": [0.0] * 6, "T_base_flange": T.tolist(),
             "T_base_cam": T.tolist()}))
        (d / "rgb.png").write_bytes(b"png")


def test_load_synthetic_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _synth_run(root)
        store = load_run(root)
        assert store.visited() == {(3, 1), (5, 2)}    # failed cell absent
        assert [v.cell for v in store.views()] == [None, (3, 1), (5, 2)]
        assert store.views(cell=(5, 2))[0].cap_dir == "003"   # pose_id 3 = 3rd cell turn


def test_reload_preserves_agent_tiers():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _synth_run(root)
        PlanWriter(load_run(root)).set_hypothesis("h1")
        store = load_run(root)                         # idempotent re-load
        assert store.hypothesis == "h1"
        assert len(store.views()) == 3                 # not duplicated


def test_real_run_if_present():
    real = Path("inspection/data/runs/2108-d")
    if not (real / "run.json").exists():
        print("  (skip: 2108-d not present)"); return
    store = load_run(real)
    assert len(store.views()) == 5 and len(store.visited()) == 4
    assert store.views()[0].cell is None               # survey first


def main():
    test_load_synthetic_run(); test_reload_preserves_agent_tiers()
    test_real_run_if_present()
    print("OK test_eyes_replay")


if __name__ == "__main__":
    main()
