"""Tests for the bake-output GT audit (MAY-173 locdev T3 → P1 gate).

Layer-1 map validation: join a bake's TUM trajectory to the dump's GT base rows
by stamp, rigid-align (Umeyama, no scale — GT is the eval ORACLE here, never a
map input, per the no-GT-in-localization-pipeline rule), and report ATE stats
against the P1 gate (full-drive backbone mean ≤ 0.15 m).

Offline tooling in the `hum` env — not brain code, no marker.
"""

import json
import math

import numpy as np
import pytest

from humanoid.logic.simulation.mapping.map_audit import (
    ate_stats,
    audit_tum_vs_gt,
    main,
    umeyama_align,
)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ─── alignment ────────────────────────────────────────────────────────────────

def test_umeyama_align_recovers_rigid_transform():
    rng = np.random.default_rng(3)
    G = rng.uniform(-10, 10, size=(40, 3))
    R, t = _rz(0.7), np.array([5.0, -2.0, 1.5])
    P = (R @ G.T).T + t  # estimated = GT moved by a rigid transform
    Pa = umeyama_align(P, G)
    assert np.allclose(Pa, G, atol=1e-9)


def test_umeyama_align_no_scale_compensation():
    """Scale error must SURVIVE alignment (stereo is metric; hiding scale drift
    would fake map quality)."""
    G = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
    Pa = umeyama_align(G * 1.1, G)
    err = np.linalg.norm(Pa - G, axis=1)
    assert err.max() > 1.0  # 10% scale over 30 m cannot vanish


# ─── stats ────────────────────────────────────────────────────────────────────

def test_ate_stats_fields():
    e = np.array([0.1, 0.2, 0.3, 10.0])
    s = ate_stats(e)
    assert s["mean"] == pytest.approx(e.mean())
    assert s["max"] == pytest.approx(10.0)
    assert s["p50"] == pytest.approx(np.percentile(e, 50))
    assert s["p95"] == pytest.approx(np.percentile(e, 95))
    assert s["n"] == 4


# ─── end-to-end: TUM vs dump GT ───────────────────────────────────────────────

@pytest.fixture()
def gt_dump(tmp_path):
    """Dump dir with base rows on a curve; stamps every 0.2 s of sim time."""
    rows = []
    for i in range(30):
        stamp = 400_000_000 + i * 200_000_000
        rows.append({"cam": "base", "stamp_ns": stamp,
                     "x": 0.3 * i, "y": math.sin(0.2 * i), "yaw": 0.05 * i})
    (tmp_path / "poses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return tmp_path, rows


def _write_tum(path, rows, transform=None, corrupt_idx=None):
    R, t = transform if transform else (np.eye(3), np.zeros(3))
    lines = []
    for k, r in enumerate(rows):
        p = R @ np.array([r["x"], r["y"], 0.0]) + t
        if k == corrupt_idx:
            p = p + np.array([5.0, 0.0, 0.0])
        lines.append(
            f'{r["stamp_ns"]/1e9:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} 0 0 0 1'
        )
    path.write_text("\n".join(lines) + "\n")


def test_audit_perfect_trajectory_scores_zero(gt_dump, tmp_path):
    dump, rows = gt_dump
    tum = tmp_path / "est.tum"
    _write_tum(tum, rows, transform=(_rz(1.1), np.array([3.0, 4.0, -1.0])))
    rep = audit_tum_vs_gt(tum, dump)
    assert rep["n"] == 30
    assert rep["mean"] < 1e-6 and rep["max"] < 1e-6


def test_audit_flags_outlier_and_skips_unmatched(gt_dump, tmp_path):
    dump, rows = gt_dump
    tum = tmp_path / "est.tum"
    _write_tum(tum, rows, corrupt_idx=15)
    # one extra TUM line with a stamp not in GT → must be skipped, not crash
    with open(tum, "a") as f:
        f.write("999.000000000 0 0 0 0 0 0 1\n")
    rep = audit_tum_vs_gt(tum, dump)
    assert rep["n"] == 30
    assert rep["max"] > 3.0  # the 5 m outlier survives (minus alignment leakage)


def test_cli_prints_json(gt_dump, tmp_path, capsys):
    dump, rows = gt_dump
    tum = tmp_path / "est.tum"
    _write_tum(tum, rows)
    rc = main(["--tum", str(tum), "--dump", str(dump)])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["mean"] < 1e-6
