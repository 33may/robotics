"""boxguard on synthetic shells: known extents, known yaw, slim contract.

The synth box is the only cuboid whose answer we know exactly. boxguard's
conventions shift the raw extents deterministically: the table plane is
the cloud's 1st z-percentile minus the recorder's 8 mm margin (so height
reads ~8 mm over the shell), and footprint extents are the OUTER noise
envelope (a few mm over the nominal walls). Windows below encode those
conventions, not looseness.
"""
import json

import numpy as np

from inspection.fit.box import boxguard, boxguard_fit
from inspection.fit.contract import decomposition_from_json
from inspection.fit.tests import synth

EXTENTS = (0.06, 0.05, 0.09)
CENTER = (0.1, -0.35, 0.06)


def _yawed(P, yaw, center=CENTER):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ctr = np.asarray(center, dtype=float)
    return (P - ctr) @ R.T + ctr


def test_recovers_extents_of_an_axis_aligned_shell():
    P = synth.box(extents=EXTENTS, center=CENTER, seed=0)
    m = boxguard(P)
    V = np.asarray(m.vertices)
    dims = V.max(0) - V.min(0)
    assert 0.058 <= dims[0] <= 0.072      # outer envelope: never under
    assert 0.048 <= dims[1] <= 0.062
    assert 0.088 <= dims[2] <= 0.108      # + table margin (8 mm) + fuzz
    ctr = 0.5 * (V.max(0) + V.min(0))
    assert np.linalg.norm(ctr[:2] - CENTER[:2]) < 0.005


def test_recovers_yaw_of_a_rotated_shell():
    yaw_true = np.radians(20.0)
    P = _yawed(synth.box(extents=EXTENTS, center=CENTER, seed=1), yaw_true)
    d = boxguard_fit(P)
    R = d.primitives[0].axes
    yaw_est = float(np.arctan2(R[1, 0], R[0, 0]))
    err = (yaw_est - yaw_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
    assert abs(np.degrees(err)) < 2.0     # mod-90: a box has no front


def test_fit_speaks_the_slim_contract():
    P = synth.box(extents=EXTENTS, center=CENTER, seed=2)
    d = boxguard_fit(P)
    assert d.method == "boxguard" and not d.abstain
    assert [p.kind for p in d.primitives] == ["box"]
    prim = d.primitives[0]
    assert np.allclose(prim.axes @ prim.axes.T, np.eye(3), atol=1e-9)
    assert prim.axes[2, 2] == 1.0         # +Z is gravity, always
    conf = prim.params["side_conf"]
    assert set(conf) == {"x+", "x-", "y+", "y-"}
    assert all(conf.values())             # full shell: every wall seen
    back = decomposition_from_json(json.loads(json.dumps(d.to_json())))
    assert back == d


def test_deterministic_across_calls():
    P = synth.box(extents=EXTENTS, center=CENTER, seed=3)
    a = np.asarray(boxguard(P).vertices)
    b = np.asarray(boxguard(P).vertices)
    assert np.array_equal(a, b)
