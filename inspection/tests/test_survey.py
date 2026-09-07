#!/usr/bin/env python3
"""The survey agent and the bearing math. Run: p inspection/tests/test_survey.py

The bearing uses only camera POSITIONS against the reconstruction centroid —
no boresight, no axis convention to get wrong. Pinned against real data: on
2608-aicam the cloud centroid reproduces the six on-grid cells' own azimuths
to <=4.3°, mean 1.9° (2026-08-27). These tests rebuild that geometry
synthetically so it cannot drift silently.

(For the record: the stored `T_base_cam` is OPTICAL convention, +Z boresight —
measured the same day when a ray-intersection variant was tried; with +X the
recovered azimuths were off by 64.6° mean. The centroid method made that
convention irrelevant here, but the fact stands for any future boresight use.)
"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.models import StubVlm
from inspection.eyes.store import FindingWriter, RunStore
from inspection.eyes.agents.survey_agent import SURVEY_RULES, describe_survey
from inspection.eyes.tools import ViewTools
from inspection.eyes.verbs_local import LocalVerbs, StubBackend
from inspection.record.run import Run
from inspection.tests.record_fixtures import make_legacy_run

CENTRE = np.array([0.40, -0.10, 0.12])
R = 0.30


def _pose(az_deg, elev_deg=20.0):
    """Camera pose looking at CENTRE from (az, elev) — +Z boresight (optical).

    Azimuth counted the grid's way: h = 0 faces the base, so the reference
    direction is -CENTRE_xy (view/viewsphere.py:68-69).
    """
    ref = np.arctan2(-CENTRE[1], -CENTRE[0])
    a, e = np.radians(az_deg) + ref, np.radians(elev_deg)
    out = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
    p = CENTRE + R * out
    z = (CENTRE - p) / np.linalg.norm(CENTRE - p)           # boresight
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, p
    return T


def _rig(tmp, cells=((3, 0), (9, 0), (6, 1)), survey_az=45.0):
    root = Path(tmp)
    img = np.zeros((480, 848, 3), np.uint8)
    img[:, :, 2] = 120
    views = [{"cell": None, "pose_id": 0, "t": 0.0, "rgb": img,
              "T_base_cam": _pose(survey_az, elev_deg=35.0)}]
    for i, (h, v) in enumerate(cells, start=1):
        views.append({"cell": (h, v), "pose_id": i, "t": float(i), "rgb": img,
                      "T_base_cam": _pose(h * 30.0, elev_deg=10.0 + 30.0 * v)})
    make_legacy_run(root, views, r=R)
    return Run.load(root)


def _notes(tmp, name="notes"):
    """A separate RunStore — task-5: ViewTools' `writer` still binds to the
    surviving RunStore, independent of the run's own views (`Run` now)."""
    return RunStore.create(Path(tmp) / name, h_bins=12,
                           v_elevs=(10.0, 40.0, 70.0), r=R)


def test_bearing_from_the_recon_centroid_matches_the_taught_azimuth():
    # Replay: the survey's own reconstruction is on disk as fused_cloud.npy;
    # its centroid is the centre. Validated on 2608-aicam: the on-grid cells
    # reproduce their own azimuths to <=4.3 deg by this method.
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp, survey_az=45.0)
        cloud = CENTRE + 0.02 * (np.random.default_rng(0)
                                 .standard_normal((300, 3)))
        np.save(Path(tmp) / "fused_cloud.npy", cloud)
        tools = ViewTools(run)
        rec = [v for v in tools._views() if v.cell is None][0]
        az, hf = tools.survey_bearing(rec)
        assert abs(az - 45.0) < 3.0
        assert 1.0 < hf < 2.0                    # between cells [1] and [2]


def test_bearing_prefers_an_exact_centre_and_survives_having_none():
    # Live at plan time: no cloud file yet, but the Supervisor knows the
    # sphere centre exactly. Neither centre nor cloud -> None, never a guess.
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp, cells=(), survey_az=100.0)
        tools = ViewTools(run)
        rec = [v for v in tools._views() if v.cell is None][0]
        assert tools.survey_bearing(rec) is None          # no cloud, no centre
        az, hf = tools.survey_bearing(rec, centre=CENTRE)
        assert abs(az - 100.0) < 1e-6


def test_describe_survey_runs_the_declaration_over_the_survey_frame():
    with tempfile.TemporaryDirectory() as tmp:
        run = _rig(tmp)
        store = _notes(tmp)
        tools = ViewTools(run, writer=FindingWriter(store))
        verbs = LocalVerbs(StubBackend())
        script = [{"evidence": ["a red box fills the frame"],
                   "reasoning": "one face dominates, a second is compressed",
                   "answer": "a red box; primitive: box; front face-on, "
                             "right side half right; printed mark on front"}]
        f = describe_survey(tools, verbs, FindingWriter(store), StubVlm(script))
        assert f is not None and f.cell is None
        assert "primitive: box" in f.answer
        assert "DECLARE" in f.transcript["prompt"]        # survey rules, not
        assert "inspecting ONE captured" not in f.transcript["prompt"]  # inspect's
        assert len(store.findings()) == 1                 # on the record


def test_no_survey_capture_means_none_not_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        # A run with only an on-grid view, no survey — pose_id starts at 1
        # (the legacy adapter hard-codes pose_id 0 to address=None, so a
        # genuine "no survey" run must never use pose_id 0 at all).
        root = Path(tmp) / "nosurvey"
        img = np.zeros((480, 848, 3), np.uint8)
        make_legacy_run(root, [{"cell": (3, 0), "pose_id": 1, "t": 1.0,
                               "rgb": img, "T_base_cam": _pose(90.0)}], r=R)
        run2 = Run.load(root)
        store2 = _notes(tmp, name="notes2")
        tools = ViewTools(run2, writer=FindingWriter(store2))
        f = describe_survey(tools, LocalVerbs(StubBackend()),
                            FindingWriter(store2), StubVlm([]))
        assert f is None


def test_survey_rules_forbid_guessing_and_planning():
    # The tier split (Anton 2026-08-27): the eyes declare, the planner infers.
    assert "Do NOT guess" in SURVEY_RULES
    assert "you do not plan" in SURVEY_RULES
    for word in ("face-on", "half left", "edge-on left", "hidden right"):
        assert word in SURVEY_RULES              # one vocabulary, both tiers


def main():
    test_bearing_from_the_recon_centroid_matches_the_taught_azimuth()
    test_bearing_prefers_an_exact_centre_and_survives_having_none()
    test_describe_survey_runs_the_declaration_over_the_survey_frame()
    test_no_survey_capture_means_none_not_a_crash()
    test_survey_rules_forbid_guessing_and_planning()
    print("OK test_survey")


if __name__ == "__main__":
    main()
