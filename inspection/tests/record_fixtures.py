"""Shared fixture: build a schema-valid run via RunWriter for record-layer tests."""
import json

import numpy as np

from inspection.record.schema import AnswerRecord, ConfigSnapshot, SessionRecord
from inspection.record.writer import RunWriter

T4 = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]

CONFIG = ConfigSnapshot(
    git_sha="76fabc3",
    cell_yaml={"sha256": "b" * 64, "content": "frames: {}\n"},
    calib={"file": "T_flange_cam_2026-08-19.npy", "sha256": "c" * 64},
)

SESSION = SessionRecord(serial="123622270954", resolution=[848, 480],
                        depth_scale_m_per_unit=1e-4)

GEO = dict(offered=100, kept=90, dropped=10, fused_points=90, source="mask")


def make_run(root, run_id="0309-boxA", *, status="aborted", object="box",
             rig="real", with_answer=False, now=1788180000.0):
    """One fused step (with binaries) + one rejected step + events, closed."""
    w = RunWriter.create(
        root, run_id=run_id, name=run_id.split("-", 1)[1], object=object,
        source="data-engine", rig=rig, question="is there a logo?",
        view_methods=[{"id": "vs1", "kind": "viewsphere",
                       "params": {"h_bins": 12, "v_elevs": [10.0]}}],
        config=CONFIG, session=SESSION if rig == "real" else None, now=now)

    sid, sdir = w.begin_step({"method": "vs1", "address": None}, t_arrived=now)
    np.save(sdir / "depth_raw.npy", np.zeros((4, 4), dtype=np.uint16))
    (sdir / "rgb.png").write_bytes(b"\x89PNG fake image bytes")
    w.write_capture(sid, t_captured=now + 2, joints_rad=[0.1] * 6,
                    T_base_flange=T4, T_base_cam=T4, rgb_rotation_deg=0)
    w.write_fused(sid, GEO, _vstate(sid))

    rid, _ = w.begin_step({"method": "vs1", "address": [4, 0]}, t_arrived=now + 5)
    w.mark_step(rid, outcome="rejected", detail="plane gate")
    w.event("approved", step_id=sid, now=now + 1)

    if with_answer:
        ans = AnswerRecord(verdict="yes", reasoning="logo on survey",
                           step_ids=[sid], transcript_ids=[])
        (w.dir / "answer.json").write_text(ans.model_dump_json())
    w.close(status, now=now + 60)
    return w.dir


def _vstate(step_id):
    from inspection.record.schema import ViewState
    return ViewState(step_id=step_id, candidates=[
        {"address": [3, 0], "pose": T4, "status": "current"}])
