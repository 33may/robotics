"""Shared fixture: build a schema-valid run via RunWriter for record-layer tests."""
import json
from pathlib import Path

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


def make_config():
    """A fresh minimal valid ConfigSnapshot (extracted from CONFIG above)."""
    return ConfigSnapshot(
        git_sha="76fabc3",
        cell_yaml={"sha256": "b" * 64, "content": "frames: {}\n"},
        calib={"file": "T_flange_cam_2026-08-19.npy", "sha256": "c" * 64},
    )

GEO = dict(offered=100, kept=90, dropped=10, fused_points=90, source="mask")


def make_run(root, run_id="0309-boxA", *, status="aborted", object="box",
             rig="real", with_answer=False, now=1788180000.0,
             rgb_rotation_deg=0):
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
                    T_base_flange=T4, T_base_cam=T4,
                    rgb_rotation_deg=rgb_rotation_deg)
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


def make_legacy_run(root, views, *, r=0.35):
    """A pre-schema run dir (run.json turns + NNN/meta.json), loadable
    through `Run.load`'s legacy branch (task-5 port: eyes-tier tests used to
    build these by hand through `RunStore.create` + `FactWriter.add_view`;
    now they need a real run directory `Run.load` can adapt).

    `views` is a list of dicts, one per pose_id in capture order:
      {"cell": (h, v) | None, "pose_id": int (default: list index),
       "t": float (default: pose_id), "T_base_cam": 4x4 (default: identity),
       "rgb": HxWx3 uint8 (default: not written)}
    `cell=None` marks the survey turn; `record/legacy.py:adapt_run` also
    hard-codes pose_id 0 to address=None regardless, so a genuine "no
    survey" run is built by giving every view an explicit non-zero pose_id.
    Returns `root`.
    """
    import cv2

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    turns = [{"target": "survey" if v.get("cell") is None else list(v["cell"]),
              "t": v.get("t", float(i))} for i, v in enumerate(views)]
    (root / "run.json").write_text(json.dumps(
        {"q_survey": [0.0] * 6, "r": r, "turns": turns}))
    ident = np.eye(4).tolist()
    for i, v in enumerate(views):
        pid = v.get("pose_id", i)
        d = root / f"{pid:03d}"
        d.mkdir()
        T_base_cam = v.get("T_base_cam")
        T_base_cam = ident if T_base_cam is None \
            else np.asarray(T_base_cam, float).tolist()
        (d / "meta.json").write_text(json.dumps(
            {"pose_id": pid, "timestamp": v.get("t", float(i)),
             "joints_rad": [0.0] * 6, "T_base_flange": ident,
             "T_base_cam": T_base_cam}))
        if v.get("rgb") is not None:
            cv2.imwrite(str(d / "rgb.png"),
                       cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2BGR))
    return root
