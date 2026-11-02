"""show — the reduced two-pane viewer over a recorded run (stories C1/D4).

`p -m inspection.record show <id>` lands here when a run has no derived
step clouds (the full six-story workspace, record/rr/workspace.py, needs
them). Same two panes as the full workspace, from run files alone:

  LEFT   3D — every camera frustum, the object cloud at step k (per-step
         `steps/NNN/fused/cloud.npy` when the run recorded them; the
         run-level final fused cloud standing in otherwise), and the
         step-k camera lit in the accent colour.
  RIGHT  2D — the rgb captured at step k, re-logged to the one stable
         `ACTIVE` entity so the pane tracks the slider.

All Rerun calls and the blueprint come from record/rr/log.py — one
schema, both viewers, so scrubbing muscle memory transfers. Works on both
run generations via `Run` (record/run.py). Read-only on the run; the only
side product is rotated-back rgb copies for 180° views (scratch, /tmp —
see `_pose_frame_rgb`).
"""
from __future__ import annotations

import json
from pathlib import Path

from inspection.record.run import Run

#: Same scratch home as workspace.py's IMG_CACHE: rotated rgb frames are an
#: artefact of how capture stored the png, reproducible in a second.
IMG_CACHE = Path("/tmp/rrd/_img")


def show_run(run_dir: Path, out: Path) -> Path:
    from inspection.record.rr import log as rr_log
    from inspection.record.rr import workspace as rr_ws

    run_dir, out = Path(run_dir), Path(out)
    run = Run.load(run_dir)
    app_id = f"record-{run.record.id}"  # rerun 0.37: no '/' in app ids
    rr_log.begin(app_id)
    rr_log.log_world()

    intr = _color_intrinsics(run_dir)
    steps = [s for s in run.steps if s.T_base_cam is not None]
    # The step-k cloud, by source precedence (Anton, 2026-09-10). The scrub
    # must show the cloud GROWING view by view whenever per-step clouds
    # exist ANYWHERE; the static final cloud is strictly the last resort.
    #   1. steps/NNN/fused/ in the run itself  (StepRecord v2 records it)
    #   2. derived/<run>/steps/step_NNN.npy/.ply (backfilled v1 runs;
    #      NNN is the step/pose id, which can be sparse)
    #   3. nothing per-step at all -> run-level final pair, static
    derived = run_dir.parent.parent / "derived" / run.record.id / "steps"
    if any((s.dir / "fused" / "cloud.npy").exists() for s in steps):
        step_cloud = rr_ws._recorded_pair
    elif any((derived / f"step_{s.id:03d}.npy").exists()
             or (derived / f"step_{s.id:03d}.ply").exists() for s in steps):
        step_cloud = lambda s: _derived_cloud(derived, s.id)  # noqa: E731
    else:
        step_cloud = None

    prev_id = None
    for s in steps:
        rr_log.at(s.id)
        # Step k's frustum arrives lit; last step's is demoted in the same
        # breath — exactly one orange frustum at any slider position.
        rr_log.log_camera(s.id, s.T_base_cam, intr,
                          image_path=rr_ws._frustum_image(
                              s, IMG_CACHE / run.record.id),
                          color=rr_log.C_CAM_ACTIVE)
        if prev_id is not None:
            rr_log.recolor_camera(prev_id, intr)
        prev_id = s.id
        vs = s.view_state
        if vs is not None:
            faces = [(c.address[1], c.pose, c.status,
                      c.scores.get("off_angle_deg")) for c in vs.candidates
                     if (isinstance(c.address, list) and len(c.address) == 2
                         and c.address[0] == "face" and c.pose is not None)]
            if faces:
                rr_log.log_face_views(faces)
            else:
                # I2: zero face candidates on a step that DOES carry a
                # ViewState — settling, or no boxguard yet — must not leave
                # stale arrows from an earlier step's box pose on screen.
                # Legacy runs with no ViewState at all never reach this
                # branch, so they stay the strict no-op they always were.
                rr_log.clear_face_views()
        if step_cloud is not None:
            fused = step_cloud(s)
            if fused is not None:
                rr_log.log_cloud(*fused)
        ply = s.dir / "cloud.ply"
        if ply.exists():
            rr_log.log_step_ply(s.id, ply)
        rgb = s.dir / "rgb.png"
        if rgb.exists():
            rr_log.log_active_view(intr, rgb)  # upright, as stored

    if step_cloud is None:
        fused = run.fused()
        if fused is not None:
            # STATIC: no per-step clouds anywhere, so the final cloud stands
            # in at every slider position — never only on the last frame.
            rr_log.log_cloud(*fused, static=True)

    bp = rr_log.blueprint(run.record.id,
                          hidden=(rr_log.OVERLAY, rr_log.CLOUD_RGB,
                                  rr_log.STEP_PLY))
    return rr_log.save(out, bp, app_id)


def _derived_cloud(dsteps: Path, step_id: int) -> tuple | None:
    """Backfilled step-k cloud from the derived tree; None when this step
    has neither file (sparse ids). Loading defers to the full workspace's
    `_step_cloud` — npy points, colours from the ply when row-aligned —
    so both viewers read the tree identically."""
    if not (dsteps / f"step_{step_id:03d}.npy").exists() and \
       not (dsteps / f"step_{step_id:03d}.ply").exists():
        return None
    from inspection.record.rr.workspace import _step_cloud
    return _step_cloud(dsteps, step_id)


def _color_intrinsics(run_dir: Path) -> dict | None:
    """The session's color intrinsics block, in the shape `rr/log.py`
    expects (fx, fy, ppx, ppy, width, height); None when absent (mock and
    old-mock runs)."""
    sp = run_dir / "session.json"
    if not sp.exists():
        return None
    try:
        raw = json.loads(sp.read_text())
        c = raw["intrinsics"]["color"]
        return {"fx": c["fx"], "fy": c["fy"], "ppx": c["ppx"], "ppy": c["ppy"],
                "width": c["width"], "height": c["height"]}
    except Exception:
        return None
