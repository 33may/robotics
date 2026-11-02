"""TDD for recording/session.py — the recorder process's testable heart
(MAY-173 slam-demo-loop 1.4).

`RecordingSession` turns live-channel reads into a bake-grade coverage dump:
  - frames deduped by (stream, stamp) — the mailbox re-delivers the latest frame
    on every poll, so repeated reads of the same stamp must persist ONCE
  - one base row PER FRAME STAMP (the synth groups samples by stamp, so base and
    cam rows must share stamps), pose = nearest GT sample from the debug-pose
    buffer; cam rows carry the FK `T_world_cam`
  - "head" RGBD frames persist depth; stereo frames are RGB-only
  - stamp-gap accounting: the lossy channel can drop frames; the session REPORTS
    gaps (it cannot recover them) so the panel/acceptance can judge the capture

Brain-pure: contracts + fk + DriveRecorder on tmp dirs.
"""

import json
import sys

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics
from humanoid.logic.oli.camera_mounts import STEREO_CAMERAS
from humanoid.logic.oli.recording.fk import cam_world
from humanoid.logic.oli.recording.session import RecordingSession

pytestmark = pytest.mark.brain

_INTR = CameraIntrinsics(width=12, height=8, fx=5.0, fy=5.0, cx=6.0, cy=4.0)


def _frame(name: str, stamp: int, rgbd: bool = False) -> CameraFrame:
    rgb = np.full((8, 12, 3), stamp % 251, dtype=np.uint8)
    depth = np.full((8, 12), 1.5, dtype=np.float32) if rgbd else None
    return CameraFrame(stamp_ns=stamp, name=name, rgb=rgb, depth=depth, intrinsics=_INTR)


def _rows(root):
    return [json.loads(l) for l in (root / "poses.jsonl").read_text().splitlines()]


def _session(tmp_path, **kw):
    kw.setdefault("codec", "png")
    kw.setdefault("streams", ["head_left", "head_right", "head"])
    return RecordingSession(tmp_path / "dump", camera_res=(12, 8), **kw)


def test_frame_persisted_with_fk_pose_and_base_row(tmp_path):
    s = _session(tmp_path)
    s.feed_base(90, x=1.0, y=2.0, yaw=0.5)
    s.feed_base(100, x=1.1, y=2.0, yaw=0.5)
    s.feed_frame(_frame("head_left", 101))
    s.stop()
    rows = _rows(tmp_path / "dump")
    base = [r for r in rows if r["cam"] == "base"]
    cams = [r for r in rows if r["cam"] == "head_left"]
    assert len(base) == 1 and len(cams) == 1
    assert base[0]["stamp_ns"] == 101          # base row carries the FRAME stamp
    assert base[0]["x"] == 1.1                 # nearest GT sample (100 not 90)
    left = next(m for m in STEREO_CAMERAS if m.name == "head_left")
    np.testing.assert_allclose(
        np.array(cams[0]["T_world_cam"]), cam_world(1.1, 2.0, 0.5, left), atol=1e-9)


def test_duplicate_mailbox_reads_persist_once(tmp_path):
    s = _session(tmp_path)
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    f = _frame("head_left", 100)
    for _ in range(5):                          # poll loop re-reads the same frame
        s.feed_frame(f)
    s.stop()
    files = list((tmp_path / "dump" / "frames" / "head_left").iterdir())
    assert len(files) == 1
    assert s.stats()["frames"] == 1


def test_one_base_row_per_stamp_across_streams(tmp_path):
    s = _session(tmp_path)
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    s.feed_frame(_frame("head_left", 100))
    s.feed_frame(_frame("head_right", 100))    # same render tick, same stamp
    s.stop()
    base = [r for r in _rows(tmp_path / "dump") if r["cam"] == "base"]
    assert len(base) == 1


def test_head_depth_persists_stereo_stays_rgb_only(tmp_path):
    s = _session(tmp_path)
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    s.feed_frame(_frame("head", 100, rgbd=True))
    s.feed_frame(_frame("head_left", 100))
    s.stop()
    root = tmp_path / "dump"
    assert (root / "frames" / "head_depth").exists()
    assert not (root / "frames" / "head_left_depth").exists()


def test_no_pose_yet_frame_is_counted_not_persisted(tmp_path):
    """Frames arriving before ANY GT pose can't produce truthful rows — they are
    skipped and counted, never written with a made-up pose."""
    s = _session(tmp_path)
    s.feed_frame(_frame("head_left", 50))
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    s.feed_frame(_frame("head_left", 101))
    s.stop()
    assert s.stats()["frames"] == 1
    assert s.stats()["skipped_no_pose"] == 1


def test_gap_accounting(tmp_path):
    s = _session(tmp_path)
    s.feed_base(0, x=0.0, y=0.0, yaw=0.0)
    dt = 33_366_666                             # ~30 Hz
    stamps = [0, dt, 2 * dt, 5 * dt, 6 * dt]    # 2 frames missing at 3dt, 4dt
    for t in stamps:
        s.feed_base(t, x=0.0, y=0.0, yaw=0.0)
        s.feed_frame(_frame("head_left", t))
    s.stop()
    st = s.stats()
    assert st["gaps"] == 1                      # one gap event...
    assert st["missed_frames"] == 2             # ...worth two frames


def test_jpeg_codec_flows_through(tmp_path):
    s = _session(tmp_path, codec="jpeg")
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    s.feed_frame(_frame("head_left", 100))
    s.stop()
    files = list((tmp_path / "dump" / "frames" / "head_left").iterdir())
    assert files[0].suffix == ".jpg"


def test_rig_json_written(tmp_path):
    s = _session(tmp_path)
    s.stop()
    rig = json.loads((tmp_path / "dump" / "rig.json").read_text())
    assert rig["stereo_pair"] == ["head_left", "head_right"]
    assert rig["cameras"]["head_left"]["intrinsics"]["width"] == 12


def test_stats_shape(tmp_path):
    s = _session(tmp_path)
    s.feed_base(100, x=0.0, y=0.0, yaw=0.0)
    s.feed_frame(_frame("head_left", 100))
    st = s.stats()
    assert {"frames", "stamps", "gaps", "missed_frames", "skipped_no_pose",
            "last_stamp_ns", "per_stream"} <= set(st)
    assert st["per_stream"]["head_left"] == 1
    s.stop()


def test_module_is_brain_pure():
    import humanoid.logic.oli.recording.session  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
