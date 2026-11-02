"""Tests for the coverage-drive DriveRecorder (MAY-173 locdev T2).

The motion-agnostic sink of the data-collection flow: whatever moves the base or
the cameras (waypoint follower today, walk-motion mimic later), the recorder just
persists what was RENDERED — RGB frames + the camera's actual world pose — plus
the rig description the map bake needs. Layout:

    <out>/rig.json                     rig dict, verbatim
    <out>/frames/<cam>/<stamp>.png     zero-padded stamp → lexical sort = time sort
    <out>/poses.jsonl                  one row per frame (+ cam="base" rows, no file)

Runs in the `brain` env (numpy + PIL + stdlib; no isaacsim).
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from humanoid.logic.simulation.mapping.recorder import DriveRecorder

pytestmark = pytest.mark.brain


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)


def _pose(tx: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = tx
    return T


def test_writes_rig_json_verbatim(tmp_path):
    rig = {"cameras": {"head_left": {"fx": 465.6}}, "baseline_m": 0.05}
    rec = DriveRecorder(tmp_path / "drive", rig)
    rec.close()
    assert json.loads((tmp_path / "drive" / "rig.json").read_text()) == rig


def test_frame_png_round_trips(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rgb = _rgb(0)
    rec.add_frame("head_left", 123, rgb, _pose(1.0))
    rec.close()
    path = tmp_path / "drive" / "frames" / "head_left" / f"{123:019d}.png"
    assert path.exists()
    np.testing.assert_array_equal(np.asarray(Image.open(path)), rgb)


def test_pose_rows_carry_stamp_cam_file_and_transform(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_frame("head_left", 123, _rgb(0), _pose(2.5))
    rec.close()
    rows = [json.loads(line) for line in
            (tmp_path / "drive" / "poses.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["cam"] == "head_left"
    assert row["stamp_ns"] == 123
    assert row["file"] == f"frames/head_left/{123:019d}.png"
    np.testing.assert_allclose(np.array(row["T_world_cam"]), _pose(2.5))


def test_base_pose_rows_have_no_file(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_base_pose(456, x=1.0, y=-2.0, yaw=0.5)
    rec.close()
    row = json.loads((tmp_path / "drive" / "poses.jsonl").read_text())
    assert row["cam"] == "base"
    assert row["file"] is None
    assert row["stamp_ns"] == 456
    assert (row["x"], row["y"], row["yaw"]) == (1.0, -2.0, 0.5)


def test_rows_survive_close_and_frames_count(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    for i in range(3):
        rec.add_frame("head_right", 100 + i, _rgb(i), _pose(float(i)))
    rec.close()
    rows = (tmp_path / "drive" / "poses.jsonl").read_text().splitlines()
    assert len(rows) == 3
    assert rec.frames_written == 3


def test_stamp_names_sort_lexically_as_time(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_frame("head_left", 999, _rgb(0), _pose(0.0))
    rec.add_frame("head_left", 10_000_000_000, _rgb(1), _pose(1.0))
    rec.close()
    names = sorted(p.name for p in (tmp_path / "drive" / "frames" / "head_left").iterdir())
    stamps = [int(n.split(".")[0]) for n in names]
    assert stamps == sorted(stamps) == [999, 10_000_000_000]


def test_depth_saved_as_uint16_mm_png(tmp_path):
    # Depth rides along optionally (head RGBD cam): float32 meters in, uint16
    # millimeter PNG on disk — cuVSLAM's own unit convention (depth_scale_factor=1000).
    rec = DriveRecorder(tmp_path / "drive", {})
    depth_m = np.array([[0.5, 1.0], [2.0, 65.535]], dtype=np.float32)
    rec.add_frame("head", 123, _rgb(0)[:2, :2], _pose(0.0), depth_m=depth_m)
    rec.close()
    path = tmp_path / "drive" / "frames" / "head_depth" / f"{123:019d}.png"
    stored = np.asarray(Image.open(path))
    assert stored.dtype == np.uint16
    np.testing.assert_array_equal(stored, [[500, 1000], [2000, 65535]])
    row = json.loads((tmp_path / "drive" / "poses.jsonl").read_text())
    assert row["depth_file"] == f"frames/head_depth/{123:019d}.png"


def test_no_depth_means_null_depth_file(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_frame("head_left", 5, _rgb(0), _pose(0.0))
    rec.close()
    row = json.loads((tmp_path / "drive" / "poses.jsonl").read_text())
    assert row["depth_file"] is None


def test_add_imu_writes_imu_jsonl(tmp_path):
    """IMU samples land in imu.jsonl (stamp_ns, acc[3], gyro[3]) — the raw
    stream rosbag_synth turns into sensor_msgs/Imu for cuVSLAM inertial mode."""
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_imu(100, acc=(0.1, 0.2, 9.81), gyro=(0.0, 0.01, 0.02))
    rec.add_imu(105, acc=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
    rec.close()
    rows = [json.loads(line) for line in
            (tmp_path / "drive" / "imu.jsonl").read_text().splitlines()]
    assert rows[0] == {"stamp_ns": 100, "acc": [0.1, 0.2, 9.81],
                       "gyro": [0.0, 0.01, 0.02]}
    assert rows[1]["stamp_ns"] == 105


def test_no_imu_calls_no_imu_file(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_frame("head_left", 5, _rgb(0), _pose(0.0))
    rec.close()
    assert not (tmp_path / "drive" / "imu.jsonl").exists()


def test_module_is_pure():
    import humanoid.logic.simulation.mapping.recorder  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules


# ── async writer + codec slot (MAY-173 slam-demo-loop 1.2) ────────────────────────
# Live teleop capture: PNG encode (~200-400 ms each) on the caller's thread stalls
# the sim loop. Async mode hands frames to a bounded queue drained by encoder
# workers — block-never-drop backpressure; close() drains everything. The codec
# slot (png|jpeg) awaits the JPEG-vs-cuVSLAM verdict; depth ALWAYS stays uint16 PNG.


def test_jpeg_codec_writes_jpg_files_and_rows(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {}, codec="jpeg")
    # Smooth gradient, NOT random noise — noise is JPEG's DCT worst case and would
    # assert nothing about the codec config. q95 on natural/smooth content is ~1-2
    # mean error; a quality-knob misconfiguration (e.g. q=5) would blow past this.
    yy, xx = np.mgrid[0:8, 0:12]
    rgb = np.stack([yy * 20, xx * 15, (yy + xx) * 8], axis=-1).astype(np.uint8)
    rec.add_frame("head_left", 123, rgb, _pose(1.0))
    rec.close()
    path = tmp_path / "drive" / "frames" / "head_left" / f"{123:019d}.jpg"
    assert path.exists()
    stored = np.asarray(Image.open(path)).astype(np.int16)
    assert np.abs(stored - rgb.astype(np.int16)).mean() < 3.0
    row = json.loads((tmp_path / "drive" / "poses.jsonl").read_text())
    assert row["file"] == f"frames/head_left/{123:019d}.jpg"


def test_jpeg_codec_keeps_depth_lossless_png(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {}, codec="jpeg")
    depth_m = np.array([[0.5, 1.0], [2.0, 65.535]], dtype=np.float32)
    rec.add_frame("head", 9, _rgb(0)[:2, :2], _pose(0.0), depth_m=depth_m)
    rec.close()
    stored = np.asarray(Image.open(
        tmp_path / "drive" / "frames" / "head_depth" / f"{9:019d}.png"))
    assert stored.dtype == np.uint16
    np.testing.assert_array_equal(stored, [[500, 1000], [2000, 65535]])


def test_unknown_codec_rejected(tmp_path):
    with pytest.raises(ValueError):
        DriveRecorder(tmp_path / "drive", {}, codec="webp")


def test_async_all_frames_on_disk_after_close(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {}, async_write=True, workers=3)
    for i in range(40):
        rec.add_frame("head_left", 1000 + i, _rgb(i), _pose(float(i)))
    rec.close()
    files = list((tmp_path / "drive" / "frames" / "head_left").iterdir())
    assert len(files) == 40
    rows = (tmp_path / "drive" / "poses.jsonl").read_text().splitlines()
    assert len(rows) == 40
    assert rec.frames_written == 40


def test_async_tiny_queue_never_drops(tmp_path):
    """Backpressure invariant: a full queue BLOCKS the producer, never drops a
    frame — with queue_frames=1 every submitted frame must still land."""
    rec = DriveRecorder(tmp_path / "drive", {}, async_write=True,
                        workers=1, queue_frames=1)
    for i in range(20):
        rec.add_frame("head_left", i, _rgb(i), _pose(float(i)))
    rec.close()
    assert len(list((tmp_path / "drive" / "frames" / "head_left").iterdir())) == 20


def test_async_frame_content_correct(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {}, async_write=True)
    rgb = _rgb(7)
    rec.add_frame("head_right", 55, rgb, _pose(2.0))
    rec.close()
    stored = np.asarray(Image.open(
        tmp_path / "drive" / "frames" / "head_right" / f"{55:019d}.png"))
    np.testing.assert_array_equal(stored, rgb)
    row = json.loads((tmp_path / "drive" / "poses.jsonl").read_text())
    np.testing.assert_allclose(np.array(row["T_world_cam"]), _pose(2.0))


def test_async_base_pose_and_imu_flow(tmp_path):
    rec = DriveRecorder(tmp_path / "drive", {}, async_write=True)
    rec.add_frame("head_left", 1, _rgb(0), _pose(0.0))
    rec.add_base_pose(1, x=0.5, y=1.5, yaw=0.1)
    rec.add_imu(2, acc=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0))
    rec.close()
    rows = [json.loads(l) for l in
            (tmp_path / "drive" / "poses.jsonl").read_text().splitlines()]
    assert {r["cam"] for r in rows} == {"head_left", "base"}
    assert (tmp_path / "drive" / "imu.jsonl").exists()


def test_rows_flushed_before_close(tmp_path):
    """Crash-safety: pose rows must be readable from disk WITHOUT close() —
    a dev_app/recorder crash must not lose the jsonl tail."""
    rec = DriveRecorder(tmp_path / "drive", {})
    rec.add_base_pose(456, x=1.0, y=-2.0, yaw=0.5)
    on_disk = (tmp_path / "drive" / "poses.jsonl").read_text()
    rec.close()
    assert json.loads(on_disk)["stamp_ns"] == 456
