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


def test_module_is_pure():
    import humanoid.logic.simulation.mapping.recorder  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
