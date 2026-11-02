"""Integration TDD for recorder_main — the standalone recording process
(MAY-173 slam-demo-loop 1.4).

Boots the REAL process (brain env, subprocess) against a REAL CameraPublisher +
DebugPoseServer driven by a fake body — no Isaac. Verifies the whole contract:
frames + poses in → bake-layout dump + status.json out; SIGTERM = graceful save
(the dev_app Stop button / teardown path); every stream lands with FK cam rows.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli import CameraIntrinsics
from humanoid.logic.oli.comm.camera_publisher import CameraPublisher
from humanoid.logic.oli.comm.debug_pose import DebugPoseServer

pytestmark = pytest.mark.brain

_REPO = Path(__file__).resolve().parents[3].parent


class _Body:
    """head RGBD + stereo pair, deterministic pixels."""

    @property
    def camera_names(self):
        return ["head"]

    @property
    def stereo_camera_names(self):
        return ["head_left", "head_right"]

    def read_camera_rgbd(self, name):
        rgb = np.full((8, 12, 3), 20, dtype=np.uint8)
        return rgb, np.full((8, 12), 1.5, dtype=np.float32)

    def read_camera_rgb(self, name):
        return np.full((8, 12, 3), 30, dtype=np.uint8)

    def camera_intrinsics(self, name):
        return CameraIntrinsics(width=12, height=8, fx=5.0, fy=5.0, cx=6.0, cy=4.0)


def test_recorder_process_end_to_end(tmp_path):
    cam_sock = str(tmp_path / "frames.sock")
    pose_sock = str(tmp_path / "pose.sock")
    out = tmp_path / "dump"

    proc = subprocess.Popen(
        [sys.executable, "-m", "humanoid.logic.oli.recording.recorder_main",
         "--out", str(out), "--codec", "jpeg",
         "--camera-socket", cam_sock, "--pose-socket", pose_sock,
         "--idle-timeout", "0"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    pub = None
    try:
        # recorder BINDS the pose socket; wait for it, then start the world side
        deadline = time.monotonic() + 10
        while not os.path.exists(pose_sock) and time.monotonic() < deadline:
            time.sleep(0.05)
        pub = CameraPublisher(_Body(), socket_path=cam_sock)
        poses = DebugPoseServer(pose_sock)

        dt = 33_366_666
        for i in range(12):
            stamp = 1_000_000_000 + i * dt
            poses.publish(stamp, 1.0 + 0.01 * i, 2.0, 0.5)
            time.sleep(0.02)                      # pose lands before its frame
            pub.publish(i, stamp_ns=stamp)
            time.sleep(0.05)

        # status heartbeats while recording
        status = json.loads((out / "status.json").read_text())
        assert status["state"] in ("recording", "waiting-frames")

        proc.send_signal(signal.SIGTERM)          # the Stop button / teardown path
        assert proc.wait(timeout=15) == 0
        poses.close()
    finally:
        if pub is not None:
            pub.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # ── the dump is bake-layout + complete ────────────────────────────────────
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "saved"
    assert status["frames"] >= 3 * 10             # 3 streams × most stamps
    rig = json.loads((out / "rig.json").read_text())
    assert rig["cameras"]["head_left"]["intrinsics"]["width"] == 12
    for stream, ext in (("head_left", ".jpg"), ("head_right", ".jpg"), ("head", ".jpg")):
        files = list((out / "frames" / stream).iterdir())
        assert files and all(f.suffix == ext for f in files), stream
    assert (out / "frames" / "head_depth").exists()  # RGBD depth rides along
    rows = [json.loads(l) for l in (out / "poses.jsonl").read_text().splitlines()]
    cams = {r["cam"] for r in rows}
    assert {"base", "head", "head_left", "head_right"} <= cams
    base_rows = [r for r in rows if r["cam"] == "base"]
    assert abs(base_rows[0]["x"] - 1.0) < 0.2     # GT pose joined, not fabricated
