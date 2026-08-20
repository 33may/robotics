#!/usr/bin/env python3
"""rigs: FakeRig interpolation+stop, CameraWorker freshness, PoseStreamer gating.
Run: p inspection/tests/test_rigs.py"""
import threading
import time

import numpy as np

from inspection.run.rigs import CameraWorker, FakeRig, PoseStreamer


def test_fake_move_interpolates_and_stops():
    stop = threading.Event()
    rig = FakeRig(np.zeros(6), stop_event=stop, dt=0.005, speed=2.0)
    goal = np.zeros(6); goal[0] = 0.4
    rep = rig.move([np.zeros(6), goal])
    assert rep["stopped"] is False and np.allclose(rig.q(), goal)

    far = np.zeros(6); far[0] = 50.0                       # long move
    t = threading.Thread(target=lambda: (time.sleep(0.05), stop.set()))
    t.start()
    rep = rig.move([goal, far])
    t.join()
    assert rep["stopped"] is True
    assert rig.q()[0] < 49.0                               # froze mid-path


def test_camera_worker_freshness_and_publish():
    calls = []
    n = [0]
    def grab():
        n[0] += 1
        return {"rgb": np.full((4, 4, 3), n[0] % 255, np.uint8)}
    cam = CameraWorker(grab, publish=lambda rgb: calls.append(1), hz=200.0)
    cam.start()
    c0, _ = cam.latest()
    b = cam.fresh_bundle(min_new=3, timeout=2.0)
    c1, _ = cam.latest()
    assert c1 >= c0 + 3 and b["rgb"] is not None
    cam.stop()
    assert calls, "publish was never called"


def test_pose_streamer_gated_by_active():
    got = []
    ps = PoseStreamer(lambda: np.zeros(6), lambda q: got.append(1), hz=200.0)
    ps.start()
    time.sleep(0.05)
    assert not got, "published while inactive"
    ps.active.set()
    time.sleep(0.05)
    ps.active.clear()
    assert got, "did not publish while active"
    ps.stop()


def main():
    test_fake_move_interpolates_and_stops()
    test_camera_worker_freshness_and_publish()
    test_pose_streamer_gated_by_active()
    print("OK test_rigs")


if __name__ == "__main__":
    main()
