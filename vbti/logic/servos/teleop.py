#!/usr/bin/env python3
"""Leader->follower teleop for two SO-ARM101s over raw scservo_sdk.

The leader (torque off, hand-moved) is mirrored tick-to-tick onto the
follower (torque on). Both arms carry EEPROM calibrations in the same
centered convention — verified 2026-08-14 — so no frame mapping is needed.
Follower goals are clamped to its own EEPROM limits and rate-limited.

Usage:
    python -m vbti.logic.servos.teleop run            # mirror until Ctrl+C
"""

import threading
import time

import numpy as np

from vbti.logic.servos.so101 import (
    SO101, JOINT_NAMES, JOINT_IDS, ADDR_GOAL_POSITION,
)

FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"

RATE_HZ = 50
MAX_STEP_TICKS = 20  # per cycle ≈ 88 deg/s cap — smooths leader jitter/jumps


class TeleopMirror:
    """Background thread mirroring leader ticks onto the follower.

    pause()/resume() freeze the follower goal (it holds position, torque on)
    so captures happen on a motionless arm even if the leader wiggles.
    """

    def __init__(self, follower: SO101, leader: SO101):
        self.follower = follower
        self.leader = leader
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.leader.set_torque(False)   # leader must move freely by hand
        self.follower.set_torque(True)

        # Ramp follower to the leader's current pose so it doesn't jump.
        leader_deg = self.leader.read_degrees()
        target = {n: leader_deg[n] for n in JOINT_NAMES}
        self.follower.move_to(target, speed_deg_s=30.0)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        goal = np.array([self.follower.read_ticks()[n] for n in JOINT_NAMES],
                        dtype=float)
        dt = 1.0 / RATE_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self._pause.is_set():
                try:
                    lead = np.array(
                        [self.leader.read_ticks()[n] for n in JOINT_NAMES],
                        dtype=float)
                except RuntimeError:
                    time.sleep(dt)
                    continue  # transient comms miss — keep last goal
                step = np.clip(lead - goal, -MAX_STEP_TICKS, MAX_STEP_TICKS)
                goal = goal + step
                for j, name in enumerate(JOINT_NAMES):
                    ticks = self.follower.clamp(name, int(round(goal[j])))
                    self.follower._write2(JOINT_IDS[name], ADDR_GOAL_POSITION,
                                          ticks)
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run(follower_port: str = FOLLOWER_PORT, leader_port: str = LEADER_PORT) -> None:
    """Mirror leader onto follower until Ctrl+C."""
    follower, leader = SO101(follower_port), SO101(leader_port)
    mirror = TeleopMirror(follower, leader)
    mirror.start()
    print("Teleop live — move the leader. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mirror.stop()
        follower.close()
        leader.close()
        print("\nTeleop stopped.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run})
