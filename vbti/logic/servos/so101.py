#!/usr/bin/env python3
"""Raw scservo_sdk driver for the SO-ARM101 — no LeRobot dependency.

The Feetech STS3215 servos store homing offset and angle limits in EEPROM,
so every read/write here is already in the calibrated frame. Degrees are
centered on tick 2048: deg = (ticks - 2048) * 360 / 4096.

Usage:
    python -m vbti.logic.servos.so101 status
    python -m vbti.logic.servos.so101 test            # wrist_roll wiggle + return
    python -m vbti.logic.servos.so101 move --shoulder_pan=10 --wrist_roll=-20
    python -m vbti.logic.servos.so101 torque_off
"""

import time
from contextlib import contextmanager

import numpy as np
from scservo_sdk import PortHandler, PacketHandler

BAUDRATE = 1_000_000
DEFAULT_PORT = "/dev/ttyACM0"

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}

# STS3215 register map (addr, bytes)
ADDR_MIN_LIMIT = 9       # EEPROM, 2B
ADDR_MAX_LIMIT = 11      # EEPROM, 2B
ADDR_TORQUE_ENABLE = 40  # 1B
ADDR_ACCELERATION = 41   # 1B
ADDR_GOAL_POSITION = 42  # 2B
ADDR_PRESENT_POSITION = 56  # 2B

CENTER_TICKS = 2048
TICKS_PER_DEG = 4096 / 360.0
LIMIT_MARGIN_TICKS = 12  # stay ~1 deg inside EEPROM limits


def deg_to_ticks(deg: float) -> int:
    return int(round(CENTER_TICKS + deg * TICKS_PER_DEG))


def ticks_to_deg(ticks: int) -> float:
    return (ticks - CENTER_TICKS) / TICKS_PER_DEG


class SO101:
    """Minimal position-mode driver. All motion is software-interpolated."""

    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = BAUDRATE):
        self.port_name = port
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            raise RuntimeError(f"Failed to open {port}")
        self.ph.setBaudRate(baudrate)
        self.pkt = PacketHandler(0)
        self.limits = self._read_limits()

    # -- low level ----------------------------------------------------------

    def _read2(self, motor_id: int, addr: int) -> int:
        val, res, err = self.pkt.read2ByteTxRx(self.ph, motor_id, addr)
        if res != 0:
            raise RuntimeError(f"Motor {motor_id}: no response (addr {addr})")
        return val

    def _write2(self, motor_id: int, addr: int, value: int) -> None:
        res, err = self.pkt.write2ByteTxRx(self.ph, motor_id, addr, value)
        if res != 0:
            raise RuntimeError(f"Motor {motor_id}: write failed (addr {addr})")

    def _write1(self, motor_id: int, addr: int, value: int) -> None:
        res, err = self.pkt.write1ByteTxRx(self.ph, motor_id, addr, value)
        if res != 0:
            raise RuntimeError(f"Motor {motor_id}: write failed (addr {addr})")

    def _read_limits(self) -> dict:
        """EEPROM min/max angle limits per joint, with a safety margin."""
        limits = {}
        for name, mid in JOINT_IDS.items():
            lo = self._read2(mid, ADDR_MIN_LIMIT)
            hi = self._read2(mid, ADDR_MAX_LIMIT)
            if hi > lo:  # wrist_roll is 0..4095 (full turn), keep as-is
                lo, hi = lo + LIMIT_MARGIN_TICKS, hi - LIMIT_MARGIN_TICKS
            limits[name] = (lo, hi)
        return limits

    # -- state --------------------------------------------------------------

    def read_ticks(self) -> dict:
        return {name: self._read2(mid, ADDR_PRESENT_POSITION)
                for name, mid in JOINT_IDS.items()}

    def read_degrees(self) -> dict:
        return {name: ticks_to_deg(t) for name, t in self.read_ticks().items()}

    # -- motion -------------------------------------------------------------

    def set_torque(self, enabled: bool) -> None:
        for mid in JOINT_IDS.values():
            self._write1(mid, ADDR_TORQUE_ENABLE, 1 if enabled else 0)

    def set_acceleration(self, acc: int = 30) -> None:
        for mid in JOINT_IDS.values():
            self._write1(mid, ADDR_ACCELERATION, acc)

    def clamp(self, name: str, ticks: int) -> int:
        lo, hi = self.limits[name]
        return int(np.clip(ticks, lo, hi))

    def move_to(self, target_deg: dict, speed_deg_s: float = 30.0,
                fps: int = 30, settle_s: float = 0.3) -> dict:
        """Interpolate from current pose to target. Blocks until arrived.

        target_deg: partial dict {joint_name: degrees}; unlisted joints hold.
        Returns final pose in degrees.
        """
        current = self.read_ticks()
        target = dict(current)
        for name, deg in target_deg.items():
            target[name] = self.clamp(name, deg_to_ticks(deg))

        cur = np.array([current[n] for n in JOINT_NAMES], dtype=float)
        tgt = np.array([target[n] for n in JOINT_NAMES], dtype=float)
        step_ticks = speed_deg_s * TICKS_PER_DEG / fps

        self.set_torque(True)
        while True:
            diff = tgt - cur
            if np.abs(diff).max() < 4:  # ~0.35 deg
                break
            cur = cur + np.clip(diff, -step_ticks, step_ticks)
            for j, name in enumerate(JOINT_NAMES):
                self._write2(JOINT_IDS[name], ADDR_GOAL_POSITION, int(round(cur[j])))
            time.sleep(1.0 / fps)
        time.sleep(settle_s)
        return self.read_degrees()

    def close(self) -> None:
        self.ph.closePort()


@contextmanager
def connect(port: str = DEFAULT_PORT):
    arm = SO101(port)
    try:
        yield arm
    finally:
        arm.close()


# -- CLI ---------------------------------------------------------------------

def status(port: str = DEFAULT_PORT) -> None:
    """Print joint angles, ticks, and EEPROM limits."""
    with connect(port) as arm:
        ticks = arm.read_ticks()
        print(f"{'Joint':<15} {'Deg':>8} {'Ticks':>6} {'Limits (ticks)':>16}")
        print("-" * 50)
        for name in JOINT_NAMES:
            lo, hi = arm.limits[name]
            print(f"{name:<15} {ticks_to_deg(ticks[name]):>+8.1f} "
                  f"{ticks[name]:>6} {f'[{lo}, {hi}]':>16}")


def test(port: str = DEFAULT_PORT) -> None:
    """Safe bring-up test: record pose, wiggle wrist_roll, return."""
    with connect(port) as arm:
        home = arm.read_degrees()
        print(f"Home: { {k: round(v, 1) for k, v in home.items()} }")
        print("wrist_roll +20 ...")
        arm.move_to({"wrist_roll": home["wrist_roll"] + 20})
        print("wrist_roll back ...")
        final = arm.move_to({"wrist_roll": home["wrist_roll"]})
        err = abs(final["wrist_roll"] - home["wrist_roll"])
        print(f"Returned. wrist_roll error: {err:.2f} deg")


def move(port: str = DEFAULT_PORT, speed: float = 30.0, **joints) -> None:
    """Move named joints to absolute degrees, e.g. --wrist_roll=-20."""
    bad = set(joints) - set(JOINT_NAMES)
    if bad:
        raise SystemExit(f"Unknown joints: {bad}. Valid: {JOINT_NAMES}")
    with connect(port) as arm:
        final = arm.move_to({k: float(v) for k, v in joints.items()},
                            speed_deg_s=speed)
        print(f"At: { {k: round(v, 1) for k, v in final.items()} }")


def torque_off(port: str = DEFAULT_PORT) -> None:
    """Release all joints (arm will fall — hold it)."""
    with connect(port) as arm:
        arm.set_torque(False)
        print("Torque off.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"status": status, "test": test, "move": move,
               "torque_off": torque_off})
