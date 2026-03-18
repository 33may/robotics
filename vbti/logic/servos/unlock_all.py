#!/usr/bin/env python3
"""Unlock EEPROM on all servos so lerobot can write calibration/config.

Usage:
    python vbti/logic/servos/unlock_all.py                    # all ports
    python vbti/logic/servos/unlock_all.py --port=/dev/ttyACM0  # specific port
"""
import glob
import argparse
from scservo_sdk import *

BAUDRATE = 1000000
MOTOR_IDS = range(1, 7)
LOCK_REGISTER = 55


def unlock(port: str):
    ph = PortHandler(port)
    if not ph.openPort():
        print(f"  {port}: FAILED TO OPEN")
        return
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(0)

    print(f"  {port}:")
    for mid in MOTOR_IDS:
        lock, res, _ = pkt.read1ByteTxRx(ph, mid, LOCK_REGISTER)
        if res != 0:
            print(f"    ID {mid}: no response")
            continue
        if lock == 0:
            print(f"    ID {mid}: already unlocked")
            continue
        comm, err = pkt.write1ByteTxRx(ph, mid, LOCK_REGISTER, 0)
        if comm != 0:
            print(f"    ID {mid}: FAILED to unlock — {pkt.getTxRxResult(comm)}")
        else:
            print(f"    ID {mid}: unlocked (was locked)")

    ph.closePort()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Specific port, or scan all /dev/ttyACM*")
    args = parser.parse_args()

    if args.port:
        ports = [args.port]
    else:
        ports = sorted(glob.glob("/dev/ttyACM*"))

    print(f"Unlocking servos on {ports}\n")
    for port in ports:
        unlock(port)
    print("\nDone.")


if __name__ == "__main__":
    main()
