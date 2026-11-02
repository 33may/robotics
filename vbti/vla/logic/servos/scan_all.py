#!/usr/bin/env python3
"""Scan all connected arms and show status of every servo."""
import glob
from scservo_sdk import *

BAUDRATE = 1000000
MOTOR_IDS = range(1, 7)
MOTOR_NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}

ports = sorted(glob.glob("/dev/ttyACM*"))
print(f"Found ports: {ports}\n")

for port in ports:
    ph = PortHandler(port)
    if not ph.openPort():
        print(f"--- {port} --- FAILED TO OPEN\n")
        continue
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(0)

    print(f"--- {port} ---")
    print(f"{'ID':>3} {'Name':<15} {'Pos':>6} {'Voltage':>8} {'Temp':>5} {'HW_Err':>7} {'Lock':>5} {'Mode':>5}")
    print("-" * 60)

    for mid in MOTOR_IDS:
        pos, res, _ = pkt.read2ByteTxRx(ph, mid, 56)
        if res != 0:
            print(f"{mid:>3} {MOTOR_NAMES[mid]:<15} {'--- NO RESPONSE ---'}")
            continue
        voltage, _, _ = pkt.read1ByteTxRx(ph, mid, 62)
        temp, _, _ = pkt.read1ByteTxRx(ph, mid, 63)
        hw_err, _, _ = pkt.read1ByteTxRx(ph, mid, 65)
        lock, _, _ = pkt.read1ByteTxRx(ph, mid, 55)
        mode, _, _ = pkt.read1ByteTxRx(ph, mid, 33)

        err_str = f"ERR={hw_err}" if hw_err != 0 else "OK"
        print(f"{mid:>3} {MOTOR_NAMES[mid]:<15} {pos:>6} {voltage/10:>7.1f}V {temp:>4}C {err_str:>7} {lock:>5} {mode:>5}")

    ph.closePort()
    print()
