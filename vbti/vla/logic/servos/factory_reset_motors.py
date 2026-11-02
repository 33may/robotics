#!/usr/bin/env python3
"""
Factory reset specific motors on an SO-ARM101 arm.

Safe flow per motor:
  1. Move ID=1 (shoulder_pan) to temp ID=7
  2. Send RESET (0x06) to target motor → it becomes ID=1
  3. Change ID=1 → target ID
  4. Move ID=7 back to ID=1

Usage: python factory_reset_motors.py
"""
import sys
import time
from scservo_sdk import PortHandler, PacketHandler

# ── Configuration ──
PORT = "/dev/ttyACM1"
BAUDRATE = 1000000
MOTORS_TO_RESET = [3]
TEMP_ID = 7


def send_reset(ser, motor_id):
    """Send raw RESET instruction (0x06) to a Feetech servo."""
    length = 0x02
    instruction = 0x06
    checksum = (~(motor_id + length + instruction)) & 0xFF
    packet = bytes([0xFF, 0xFF, motor_id, length, instruction, checksum])
    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.1)
    response = ser.read(6)
    return response


def change_id(pkt, ph, from_id, to_id):
    """Change servo ID. Returns True on success."""
    # Unlock EEPROM
    pkt.write1ByteTxRx(ph, from_id, 55, 0)
    # Write new ID (response will timeout since servo changes ID mid-packet)
    pkt.write1ByteTxRx(ph, from_id, 5, to_id)
    time.sleep(0.1)
    # Verify at new ID
    _, res, _ = pkt.ping(ph, to_id)
    return res == 0


def main():
    ph = PortHandler(PORT)
    if not ph.openPort():
        print(f"ERROR: Cannot open {PORT}")
        return
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(0)

    # Step 0: Confirm state
    print("=" * 50)
    print("STEP 0: Checking bus state")
    print("=" * 50)

    _, res1, _ = pkt.ping(ph, 1)
    if res1 != 0:
        print(f"  ID=1 (shoulder_pan): NO RESPONSE — cannot proceed safely")
        ph.closePort()
        return
    print(f"  ID=1 (shoulder_pan): OK")

    for mid in MOTORS_TO_RESET:
        hw_err, res, _ = pkt.read1ByteTxRx(ph, mid, 65)
        if res != 0:
            print(f"  Motor {mid}: NO RESPONSE — cannot reset")
            ph.closePort()
            return
        print(f"  Motor {mid}: hw_err={hw_err} — {'needs reset' if hw_err else 'already OK'}")

    _, res_temp, _ = pkt.ping(ph, TEMP_ID)
    if res_temp == 0:
        print(f"  ID={TEMP_ID} already occupied! Change TEMP_ID and retry.")
        ph.closePort()
        return
    print(f"  ID={TEMP_ID} (temp): free")

    input("\nPress ENTER to start, or Ctrl+C to abort...")

    raw_ser = ph.ser

    for mid in MOTORS_TO_RESET:
        print(f"\n{'=' * 50}")
        print(f"Resetting motor {mid}")
        print(f"{'=' * 50}")

        # 1. Move shoulder_pan (ID=1) out of the way
        print(f"\n  [1/4] Moving ID=1 → ID={TEMP_ID} (temp)...")
        if not change_id(pkt, ph, 1, TEMP_ID):
            print(f"  FAILED to move ID=1 → {TEMP_ID}. Aborting.")
            ph.closePort()
            return
        print(f"  Done. shoulder_pan is now at ID={TEMP_ID}")

        # 2. Factory reset target motor → becomes ID=1
        print(f"\n  [2/4] Sending RESET to motor {mid}...")
        response = send_reset(raw_ser, mid)
        if len(response) >= 4:
            print(f"  Response: {response.hex()}")
        else:
            print(f"  No/partial response: {response.hex() if response else 'empty'}")
            print("  (May be OK — some firmware doesn't respond to reset)")

        time.sleep(0.5)

        # Verify it's now at ID=1
        model, res, _ = pkt.ping(ph, 1)
        if res == 0:
            print(f"  Motor appeared at ID=1 (model={model})")
        else:
            # Maybe reset worked but ID didn't change
            _, res2, _ = pkt.ping(ph, mid)
            if res2 == 0:
                hw, _, _ = pkt.read1ByteTxRx(ph, mid, 65)
                if hw == 0:
                    print(f"  hw_err cleared but ID stayed at {mid}. Moving shoulder_pan back...")
                    change_id(pkt, ph, TEMP_ID, 1)
                    print(f"  shoulder_pan back at ID=1. Motor {mid} is fixed.")
                    continue
            print(f"  Cannot find motor after reset. Aborting.")
            print(f"  NOTE: shoulder_pan is at ID={TEMP_ID}! Run: python change_id.py {PORT} {TEMP_ID} 1")
            ph.closePort()
            return

        # 3. Change reset motor from ID=1 → target ID
        print(f"\n  [3/4] Changing ID=1 → ID={mid}...")
        if not change_id(pkt, ph, 1, mid):
            print(f"  FAILED. Aborting.")
            print(f"  NOTE: shoulder_pan is at ID={TEMP_ID}! Run: python change_id.py {PORT} {TEMP_ID} 1")
            ph.closePort()
            return

        hw_err, _, _ = pkt.read1ByteTxRx(ph, mid, 65)
        print(f"  Motor {mid}: hw_err={hw_err}")

        # 4. Move shoulder_pan back: ID=7 → ID=1
        print(f"\n  [4/4] Moving ID={TEMP_ID} → ID=1 (restoring shoulder_pan)...")
        if not change_id(pkt, ph, TEMP_ID, 1):
            print(f"  FAILED to restore shoulder_pan!")
            print(f"  Run manually: python change_id.py {PORT} {TEMP_ID} 1")
            ph.closePort()
            return
        print(f"  shoulder_pan back at ID=1")

    # Final scan
    print(f"\n{'=' * 50}")
    print("FINAL: Verifying all motors")
    print(f"{'=' * 50}")
    motor_names = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
                   4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
    for mid in range(1, 7):
        hw_err, res, _ = pkt.read1ByteTxRx(ph, mid, 65)
        pos, _, _ = pkt.read2ByteTxRx(ph, mid, 56)
        status = f"hw_err={hw_err}" if res == 0 else "NO RESPONSE"
        print(f"  Motor {mid} ({motor_names[mid]}): pos={pos} {status}")

    ph.closePort()
    print("\nDone!")


if __name__ == "__main__":
    main()
