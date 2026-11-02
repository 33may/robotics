#!/usr/bin/env python3
"""Change a servo's ID on a given port.

Usage: python change_id.py <port> <current_id> <new_id>
Example: python change_id.py /dev/ttyACM1 6 1
"""
import sys
from scservo_sdk import PortHandler, PacketHandler

BAUDRATE = 1000000

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <port> <current_id> <new_id>")
        sys.exit(1)

    port = sys.argv[1]
    current_id = int(sys.argv[2])
    new_id = int(sys.argv[3])

    ph = PortHandler(port)
    if not ph.openPort():
        print(f"ERROR: Cannot open {port}")
        sys.exit(1)
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(0)

    # Verify motor responds at current_id
    model, res, _ = pkt.ping(ph, current_id)
    if res != 0:
        print(f"No motor found at ID={current_id} on {port}")
        ph.closePort()
        sys.exit(1)
    print(f"Found motor at ID={current_id} (model={model})")

    # Check no motor already at new_id
    _, res2, _ = pkt.ping(ph, new_id)
    if res2 == 0:
        print(f"WARNING: A motor already responds at ID={new_id}! Aborting to avoid collision.")
        ph.closePort()
        sys.exit(1)

    input(f"Will change ID {current_id} → {new_id} on {port}. Press ENTER to confirm...")

    # Unlock EEPROM (addr 55 = 0)
    pkt.write1ByteTxRx(ph, current_id, 55, 0)
    # Write new ID (addr 5)
    res, err = pkt.write1ByteTxRx(ph, current_id, 5, new_id)
    if res != 0:
        print(f"Write failed: res={res} err={err}")
        ph.closePort()
        sys.exit(1)

    # Verify at new ID
    model, res, _ = pkt.ping(ph, new_id)
    if res == 0:
        print(f"Success! Motor now at ID={new_id} (model={model})")
    else:
        print(f"Motor not responding at ID={new_id} — check manually")

    ph.closePort()


if __name__ == "__main__":
    main()
