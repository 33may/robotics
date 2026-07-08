"""sensorjoy_bridge.py — vendor SensorJoy (MROS bus) → our JoyPacket (UDP) → brain.

The REAL-path joystick source (design Open Question: "joystick on real = limxsdk
SensorJoy over the bus"). The operator drives the EXACT vendor `robot-joystick` app,
which publishes `SensorJoy` on the bus; the brain (py3.11, no limxsdk) cannot read it.
This tiny py3.8 limxsdk POLICY peer subscribes `SensorJoy` and re-emits each sample as
our `JoyPacket` UDP datagram to the brain's `SocketJoystickSource` — so the unmodified
brain is steered by the unmodified vendor pad. Axes/buttons pass straight through (the
vendor layout IS what `walk_controller`/our `Teleop` expect: axis0=v_y, axis1=v_x,
axis3=w_z; buttons in PlayStation indices, R1+X = WALK, L1+Y = STAND).

    conda run -n limx python logic/simulation/mujoco/sensorjoy_bridge.py --joy-port 9001

Reuses the joystick wire (`reason/teleoperation/joystick/protocol.py`, pure stdlib) as a
read-only format — it does not touch the brain's teleop code.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.reason.teleoperation.joystick.protocol import (  # noqa: E402
    JoyPacket,
    pack_joy,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward vendor SensorJoy → JoyPacket/UDP.")
    ap.add_argument("--host", default="127.0.0.1", help="brain host to send JoyPackets to")
    ap.add_argument("--joy-port", type=int, default=9001, help="brain SocketJoystickSource port")
    ap.add_argument("--robot-ip", default="127.0.0.1", help="bus IP (loopback for sim)")
    args = ap.parse_args()

    import limxsdk.robot.Robot as Robot
    import limxsdk.robot.RobotType as RobotType

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.joy_port)
    n_fwd = [0]

    def on_joy(sj) -> None:  # fires on an SDK thread
        pkt = JoyPacket(
            stamp_ns=time.monotonic_ns(),
            axes=[float(a) for a in sj.axes],
            buttons=[int(b) for b in sj.buttons],
        )
        try:
            sock.sendto(pack_joy(pkt), dest)
            n_fwd[0] += 1
        except OSError:
            pass

    robot = Robot(RobotType.Humanoid)  # is_sim=False → POLICY role (sees SensorJoy)
    if not robot.init(args.robot_ip):
        raise SystemExit("[joy-bridge] robot.init failed — is the sim up?")
    robot.subscribeSensorJoy(on_joy)
    print(f"[joy-bridge] vendor SensorJoy → JoyPacket udp://{args.host}:{args.joy_port}",
          flush=True)
    try:
        while True:
            time.sleep(1.0)
            print(f"[joy-bridge] forwarded {n_fwd[0]} SensorJoy packets", flush=True)
    except KeyboardInterrupt:
        print("\n[joy-bridge] stopping.", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
