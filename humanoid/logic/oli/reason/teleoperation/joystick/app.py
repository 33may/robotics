"""app.py — the joystick app: a keyboard-driven on-screen pad → JoyPacket over UDP.

A from-scratch port of LimX's `robot-joystick` (vendored under
`humanoid-mujoco-sim/robot-joystick`), which is a PyInstaller-frozen pygame window that
reads the keyboard and publishes `SensorJoy`. We keep its UX (so muscle memory carries
over) but publish OUR `JoyPacket` over UDP instead of `limxsdk.SensorJoy` on the MROS
bus — so the app needs neither `limxsdk` nor a physical gamepad, and the brain stays
pure. Spawn it standalone:  ``python -m humanoid.logic.oli.reason.teleoperation.joystick.app``

Bindings (from the LimX app, see robot-joystick/doc/joystick.png):
    Arrow keys     left stick   → translation   (Up/Down = v_x, Left/Right = v_y)
    Numpad 4 / 6   right stick X → yaw           (w_z)
    Q + I          L1 + Y       → STAND mode
    U + J          R1 + X       → WALK mode

The keyboard→packet mapping (`keyboard_to_packet`) is PURE and unit-tested; pygame is
imported lazily inside `run()` so this module imports cleanly in the pygame-less brain.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Iterable

from .protocol import JoyPacket, pack_joy

# Axes follow walk_controller: axis0 = v_y, axis1 = v_x, axis2 unused, axis3 = w_z.
NUM_AXES: int = 4
# Buttons follow the PlayStation indices main.py reads (L1=4, Y=3 → stand; R1=7, X=2 → walk).
NUM_BUTTONS: int = 8
BUTTON_INDEX = {
    "b_A": 0, "b_B": 1, "b_X": 2, "b_Y": 3, "b_L1": 4, "b_L2": 5, "b_R1": 7,
}


def keyboard_to_packet(held: Iterable[str], stamp_ns: int) -> JoyPacket:
    """Map a set of logical key names to a JoyPacket (pure; no pygame).

    `held` are logical names ("up"/"down"/"left"/"right"/"yaw_left"/"yaw_right" and
    button names "b_L1"/"b_Y"/...). Opposed keys cancel. The `run()` loop translates
    physical pygame keys to these names, keeping this core testable and pygame-free.
    """
    held = set(held)

    def axis(pos: str, neg: str) -> float:
        return (1.0 if pos in held else 0.0) - (1.0 if neg in held else 0.0)

    axes = [0.0] * NUM_AXES
    axes[0] = axis("right", "left")     # v_y
    axes[1] = axis("up", "down")        # v_x
    axes[3] = axis("yaw_left", "yaw_right")  # w_z

    buttons = [0] * NUM_BUTTONS
    for name, idx in BUTTON_INDEX.items():
        if name in held:
            buttons[idx] = 1

    return JoyPacket(stamp_ns=stamp_ns, axes=axes, buttons=buttons)


def run(host: str = "127.0.0.1", port: int = 9001, hz: float = 50.0) -> None:
    """Open the pygame window and stream JoyPackets to (host, port) at ~hz.

    Imports pygame lazily so the module stays importable in the brain env. Sends a full
    stick state every tick (latest-wins on the receiver), so a dropped datagram is a
    non-event. Close the window or Ctrl-C to stop.
    """
    import pygame  # lazy: only the app process needs pygame

    bindings = {
        pygame.K_UP: "up", pygame.K_DOWN: "down",
        pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
        pygame.K_KP4: "yaw_left", pygame.K_KP6: "yaw_right",
        pygame.K_q: "b_L1", pygame.K_i: "b_Y",   # Q+I → stand
        pygame.K_u: "b_R1", pygame.K_j: "b_X",   # U+J → walk
        pygame.K_k: "b_A", pygame.K_l: "b_B", pygame.K_e: "b_L2",
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pygame.init()
    screen = pygame.display.set_mode((520, 220))
    pygame.display.set_caption(f"Oli Joystick → {host}:{port}")
    font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()
    hud = [
        "Arrows: walk   Numpad 4/6: turn",
        "Q+I: STAND     U+J: WALK",
        "(focus this window to drive)",
    ]

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pressed = pygame.key.get_pressed()
            held = {name for key, name in bindings.items() if pressed[key]}
            pkt = keyboard_to_packet(held, stamp_ns=time.monotonic_ns())
            sock.sendto(pack_joy(pkt), (host, port))

            screen.fill((24, 24, 28))
            for i, line in enumerate(hud):
                screen.blit(font.render(line, True, (210, 210, 210)), (16, 16 + i * 26))
            live = f"axes={[round(a, 2) for a in pkt.axes]}"
            screen.blit(font.render(live, True, (120, 220, 140)), (16, 150))
            pygame.display.flip()
            clock.tick(hz)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Oli keyboard joystick → JoyPacket/UDP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()
    run(host=args.host, port=args.port, hz=args.hz)


if __name__ == "__main__":
    main()
