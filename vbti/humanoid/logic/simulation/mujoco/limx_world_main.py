"""limx_world_main.py — the MuJoCo (and later real) World EDGE process.

This is the py3.8 `limx`-env half of the World for the bus path. It is a limxsdk
POLICY peer (drop-in for the deploy `walk_controller`, minus the ONNX) that bridges the
MROS bus to our UDS contracts, so the unchanged Brain drives the unchanged MuJoCo sim:

    Brain (py3.11)  ──UDS, 3 PR contracts──►  THIS EDGE (py3.11→limxsdk policy peer)
                                                   │ MROS bus
                                                   ▼
                              simulator.py + kinematic_projection ELF (AB physics + PD)

The edge owns the 1 kHz `RobotCmd` republish (LD3): the Brain emits ~100 Hz `PolicyOut`;
the edge streams the last one to the bus every tick (the cmd is a setpoint, not latched).
Physics, PD, and PR↔AB all live in the untouched sim process — never here.

Run order (the ELF must be up before RobotState/RobotCmd flow — memory
`limx-kinematic-projection-bus-relay`):

    # terminal 1 — the MuJoCo sim (env `limx`), unchanged, spawns the ELF:
    python vendor/humanoid-mujoco-sim/simulator.py
    # terminal 2 — this edge (env `limx`):
    python logic/simulation/mujoco/limx_world_main.py
    # terminal 3 — the brain (env `brain`), unchanged + our joystick:
    python logic/oli/brain_main.py --joystick socket

References: spec `docs/superpowers/specs/2026-06-25-mujoco-limx-world-design.md`
(LD1–LD6); design.md D4/D8/D9/D10.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.oli.comm.world import WorldComm, WorldCommError  # noqa: E402
from humanoid.logic.oli.contracts import PolicyOut  # noqa: E402


class EdgeState:
    """Mutable per-loop state: the last command held + ticks since a fresh one."""

    __slots__ = ("last", "stale_ticks", "n_cmds")

    def __init__(self) -> None:
        self.last: Optional[PolicyOut] = None
        self.stale_ticks: int = 0
        self.n_cmds: int = 0


def edge_tick(worldcomm, body, state: EdgeState, *, watchdog_ticks: int,
              kd_damp: float) -> EdgeState:
    """One edge iteration (NO sleep — the caller paces). Pure enough to unit-test.

    Publish the latest bus Observation (stamp = bus/sim time, D8), drain the newest
    PolicyOut (latest-wins), then drive the bus:
      - no command yet  → publish nothing (freeze-until-first-cmd, D9);
      - fresh/held cmd  → republish it (hold last between policy steps);
      - brain gone stale → damp in place (kp=0, small kd, q_des=q) as a fail-safe.
    """
    worldcomm.publish(stamp_ns=body.latest_stamp_ns())

    cmd = worldcomm.receive_latest()
    if cmd is not None:
        state.last = cmd
        state.stale_ticks = 0
        state.n_cmds += 1
    else:
        state.stale_ticks += 1

    if state.last is None:
        return state  # D9: nothing to apply until the brain's first command

    if state.stale_ticks > watchdog_ticks:
        q_now, _, _ = body.read_joints_isaac()  # native order
        n = q_now.shape[0]
        z = np.zeros(n, dtype=np.float32)
        body.apply_isaac(q_now, z, z, z, np.full(n, kd_damp, dtype=np.float32))
    else:
        worldcomm.apply(state.last)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/oli-world.sock")
    ap.add_argument("--robot-ip", default="127.0.0.1")
    ap.add_argument("--rate", type=float, default=1000.0,
                    help="bus republish rate (Hz); match the deploy 1 kHz")
    ap.add_argument("--watchdog-ms", type=float, default=200.0,
                    help="damp in place if no fresh PolicyOut for this long")
    ap.add_argument("--kd-damp", type=float, default=5.0)
    ap.add_argument("--serve-timeout", type=float, default=120.0)
    args = ap.parse_args()

    # limxsdk is only needed live (py3.8 `limx` env); import lazily.
    import limxsdk.robot.Rate as Rate

    from humanoid.logic.simulation.mujoco.limx_body import connect

    print(f"[edge] connecting to the bus at {args.robot_ip} (policy role)...", flush=True)
    body = connect(args.robot_ip)
    print("[edge] waiting for the first RobotState + ImuData...", flush=True)
    while not body.ready():
        time.sleep(0.01)
    print("[edge] bus live. serving the brain on "
          f"{args.socket}; waiting for connect...", flush=True)

    worldcomm = WorldComm(body, socket_path=args.socket)
    worldcomm.serve(timeout=args.serve_timeout)
    print("[edge] brain connected. relaying (freeze-until-first-cmd).", flush=True)

    watchdog_ticks = int(args.watchdog_ms / 1000.0 * args.rate)
    state = EdgeState()
    rate = Rate(args.rate)
    t_log = time.monotonic()
    try:
        while True:
            edge_tick(worldcomm, body, state,
                      watchdog_ticks=watchdog_ticks, kd_damp=args.kd_damp)
            if time.monotonic() - t_log > 1.0:
                print(f"[edge] cmds={state.n_cmds} stale_ticks={state.stale_ticks}",
                      flush=True)
                t_log = time.monotonic()
            rate.sleep()
    except KeyboardInterrupt:
        print("\n[edge] stopping.", flush=True)
    except WorldCommError as e:
        print(f"[edge] brain disconnected: {e}; stopping.", flush=True)
    finally:
        worldcomm.close()


if __name__ == "__main__":
    main()
