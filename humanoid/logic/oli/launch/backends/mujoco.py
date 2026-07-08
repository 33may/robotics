"""mujoco.py — launcher backend for the LimX MuJoCo bus stack.

Boots three core processes: the vendor MuJoCo sim (which spawns the `kinematic_projection`
ELF) in the limx env, then our limxsdk POLICY-peer edge (`limx_world_main.py`) which serves
the brain socket, then the brain in the brain env. In `--mode walk` it also launches the
EXACT vendor joystick pad + the SensorJoy→JoyPacket bridge as *extras* (non-core: closing
the pad window must not kill the sim).

`reap()` kills any bus-stack processes a crashed run left behind (they block the MROS bus).
The `*_argv` builders are pure (unit-tested without spawning). Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
from pathlib import Path

from ..supervisor import Stage

NAME = "mujoco"

_HERE = Path(__file__).resolve()
_HUMANOID = _HERE.parents[4]                        # humanoid/
_REPO_ROOT = _HERE.parents[5]                       # repo root
_MJ_DIR = _HUMANOID / "logic" / "simulation" / "mujoco"
SIM_REPO = _HUMANOID / "vendor" / "humanoid-mujoco-sim"
SIM_ENTRY = SIM_REPO / "simulator.py"               # vendor sim (spawns the ELF itself)
EDGE_ENTRY = _MJ_DIR / "limx_world_main.py"
BRAIN_ENTRY = _HUMANOID / "logic" / "oli" / "brain_main.py"
BRIDGE_ENTRY = _MJ_DIR / "sensorjoy_bridge.py"      # vendor SensorJoy → our JoyPacket
JOYSTICK_BIN = SIM_REPO / "robot-joystick" / "robot-joystick"

#: default forward speed (m/s) for --mode forward without an explicit --vx
_FORWARD_VX = 0.3

#: substring the EDGE prints right after it binds the socket and waits for the brain
_SERVING_MARKER = "serving the brain"

# Cmdline fragments of any bus-stack process a crashed run may leave behind (block the bus).
_ORPHAN_PATTERNS = [
    re.compile(r"humanoid-mujoco-sim/simulator\.py"),
    re.compile(r"prebuild/kinematic_projection"),
    re.compile(r"limx_world_main\.py"),
    re.compile(r"sensorjoy_bridge\.py"),
    re.compile(r"robot-joystick/robot-joystick"),
]


# ── CLI (backend-specific flags; the launcher adds the common brain/mode flags) ──────

def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ip", default="127.0.0.1", help="bus IP (loopback for sim)")
    ap.add_argument("--rate", type=float, default=1000.0)
    ap.add_argument("--watchdog-ms", type=float, default=200.0)
    ap.add_argument("--sim-delay", type=float, default=3.0,
                    help="seconds to let MuJoCo load + the ELF come up before the edge")
    ap.add_argument("--sim-env", default="limx")


def _sim_env(a: argparse.Namespace) -> dict:
    return {"ROBOT_TYPE": "HU_D04_01", "ROBOT_IP": a.ip, "PYTHONUNBUFFERED": "1"}


# ── pure command builders ────────────────────────────────────────────────────────

def sim_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the vendor MuJoCo sim (+ ELF). NO deploy policy runs."""
    return ["conda", "run", "--no-capture-output", "-n", a.sim_env, "python", "-u",
            str(SIM_ENTRY), a.ip]


def edge_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for our limxsdk policy-peer edge (serves the brain socket)."""
    return ["conda", "run", "--no-capture-output", "-n", a.sim_env, "python", "-u",
            str(EDGE_ENTRY),
            "--socket", a.socket,
            "--robot-ip", a.ip,
            "--rate", str(a.rate),
            "--watchdog-ms", str(a.watchdog_ms)]


def brain_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the brain, derived from --mode (stand/walk/forward)."""
    py = ["conda", "run", "--no-capture-output", "-n", a.brain_env, "python", "-u",
          str(BRAIN_ENTRY), "--socket", a.socket]
    if a.mode == "stand":
        py += ["--mode", "stand", "--joystick", "fixed"]
    elif a.mode == "walk":
        # operator-steered: the launcher owns the vendor pad + bridge; the brain only
        # LISTENS on the joystick socket (it never spawns an app itself).
        py += ["--mode", "walk", "--joystick", "socket", "--joy-port", str(a.joy_port)]
    else:  # forward
        vx = a.vx if a.vx is not None else _FORWARD_VX
        py += ["--mode", "walk", "--joystick", "fixed",
               "--vx", str(vx), "--vy", str(a.vy or 0.0), "--wz", str(a.wz or 0.0)]
    if a.walk_after is not None:
        py += ["--walk-after", str(a.walk_after)]
    if a.duration:
        py += ["--duration", str(a.duration)]
    return py


def bridge_argv(a: argparse.Namespace) -> list[str]:
    """conda-run argv for the SensorJoy→JoyPacket bridge (vendor pad → brain)."""
    return ["conda", "run", "--no-capture-output", "-n", a.sim_env, "python", "-u",
            str(BRIDGE_ENTRY),
            "--host", "127.0.0.1",
            "--joy-port", str(a.joy_port),
            "--robot-ip", a.ip]


def joystick_argv(a: argparse.Namespace) -> list[str]:
    """argv for the EXACT vendor robot-joystick binary (self-contained PyInstaller app)."""
    return [str(JOYSTICK_BIN)]


# ── orphan reaping (leftover sim/ELF/edge from a crashed run block the bus) ────────

def reap(label: str = "") -> int:
    my_uid = os.getuid()
    killed = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if entry.stat().st_uid != my_uid:
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(p.search(cmdline) for p in _ORPHAN_PATTERNS):
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
    if killed:
        print(f"[run] reaped {killed} bus orphan(s) [{label}]", flush=True)
    return killed


# ── the ordered boot plan ──────────────────────────────────────────────────────────

def stages(a: argparse.Namespace) -> list[Stage]:
    if a.mode == "glide":
        raise SystemExit("--mode glide is isaac-only (no MuJoCo glide world yet)")
    if getattr(a, "dev_app", False):
        raise SystemExit("--dev-app is not wired for --sim mujoco yet (isaac only)")

    st = [
        Stage("sim", sim_argv(a), cwd=SIM_REPO, env=_sim_env(a), core=True),
        Stage("edge", edge_argv(a), cwd=_REPO_ROOT,
              serving_marker=_SERVING_MARKER, wait_for_path=a.socket,
              core=True, boot_delay=a.sim_delay),
        Stage("brain", brain_argv(a), cwd=_REPO_ROOT, core=True),
    ]
    if a.mode == "walk":
        st.append(Stage("joy-bridge", bridge_argv(a), cwd=_REPO_ROOT, core=False))
        if JOYSTICK_BIN.exists():
            st.append(Stage("joystick", joystick_argv(a), cwd=JOYSTICK_BIN.parent,
                            env=_sim_env(a), core=False))
    return st
