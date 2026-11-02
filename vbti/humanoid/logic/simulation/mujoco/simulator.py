#!/usr/bin/env python3
"""
Humanoid MuJoCo simulator launcher — one-command bring-up of the LimX stack.

Orchestrates three vendor processes plus an auto-stand trigger:

  1. vendor/humanoid-mujoco-sim/simulator.py    — MuJoCo, robot-side (publishes RobotState/IMU)
  2. vendor/humanoid-rl-deploy-python/main.py   — controller dispatcher + joystick callbacks
  3. vendor/humanoid-mujoco-sim/robot-joystick/robot-joystick   — visual SensorJoy pad

Stand publishes immediately on bring-up because controllers.yaml has
`stand.autostart: true` (vendor patch). On sim restart we re-trigger stand via
`limxsdk.ability.cli switch` so the controller re-reads the fresh sim's joint
state and interpolates smoothly to stand_pos.

Interactive commands (type a single letter + Enter in the launcher terminal):

  r   restart the MuJoCo simulator subprocess (PD/IMU/state all reset) and re-trigger stand
  q   quit everything

Ctrl+C also quits cleanly.

Run:

  conda activate limx
  python humanoid/logic/simulation/mujoco/simulator.py

Run without joystick (headless tests):

  python humanoid/logic/simulation/mujoco/simulator.py --no-joystick

Run sim + kinematic_projection ELF ONLY — no LimX deploy policy — so OUR limx edge
(`limx_world_main.py`) drives it as the policy peer (implies --no-joystick):

  python humanoid/logic/simulation/mujoco/simulator.py --no-deploy

Notes
-----

- This launcher only orchestrates vendor binaries. It does not reimplement any
  control logic. See `docs/vendor/humanoid-mujoco-sim.md` and
  `docs/vendor/humanoid-rl-deploy-python.md` for what each subprocess does.
- Must run inside the `limx` conda env (Python 3.8, limxsdk wheel installed).
  See `requirements/README.md`.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parents[3]   # humanoid/
VENDOR = REPO_ROOT / "vendor"
SIM_REPO = VENDOR / "humanoid-mujoco-sim"
DEPLOY_REPO = VENDOR / "humanoid-rl-deploy-python"

SIM_ENTRY = SIM_REPO / "simulator.py"
DEPLOY_ENTRY = DEPLOY_REPO / "main.py"
JOYSTICK_BIN = SIM_REPO / "robot-joystick" / "robot-joystick"

# --------------------------------------------------------------------------- logging

log = logging.getLogger("launcher")
logging.basicConfig(
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# --------------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default=os.environ.get("ROBOT_TYPE", "HU_D04_01"),
                   help="LimX robot type (sets ROBOT_TYPE env). Default: HU_D04_01.")
    p.add_argument("--ip", default="127.0.0.1",
                   help="Robot IP. Loopback for sim, 10.192.1.2 for real robot. Default: 127.0.0.1.")
    p.add_argument("--no-joystick", action="store_true",
                   help="Do not launch the visual SensorJoy pad (headless tests).")
    p.add_argument("--no-deploy", action="store_true",
                   help="Run sim + ELF only; do NOT launch the LimX deploy policy. Use when "
                        "OUR edge (limx_world_main.py) is the policy peer. Implies --no-joystick.")
    p.add_argument("--initial-mode", default="stand",
                   choices=["stand", "walk", "mimic", "damping"],
                   help="Ability to re-trigger after a sim restart. Default: stand. (Bring-up uses controllers.yaml autostart and does not call cli switch.)")
    return p.parse_args()


# --------------------------------------------------------------------------- subprocess wrapper

@dataclass
class Proc:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict = field(default_factory=dict)
    proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        full_env = {**os.environ, **self.env}
        log.info("starting %s …  (%s)", self.name, " ".join(str(c) for c in self.cmd))
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=full_env,
            stdin=subprocess.DEVNULL,  # detach from launcher TTY so our stdin loop owns keys
            stdout=None,               # inherit — let vendor logs hit our terminal
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,      # own process group so we can SIGTERM cleanly
        )

    def stop(self, timeout: float = 3.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        log.info("stopping %s (pid %d) …", self.name, self.proc.pid)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("%s did not exit in %.1fs, sending SIGKILL", self.name, timeout)
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()
        self.proc = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


# --------------------------------------------------------------------------- orphan reaper

# Match cmdlines of any LimX-stack process this launcher (or a previous crashed run) may leave behind.
ORPHAN_PATTERNS = [
    re.compile(r"humanoid-mujoco-sim/simulator\.py"),
    re.compile(r"humanoid-rl-deploy-python/main\.py"),
    re.compile(r"limxsdk\.ability\.cli"),
    re.compile(r"robot-joystick/robot-joystick"),
    re.compile(r"prebuild/kinematic_projection"),
]


def find_orphans(exclude_pids: set[int] | None = None) -> list[tuple[int, str]]:
    """Return [(pid, cmdline)] for any LimX-stack process not in exclude_pids."""
    exclude_pids = exclude_pids or set()
    out: list[tuple[int, str]] = []
    my_uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude_pids or pid == os.getpid():
            continue
        try:
            if entry.stat().st_uid != my_uid:
                continue
            cmdline_bytes = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        cmdline = cmdline_bytes.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if any(p.search(cmdline) for p in ORPHAN_PATTERNS):
            out.append((pid, cmdline))
    return out


def reap_orphans(exclude_pids: set[int] | None = None, label: str = "") -> int:
    """SIGTERM then SIGKILL any LimX orphans. Returns count killed."""
    orphans = find_orphans(exclude_pids)
    if not orphans:
        return 0
    log.warning("%s%d LimX orphan process(es) found — reaping", f"[{label}] " if label else "", len(orphans))
    for pid, cmd in orphans:
        log.warning("  → pid %d: %s", pid, cmd[:120])
    for pid, _ in orphans:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    survivors = find_orphans(exclude_pids)
    for pid, _ in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(orphans)


# --------------------------------------------------------------------------- ability switch helper


def cli_switch(from_modes: str, to_mode: str, timeout: float = 5.0, quiet: bool = False) -> None:
    """Fire `limxsdk.ability.cli switch <from> <to>` synchronously.

    `quiet=True` suppresses the cli's stderr (used during shutdown where
    `ConnectionResetError` against a dying HTTP server is expected and noisy).
    """
    cmd = [sys.executable, "-m", "limxsdk.ability.cli", "switch", from_modes, to_mode]
    log.info("ability switch: %r → %r", from_modes, to_mode)
    try:
        subprocess.run(
            cmd,
            check=False,
            timeout=timeout,
            stderr=subprocess.DEVNULL if quiet else None,
            stdout=subprocess.DEVNULL if quiet else None,
        )
    except subprocess.TimeoutExpired:
        log.warning("cli switch timed out — ability framework may not be up yet")


# --------------------------------------------------------------------------- the launcher


class Launcher:
    ALL_MODES = "stand walk mimic damping"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # PYTHONUNBUFFERED=1 propagates through main.py AND its grandchild
        # `python3 -m limxsdk.ability.cli load …` (spawned via os.system) so any
        # walk/mimic controller traceback hits our TTY immediately, not buffered
        # behind the controller's `\r`-style FPS counter.
        env = {
            "ROBOT_TYPE": args.robot,
            "ROBOT_IP": args.ip,
            "PYTHONUNBUFFERED": "1",
        }

        self.sim = Proc(
            name="simulator",
            cmd=[sys.executable, str(SIM_ENTRY), args.ip],
            cwd=SIM_REPO, env=env,
        )
        self.deploy: Optional[Proc] = None
        if not args.no_deploy:
            self.deploy = Proc(
                name="deploy",
                cmd=[sys.executable, str(DEPLOY_ENTRY), args.ip],
                cwd=DEPLOY_REPO, env=env,
            )
        # The LimX SensorJoy pad only feeds the deploy policy; with --no-deploy our
        # brain owns the joystick, so there is nothing to feed — skip it.
        self.joystick: Optional[Proc] = None
        if not args.no_joystick and not args.no_deploy:
            self.joystick = Proc(
                name="joystick",
                cmd=[str(JOYSTICK_BIN)],
                cwd=JOYSTICK_BIN.parent, env=env,
            )

        self._stop = threading.Event()
        # True while restart_sim() is mid-flight — tells the monitor loop to ignore
        # the intentional simulator-dead gap and not declare a phantom crash.
        self._restarting = threading.Event()

    # ---- lifecycle

    def bring_up(self) -> None:
        reap_orphans(label="startup")          # clear leftovers from any prior crashed run
        self.sim.start()
        time.sleep(1.0)            # let MuJoCo load XML before a controller attaches
        if self.deploy:
            self.deploy.start()
        if self.joystick:
            self.joystick.start()

        if self.deploy:
            # No cli_switch needed: controllers.yaml has `stand.autostart: true`, so the
            # deploy stack publishes RobotCmd the moment it finishes initialising.
            log.info("ready. autostarted mode = stand. press joystick buttons to switch.")
        else:
            log.info("ready — sim + ELF only, NO deploy policy. Now drive it with OUR stack:")
            log.info("    conda run -n limx  python logic/simulation/mujoco/limx_world_main.py")
            log.info("    conda run -n brain python logic/oli/brain_main.py --joystick socket")
        self._print_help()

    def shutdown(self) -> None:
        # Release current ability cleanly — but only if the deploy stack is still
        # alive. Talking to a half-dead HTTP server raises ConnectionResetError.
        if self.deploy and self.deploy.alive():
            cli_switch(self.ALL_MODES, "", timeout=1.0, quiet=True)
        for p in (self.joystick, self.deploy, self.sim):
            if p:
                p.stop()
        # main.py spawns `python3 -m limxsdk.ability.cli load` via os.system; if that
        # subprocess got reparented to init (e.g. parent SIGSEGV'd), our process-group
        # kill misses it. Final pass catches any survivors.
        reap_orphans(label="shutdown")

    # ---- restart

    def restart_sim(self) -> None:
        log.info("=== restarting simulator ===")
        self._restarting.set()
        try:
            self.sim.stop()
            time.sleep(0.5)
            self.sim.start()
            time.sleep(1.0)         # let MuJoCo reload the XML + open viewer
            if self.deploy:
                # Re-trigger the initial mode: StandController.on_start() re-reads
                # robot_state.q as init_joint_angles, so the 2 s linear ramp now runs
                # from the *fresh* sim's rest pose to stand_pos — smooth, no jerk.
                cli_switch(self.ALL_MODES, self.args.initial_mode, timeout=2.0)
                log.info("=== restart complete, mode = %s ===", self.args.initial_mode)
            else:
                log.info("=== restart complete (sim only); our edge resumes on the fresh sim ===")
        finally:
            self._restarting.clear()

    # ---- main loop

    def _print_help(self) -> None:
        print()
        print("─" * 60)
        print(" Launcher commands (type letter + Enter):")
        print("   r  restart MuJoCo simulator (re-triggers stand)")
        print("   q  quit everything")
        print("   Ctrl+C also quits cleanly")
        print("─" * 60, flush=True)

    def _stdin_loop(self) -> None:
        """Background thread: read single-letter commands from stdin."""
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not ready:
                continue
            line = sys.stdin.readline().strip().lower()
            if line in ("q", "quit", "exit"):
                self._stop.set()
                return
            elif line in ("r", "restart"):
                self.restart_sim()
            elif line == "":
                continue
            else:
                log.info("unknown command %r — try 'r' or 'q'", line)

    def run(self) -> int:
        self.bring_up()

        stdin_thread = threading.Thread(target=self._stdin_loop, name="stdin", daemon=True)
        stdin_thread.start()

        try:
            while not self._stop.is_set():
                # watch for unexpected subprocess crashes — but skip the sim check
                # while a restart is in flight (the dead-sim gap is intentional).
                for p in (self.sim, self.deploy, self.joystick):
                    if not p:
                        continue
                    if p is self.sim and self._restarting.is_set():
                        continue
                    if not p.alive():
                        log.error("%s exited unexpectedly (code %s) — shutting down",
                                  p.name, p.proc.returncode if p.proc else "?")
                        self._stop.set()
                        break
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down")
        finally:
            self.shutdown()
        return 0


# --------------------------------------------------------------------------- entry


def main() -> int:
    args = parse_args()

    # sanity-check vendor presence
    required = [SIM_ENTRY] if args.no_deploy else [SIM_ENTRY, DEPLOY_ENTRY]
    for p in required:
        if not p.exists():
            log.error("vendor file not found: %s", p)
            return 2
    if not args.no_joystick and not args.no_deploy and not JOYSTICK_BIN.exists():
        log.error("joystick binary not found: %s (use --no-joystick to skip)", JOYSTICK_BIN)
        return 2

    return Launcher(args).run()


if __name__ == "__main__":
    sys.exit(main())
