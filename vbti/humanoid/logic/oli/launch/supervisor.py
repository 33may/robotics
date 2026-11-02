"""supervisor.py — generic, backend-agnostic process supervisor for the Oli launcher.

Boots an ordered list of `Stage`s (each a child process), tees every child's merged
stdout+stderr into ONE timestamped stream printed live AND written to a shared log,
gates on serving markers where a stage declares one, supervises the *core* processes,
and tears the whole stack down (reverse order, SIGINT→SIGTERM→SIGKILL per group) on
Ctrl-C or any core process exiting.

This is the single copy of the plumbing that `run_oli_sim.py` and `run_oli_mujoco.py`
used to each carry. It knows NOTHING about isaac/mujoco/real — a backend just hands it
Stages. Stdlib only (imports no isaacsim/limxsdk/contracts).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Stage:
    """One child process in the boot sequence.

    name           : short tag for banners/logs (e.g. "world", "edge", "brain").
    argv           : the full command (typically a `conda run … python -u …` list).
    cwd            : working directory for the child.
    serving_marker : if set, the supervisor waits until the child prints this substring
                     before starting the NEXT stage (e.g. "serving on").
    wait_for_path  : if set, after the marker also wait until this file exists — the UDS
                     socket the next stage connects to (avoids a connect race).
    env            : extra environment merged over os.environ.
    core           : a core stage exiting tears the stack down; extras (joystick pad,
                     bridge) do not — closing the pad window must not kill the sim.
    boot_delay     : sleep this long BEFORE spawning (let a prior child settle, e.g. give
                     MuJoCo time to load + register the ELF on the bus before the edge).
    """

    name: str
    argv: list[str]
    cwd: Path
    serving_marker: str | None = None
    wait_for_path: str | None = None
    env: dict | None = None
    core: bool = True
    boot_delay: float = 0.0


class _StackDown(Exception):
    """Internal: a core child exited / failed to boot → unwind to teardown."""


class Supervisor:
    def __init__(self, log_path, boot_timeout: float = 240.0,
                 reap: Callable[[str], int] | None = None):
        self.log_path = Path(log_path)
        self.boot_timeout = boot_timeout
        self._reap = reap
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._sink = None

    # ── logging: one interleaved, timestamped stream to console + shared log ──────
    def _emit(self, text: str) -> None:
        with self._lock:
            sys.stdout.write(text)
            sys.stdout.flush()
            if self._sink is not None:
                self._sink.write(text)
                self._sink.flush()

    def banner(self, msg: str) -> None:
        self._emit(f"[+{time.monotonic() - self._start:6.1f}s] [run] {msg}\n")

    # ── child plumbing ───────────────────────────────────────────────────────────
    def _spawn(self, stage: Stage) -> subprocess.Popen:
        return subprocess.Popen(
            stage.argv,
            cwd=str(stage.cwd),
            env={**os.environ, **stage.env} if stage.env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into the teed stream
            text=True,
            bufsize=1,                  # line-buffered (valid in text mode)
            start_new_session=True,     # own group → killpg tears down grandchildren (ELF)
        )

    def _pump(self, proc: subprocess.Popen, marker_event, marker) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            self._emit(f"[+{time.monotonic() - self._start:6.1f}s] {line}\n")
            if marker_event is not None and marker in line:
                marker_event.set()

    def _shutdown(self, proc: subprocess.Popen, tag: str) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        for sig, grace in ((signal.SIGINT, 4.0), (signal.SIGTERM, 3.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            self.banner(f"{tag}: sent {sig.name}, waiting {grace:.0f}s")
            try:
                proc.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                break   # impatient operator hit ^C again → SIGKILL the group now
        try:
            os.killpg(pgid, signal.SIGKILL)
            self.banner(f"{tag}: SIGKILL")
        except ProcessLookupError:
            pass

    # ── boot orchestration ─────────────────────────────────────────────────────────
    def run(self, stages: list[Stage], dry_run: bool = False) -> int:
        if dry_run:
            return self._print_stages(stages)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = self.log_path.open("w")
        self.banner(f"log → {self.log_path}")
        if self._reap is not None:
            self._reap("startup")

        started: list[tuple[subprocess.Popen, Stage, threading.Thread]] = []
        rc = 0

        def core_died():
            for proc, stage, _ in started:
                if stage.core and proc.poll() is not None:
                    return proc, stage
            return None

        try:
            for stage in stages:
                # 0) optional settle delay, aborting if a core child already died
                if stage.boot_delay:
                    deadline = time.monotonic() + stage.boot_delay
                    while time.monotonic() < deadline:
                        dead = core_died()
                        if dead:
                            self.banner(f"{dead[1].name} exited (code {dead[0].returncode}) "
                                        f"during boot — aborting")
                            rc = 1
                            raise _StackDown
                        time.sleep(0.1)

                # 1) spawn + start teeing this child
                self.banner(f"launching {stage.name}: " + " ".join(stage.argv))
                proc = self._spawn(stage)
                ev = threading.Event() if stage.serving_marker else None
                t = threading.Thread(
                    target=self._pump, args=(proc, ev, stage.serving_marker), daemon=True)
                t.start()
                started.append((proc, stage, t))

                # 2) gate on the serving marker before booting the next stage
                if stage.serving_marker:
                    assert ev is not None
                    while not ev.is_set():
                        dead = core_died()
                        if dead:
                            self.banner(f"{dead[1].name} exited (code {dead[0].returncode}) "
                                        f"before {stage.name} served — aborting")
                            rc = 1
                            raise _StackDown
                        if time.monotonic() - self._start > self.boot_timeout:
                            self.banner(f"{stage.name} did not serve within "
                                        f"{self.boot_timeout:.0f}s — aborting")
                            rc = 1
                            raise _StackDown
                        time.sleep(0.2)

                # 3) wait for the UDS socket to appear (the next stage connects to it)
                if stage.wait_for_path:
                    sock_deadline = time.monotonic() + 10.0
                    while (not os.path.exists(stage.wait_for_path)
                           and time.monotonic() < sock_deadline):
                        if core_died():
                            self.banner(f"{stage.name} exited while binding "
                                        f"{stage.wait_for_path} — aborting")
                            rc = 1
                            raise _StackDown
                        time.sleep(0.05)

            # 4) supervise: run until a CORE child exits, then tear the stack down
            self.banner("stack up — supervising (Ctrl-C to stop)")
            while True:
                dead = core_died()
                if dead:
                    self.banner(f"{dead[1].name} exited (code {dead[0].returncode}) "
                                f"— stopping the stack")
                    break
                time.sleep(0.2)
        except _StackDown:
            pass
        except KeyboardInterrupt:
            self.banner("Ctrl-C — tearing down the stack")
        finally:
            for proc, stage, _ in reversed(started):   # extras first, core last
                self._shutdown(proc, stage.name)
            if self._reap is not None:
                self._reap("shutdown")
            for _, _, t in started:
                t.join(timeout=2.0)
            self.banner(f"done; full log at {self.log_path}")
            if self._sink is not None:
                self._sink.close()
        return rc

    def _print_stages(self, stages: list[Stage]) -> int:
        """--dry-run: show the resolved boot plan without spawning anything."""
        print(f"[dry-run] {len(stages)} stage(s), boot-timeout {self.boot_timeout:.0f}s, "
              f"log → {self.log_path}")
        for i, s in enumerate(stages, 1):
            gate = f"  ⟶ wait '{s.serving_marker}'" if s.serving_marker else ""
            sock = f"  ⟶ wait file {s.wait_for_path}" if s.wait_for_path else ""
            delay = f"  (after {s.boot_delay:.0f}s settle)" if s.boot_delay else ""
            kind = "" if s.core else " [extra]"
            print(f"  {i}. {s.name}{kind}{delay}: {' '.join(s.argv)}{gate}{sock}")
        return 0
