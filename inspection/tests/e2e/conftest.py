"""Fixtures for the headless e2e suite (task-11): launch the mock app as a
real subprocess — same HTTP + bus wiring `ui/app.py mock` uses — on free
ports, and hand the test a live URL plus the run directory it is writing to.

The subprocess runs `_mock_launcher.py`, not `ui/app.py` directly — see that
module's docstring for why (a sandbox-local planner substitution, not a
change in what is being tested).

Each fixture is function-scoped: a fresh subprocess, fresh ports and a fresh
run directory per test, because `RunWriter.create` refuses a run directory
that already exists.
"""
from __future__ import annotations

import contextlib
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

PYTHON = str(Path.home() / "miniconda3" / "envs" / "robo" / "bin" / "python")
REPO_ROOT = Path(__file__).resolve().parents[3]
READY_TIMEOUT_S = 30.0


def _free_port() -> int:
    """Ask the OS for an unused TCP port, then release it immediately.

    A tiny race (something else grabs it before the subprocess binds) is
    possible but unlikely enough that no test in this repo's suite bothers
    guarding against it further.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class MockApp:
    url: str
    run_dir: Path
    proc: subprocess.Popen
    log_path: Path

    def shutdown(self, timeout: float = 10.0) -> None:
        """Always terminates the subprocess — bounded, then a hard kill."""
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


def _wait_ready(url: str, proc: subprocess.Popen, log_path: Path,
                timeout: float) -> None:
    """Poll the HTTP port rather than parse stdout — `serve_ui` only starts
    once `start_mock`'s `RunWriter.create` has already made the run dir, so
    "the port answers" is also "the run directory exists"."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"mock app exited early (code {proc.returncode}):\n"
                f"{log_path.read_text() if log_path.exists() else '(no log)'}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(
        f"mock app never answered {url} within {timeout}s:\n"
        f"{log_path.read_text() if log_path.exists() else '(no log)'}")


def _launch(tmp_path: Path, collect: bool, auto: bool = False) -> MockApp:
    run_root = tmp_path / "run"
    port, bus_port = _free_port(), _free_port()
    # `?bus=` is how the frontend learns which bus port to dial
    # (`InspectionApp.tsx:busUrlFromQuery`) — it defaults to 8765 without
    # it, which is not the port this fixture just started. Mirrors
    # `ui/app.py:mock`'s own `if bus_port != 8765: url = f"{url}/?bus=..."`.
    url = f"http://127.0.0.1:{port}/?bus={bus_port}"
    log_path = tmp_path / ("mock-collect.log" if collect else "mock-live.log")

    cmd = [
        PYTHON, "-m", "inspection.tests.e2e._mock_launcher", "mock",
        f"--port={port}", f"--bus_port={bus_port}",
        f"--run_root={run_root}",
        "--no_window", "--open_browser=False",
        f"--collect={collect}", f"--auto={auto}",
    ]
    # Logs go to a real file, not a PIPE: a subprocess that runs for the
    # length of a whole test writes enough log lines to fill a pipe's OS
    # buffer and deadlock on it, and a file never blocks the writer.
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_file,
                                stderr=subprocess.STDOUT)

    _wait_ready(url, proc, log_path, READY_TIMEOUT_S)
    return MockApp(url=url, run_dir=run_root / "mock-run", proc=proc,
                   log_path=log_path)


@pytest.fixture
def mock_app_live(tmp_path):
    app = _launch(tmp_path, collect=False)
    try:
        yield app
    finally:
        app.shutdown()


@pytest.fixture
def mock_app_collect(tmp_path):
    app = _launch(tmp_path, collect=True)
    try:
        yield app
    finally:
        app.shutdown()


@pytest.fixture
def mock_app_collect_auto(tmp_path):
    app = _launch(tmp_path, collect=True, auto=True)
    try:
        yield app
    finally:
        app.shutdown()
