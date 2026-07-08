"""launcher.py — a small reusable subprocess manager for the dev app.

`ProcessLauncher` starts / stops / polls a child process (the joystick teleop app now,
other tools later). It is deliberately generic — it knows nothing about joysticks — so
any panel can drive an external process with start/stop buttons and a status line.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional


class ProcessLauncher:
    """Manage one child process: start, stop, and report status. Idempotent and safe."""

    def __init__(self, cmd: List[str], cwd: Optional[str] = None, name: str = "process") -> None:
        self._cmd = list(cmd)
        self._cwd = cwd
        self.name = name
        self._proc: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        """True iff the child has been started and has not yet exited."""
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """Start the process if not already running. Returns True if it launched now."""
        if self.is_running():
            return False
        self._proc = subprocess.Popen(self._cmd, cwd=self._cwd)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Terminate the process if running (SIGTERM, then SIGKILL on timeout)."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=timeout)
        self._proc = None

    @property
    def returncode(self) -> Optional[int]:
        """Exit code of the last run (None if running or never started)."""
        return None if self._proc is None else self._proc.poll()

    def status(self) -> str:
        """One-line human status for a UI label."""
        if self.is_running():
            return f"running (pid {self._proc.pid})"  # type: ignore[union-attr]
        rc = self.returncode
        if rc is None:
            return "idle"
        return f"exited (code {rc})"
