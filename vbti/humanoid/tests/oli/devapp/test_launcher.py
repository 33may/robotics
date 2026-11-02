"""Tests for the reusable ProcessLauncher (real short-lived child processes, no GUI)."""

import sys

import pytest

from humanoid.logic.oli.devapp.launcher import ProcessLauncher

pytestmark = pytest.mark.brain


def _sleeper(seconds: float) -> ProcessLauncher:
    return ProcessLauncher([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def test_starts_and_reports_running():
    lp = _sleeper(30)
    try:
        assert lp.is_running() is False
        assert lp.status() == "idle"
        assert lp.start() is True
        assert lp.is_running() is True
        assert "running" in lp.status()
    finally:
        lp.stop()
    assert lp.is_running() is False


def test_start_is_idempotent():
    lp = _sleeper(30)
    try:
        assert lp.start() is True
        assert lp.start() is False  # already running → no second process
    finally:
        lp.stop()


def test_stop_is_safe_before_start():
    ProcessLauncher([sys.executable, "-c", "pass"]).stop()  # must not raise


def test_returncode_after_natural_exit():
    lp = ProcessLauncher([sys.executable, "-c", "import sys; sys.exit(7)"])
    lp.start()
    lp._proc.wait(timeout=5)  # let it exit on its own
    assert lp.is_running() is False
    assert lp.returncode == 7
    assert "exited (code 7)" in lp.status()
