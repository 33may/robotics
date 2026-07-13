"""TDD for the `brain_main --service` flag gating (§2.4).

The service brain refuses to boot half-configured: Stage 1 drives the glide path on the GT
debug-pose channel over a baked map, so --service demands --mode glide + --debug-pose +
--map-dir up front (argparse error, exit 2) instead of dying later mid-connect. The full
live path is the §2.5 integration smoke. `brain` env.
"""

import sys

import pytest

from humanoid.logic.oli import brain_main

pytestmark = pytest.mark.brain


def _main_with(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["brain_main.py", *argv])
    brain_main.main()


def test_service_requires_glide_mode(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main_with(monkeypatch, "--service", "--mode", "walk",
                   "--debug-pose", "/tmp/p.sock", "--map-dir", "/tmp/m")
    assert exc.value.code == 2
    assert "--mode glide" in capsys.readouterr().err


def test_service_requires_debug_pose_and_map_dir(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main_with(monkeypatch, "--service", "--mode", "glide")
    assert exc.value.code == 2
    assert "--debug-pose" in capsys.readouterr().err
