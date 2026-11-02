"""TDD for the `brain_main --service` flag gating (§2.4).

The service brain refuses to boot half-configured: Stage 1 drives the glide path on the GT
debug-pose channel over a baked map, so --service demands --mode glide + --debug-pose +
--map-dir up front (argparse error, exit 2) instead of dying later mid-connect. The full
live path is the §2.5 integration smoke. `brain` env.
"""

import sys
from pathlib import Path

import pytest

from humanoid.logic.oli import brain_main

_REPO_ROOT = Path(brain_main.__file__).resolve().parents[3]

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


def test_shadow_requires_service(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main_with(monkeypatch, "--shadow", "reference", "--mode", "glide")
    assert exc.value.code == 2
    assert "--service" in capsys.readouterr().err


def test_import_surface_stays_light():
    # The service brain boots in disposable bench-<candidate> envs carrying only
    # numpy/stdlib (locbench D8): importing brain_main must not drag in the walk stack
    # (onnxruntime/scipy) or UI bits (pygame). PolicyRunner imports lazily instead.
    import subprocess
    import sys

    code = (
        "import sys; import humanoid.logic.oli.brain_main; "
        "heavy = [m for m in sys.modules "
        "         if m.startswith(('onnxruntime', 'scipy', 'pygame', 'matplotlib'))]; "
        "assert not heavy, f'heavy imports at brain_main import time: {heavy}'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(_REPO_ROOT))
    assert r.returncode == 0, r.stderr
