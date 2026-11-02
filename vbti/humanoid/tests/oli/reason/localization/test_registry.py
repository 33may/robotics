"""TDD for the realization registry (reason/localization/registry.py) — locbench D7.

A candidate = `realizations/<name>/` (locdev-flow convention: `module.py` exposing
`build(config) -> LocalizationModule`, all tunables in `config.yaml`). The registry
lazy-imports by name — a realization's (heavy, env-specific) dependencies are NEVER
imported unless that name is selected, so the brain boots in any env that satisfies just
the chosen candidate. `config.yaml` is parsed and passed to `build`, with caller overrides
merged on top (how the bench injects bias/noise without editing the file). `brain` env.
"""

import sys

import pytest

import humanoid.logic.oli.reason.localization.registry as reg
from humanoid.logic.oli.reason.localization import LocalizationModule
from humanoid.logic.oli.reason.localization.registry import (
    list_realizations,
    load_realization,
)

pytestmark = pytest.mark.brain


def test_unknown_name_fails_loud_with_the_menu():
    with pytest.raises(ValueError, match="unknown realization"):
        load_realization("no-such-slam")


def test_template_is_hidden_from_the_menu():
    names = list_realizations()
    assert "_template" not in names   # scaffold source, not a candidate


def test_template_loads_via_the_registry():
    # `_template` is the conforming no-op — loadable explicitly (hidden ≠ forbidden).
    module = load_realization("_template")
    assert isinstance(module, LocalizationModule)


@pytest.fixture()
def fake_realizations(tmp_path, monkeypatch):
    root = tmp_path / "fake_realizations"
    root.mkdir()
    (root / "__init__.py").touch()
    monkeypatch.setattr(reg, "_REALIZATIONS_DIR", root)
    monkeypatch.setattr(reg, "_REALIZATIONS_PKG", "fake_realizations")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    for m in [m for m in sys.modules if m.startswith("fake_realizations")]:
        del sys.modules[m]


def _make(root, name, module_py, config_yaml=None):
    d = root / name
    d.mkdir()
    (d / "__init__.py").touch()
    (d / "module.py").write_text(module_py)
    if config_yaml is not None:
        (d / "config.yaml").write_text(config_yaml)


_FAKE = (
    "class M:\n"
    "    def __init__(self, config): self.config = config\n"
    "    def start(self, setup): pass\n"
    "    def step(self, loc_in): raise NotImplementedError\n"
    "    def stop(self): pass\n"
    "def build(config):\n"
    "    return M(config)\n"
)


def test_lazy_import_only_on_selection(fake_realizations):
    _make(fake_realizations, "boom", "raise ImportError('heavy dep missing')\n")
    assert "boom" in list_realizations()          # listed without importing
    with pytest.raises(ImportError, match="heavy dep"):
        load_realization("boom")                  # the import happens HERE, on selection


def test_config_yaml_parsed_and_overrides_win(fake_realizations):
    _make(fake_realizations, "fake", _FAKE,
          config_yaml="noise:\n  bias_m: 0.0\nfeatures: 400\n")
    m = load_realization("fake")
    assert m.config == {"noise": {"bias_m": 0.0}, "features": 400}
    m2 = load_realization("fake", overrides={"noise": {"bias_m": 0.2}})
    assert m2.config["noise"]["bias_m"] == 0.2    # override wins (deep-merged)
    assert m2.config["features"] == 400           # untouched keys survive


def test_no_config_yaml_means_empty_config(fake_realizations):
    _make(fake_realizations, "bare", _FAKE)
    assert load_realization("bare").config == {}


def test_module_without_build_fails_loud(fake_realizations):
    _make(fake_realizations, "nobuild", "class M:\n    pass\n")
    with pytest.raises(ValueError, match="build"):
        load_realization("nobuild")
