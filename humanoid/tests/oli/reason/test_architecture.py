"""ARCHITECTURE CONFORMANCE GUARD — fails loudly if the reason-module design drifts.

This file is enforcement, not examples. It pins the decisions of change
`may-173-reason-module-separation` (openspec design.md D1–D14) so a future session cannot
erode them silently:

  1. every seam realization actually satisfies its Protocol (static canary + runtime check);
  2. the import DIRECTION between packages: nav → localization, nav → mapping, and
     localization ⊥ mapping (no edge either way);
  3. the invariance boundary: none of the three packages pulls isaacsim/limxsdk into the
     process, even transitively;
  4. modules consume emitted data contracts — Planner must never hold a MappingModule ref.

If you are an agent about to change these packages: read `logic/oli/reason/AGENTS.md` first.
A red test here means your change breaks the architecture, not that the test is stale.
Pure: runs in `brain`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from humanoid.logic.oli.reason import localization as loc_pkg
from humanoid.logic.oli.reason import mapping as map_pkg
from humanoid.logic.oli.reason import nav as nav_pkg
from humanoid.logic.oli.reason.localization import (
    DebugPoseLocalizer,
    GroundTruthLocalizer,
    LocalizationModule,
    Localizer,
)
from humanoid.logic.oli.reason.mapping import MappingModule, OccupancyGrid, StaticMapping
from humanoid.logic.oli.reason.nav.planner import Planner

pytestmark = pytest.mark.brain


class _FakePoseClient:
    def latest(self):
        return None


# ── 1. Protocol conformance — the "implements" lines of this codebase ────────────────
# Static canaries: Pyright checks these ASSIGNMENTS at edit time — if a realization's shape
# drifts from its Protocol, the error appears HERE, at declaration, Java-style.
_static_localizer_gt: Localizer = GroundTruthLocalizer(pose_reader=lambda: None)
_static_localizer_dbg: Localizer = DebugPoseLocalizer(_FakePoseClient())
_static_mapping: MappingModule = StaticMapping.from_grid(
    OccupancyGrid(np.zeros((2, 2), dtype=bool), 1.0)
)


def test_every_localizer_realization_satisfies_the_seam():
    assert isinstance(GroundTruthLocalizer(pose_reader=lambda: None), Localizer)
    assert isinstance(DebugPoseLocalizer(_FakePoseClient()), Localizer)


def test_static_mapping_satisfies_mapping_module():
    grid = OccupancyGrid(np.zeros((2, 2), dtype=bool), 1.0)
    assert isinstance(StaticMapping.from_grid(grid), MappingModule)


def test_protocols_reject_shapeless_objects():
    assert not isinstance(object(), Localizer)
    assert not isinstance(object(), MappingModule)
    assert not isinstance(object(), LocalizationModule)


# ── 2. Import direction between the reason packages (design.md D1) ───────────────────

_IMPORT_RULES = {
    # package → substrings its source may NEVER contain in an import line
    "localization": ["reason.mapping", "reason.nav", "..mapping", "..nav", ".nav "],
    "mapping": ["reason.localization", "reason.nav", "..localization", "..nav"],
    # nav MAY import localization + mapping — that direction is the design.
}

_IMPORT_LINE = re.compile(r"^\s*(from|import)\s+(.+)$")


def _package_import_lines(pkg):
    root = Path(pkg.__file__).parent
    for py in sorted(root.glob("*.py")):
        for n, line in enumerate(py.read_text().splitlines(), 1):
            m = _IMPORT_LINE.match(line)
            if m:
                yield f"{py.name}:{n}", m.group(2)


@pytest.mark.parametrize("pkg,forbidden", [
    (loc_pkg, _IMPORT_RULES["localization"]),
    (map_pkg, _IMPORT_RULES["mapping"]),
], ids=["localization", "mapping"])
def test_package_never_imports_its_siblings(pkg, forbidden):
    violations = [
        f"{where}: {stmt}"
        for where, stmt in _package_import_lines(pkg)
        for bad in forbidden
        if bad in stmt
    ]
    assert not violations, (
        "IMPORT DIRECTION VIOLATION (design.md D1: nav→localization, nav→mapping, "
        f"localization⊥mapping):\n" + "\n".join(violations)
    )


# ── 3. Invariance boundary (repo golden rule) ─────────────────────────────────────────

def test_reason_packages_never_touch_world_sdks():
    # source-level: no import line mentions the world SDKs
    for pkg in (loc_pkg, map_pkg, nav_pkg):
        for where, stmt in _package_import_lines(pkg):
            assert "isaacsim" not in stmt and "limxsdk" not in stmt, (
                f"INVARIANCE VIOLATION at {where}: {stmt}"
            )
    # process-level: importing all three packages leaked no world SDK module
    leaks = [m for m in sys.modules if m.startswith(("isaacsim", "isaacsim.", "limxsdk"))]
    assert not leaks, f"INVARIANCE VIOLATION — world SDKs in sys.modules: {leaks}"


# ── 3b. The service seam stays brain-pure (locbench design.md D5) ─────────────────────

def test_service_seam_is_brain_pure():
    # `oli/service/` is brain-side code: it may import reason contracts, but never a world
    # SDK and never devapp (the seam exists precisely so clients DON'T live in the brain).
    from humanoid.logic.oli import service as service_pkg

    for where, stmt in _package_import_lines(service_pkg):
        for bad in ("isaacsim", "limxsdk", "devapp"):
            assert bad not in stmt, (
                f"SERVICE SEAM VIOLATION at {where}: {stmt} — oli/service must stay "
                "brain-pure (no world SDKs, no devapp)"
            )


# ── 4. Modules consume emitted contracts, not each other (design.md D8) ──────────────

def test_planner_holds_no_mapping_module_ref():
    # Planner's constructor takes ONLY robot/policy knobs; the world arrives per call as a
    # Map VALUE. If someone adds a mapping ref, this signature check catches it.
    import inspect

    params = inspect.signature(Planner.__init__).parameters
    assert set(params) == {
        "self", "robot_radius_m", "inflation_radius_m", "clearance_weight",
        "heuristic_weight", "horizon_m",
    }, (
        "Planner.__init__ grew a parameter — if it is a module ref (mapping/localizer), "
        "that violates design.md D8: Planner consumes the emitted Map value only."
    )


# ── locbench D6/D7: host seam realizations + the no-realization-import rule ──────────

def test_host_localizer_satisfies_the_localizer_seam():
    from humanoid.logic.oli.reason.localization import HostLocalizer, LocalizationHost

    class _NoFrames:
        def read(self, name):
            return None

        def stream_names(self):
            return []

    host = LocalizationHost(lambda: None, _NoFrames())
    _static_host_localizer: Localizer = HostLocalizer(host)   # declaration-site canary
    assert isinstance(_static_host_localizer, Localizer)


def test_no_brain_code_imports_realizations():
    # Candidates carry heavy, env-specific deps: ONLY the registry may reach them, and only
    # lazily (importlib by name). A static `realizations` import anywhere under logic/oli/
    # would drag a candidate's stack into every brain boot. (Scoped to logic/oli/ sources —
    # tests and locbench import pure realizations deliberately.)
    root = Path(loc_pkg.__file__).parents[2]     # logic/oli/
    violations = []
    for py in sorted(root.rglob("*.py")):
        if "realizations" in py.parts:
            continue                              # realizations may import themselves
        for n, line in enumerate(py.read_text().splitlines(), 1):
            m = _IMPORT_LINE.match(line)
            if not m:
                continue
            # judge the MODULE PATH only ("from X import Y" → X) — imported names like
            # `list_realizations` are fine, importing the realizations package is not
            module_path = m.group(2).split(" import ")[0]
            if "realizations" in module_path.replace(".", " ").split():
                violations.append(f"{py.relative_to(root)}:{n}: {m.group(2)}")
    assert not violations, (
        "REALIZATION IMPORT VIOLATION (locbench D7 — lazy registry only):\n"
        + "\n".join(violations)
    )
