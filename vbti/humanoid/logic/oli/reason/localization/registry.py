"""localization/registry.py — resolve a realization name to a live module (locbench D7).

`realizations/<name>/` (locdev-flow convention) holds `module.py` with
`build(config: dict) -> LocalizationModule` and every tunable in `config.yaml`. This registry
is the ONLY brain-side code that touches `realizations/` — and only lazily, on selection
(`--shadow <name>` / `--localizer <name>`): candidates carry heavy, mutually-incompatible
dependency stacks, so nothing may import them at module scope (architecture guard, locbench
task 11.3). `config.yaml` is parsed and handed to `build`, with caller `overrides`
deep-merged on top — how the bench injects bias/noise/dropout without editing the file.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional

from .module import LocalizationModule

_REALIZATIONS_DIR = Path(__file__).parent / "realizations"
_REALIZATIONS_PKG = __package__ + ".realizations"


def list_realizations() -> List[str]:
    """Candidate names (directories under realizations/), scaffold template hidden."""
    if not _REALIZATIONS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in _REALIZATIONS_DIR.iterdir()
        if d.is_dir() and (d / "module.py").exists() and not d.name.startswith("_")
    )


def load_realization(
    name: str,
    overrides: Optional[Dict] = None,
) -> LocalizationModule:
    """Import `realizations/<name>/module.py` NOW and build its module with its config."""
    folder = _REALIZATIONS_DIR / name
    if not (folder / "module.py").exists():
        menu = ", ".join(list_realizations()) or "(none yet)"
        raise ValueError(f"unknown realization {name!r} — available: {menu}")
    mod = importlib.import_module(f"{_REALIZATIONS_PKG}.{name}.module")
    build = getattr(mod, "build", None)
    if build is None:
        raise ValueError(
            f"realization {name!r} has no build(config) in module.py — "
            "the locdev scaffold provides one; see realizations/_template/module.py"
        )
    return build(_merge(_load_config(folder / "config.yaml"), overrides or {}))


def _load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    import yaml  # deferred: config parsing only happens on selection

    return yaml.safe_load(path.read_text()) or {}


def _merge(base: Dict, overrides: Dict) -> Dict:
    """Deep-merge `overrides` into `base` (dicts recurse, scalars replace)."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
