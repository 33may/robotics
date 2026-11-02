"""derive — idempotent derivations with provenance (user stories D1-D5).

Helper library, not a framework: a derivation is a plain function
`fn(run_dir, out_dir, params) -> {step_id: "ok" | "failed: why"}`.
Addressing: derived/<run>/<method>[/<variant>]. Skip-if-fresh: recorded
source hashes + version vs current. Explicit per-step failure entries —
never a silent gap (the cup5 step-029 lesson).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable

from inspection.record.schema import DerivationMeta
from inspection.record.writer import _write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_derivation(run_dir: Path, derived_root: Path, *, method: str,
                   fn: Callable[[Path, Path, dict], dict[int, str]],
                   params: dict[str, Any], semver: str, code_sha: str,
                   sources: list[str], variant: str | None = None,
                   now: float | None = None) -> Path:
    """Run (or skip) one derivation for one run. Returns the output dir."""
    run_dir = Path(run_dir)
    out_dir = Path(derived_root) / run_dir.name / method
    if variant:
        out_dir = out_dir / variant
    version = f"{method}/{semver}+p:{code_sha[:8]}"
    current = {s: _sha256(run_dir / s) for s in sources}

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        prior = DerivationMeta.model_validate_json(meta_path.read_text())
        if prior.version == version and prior.source_hashes == current:
            return out_dir  # fresh — idempotent skip, no timestamped twins

    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = fn(run_dir, out_dir, params)  # deterministic overwrite
    _write_json(meta_path, DerivationMeta(
        method=method, variant=variant, version=version, params=params,
        source_hashes=current, steps=statuses,
        created_at=now if now is not None else time.time()))
    return out_dir
