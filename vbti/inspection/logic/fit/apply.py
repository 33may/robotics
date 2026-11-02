#!/usr/bin/env python3
"""Apply a fitting algorithm to every step of a run — one compute+write path.

The step's `decomposition.json` is a single polymorphic slot (Anton,
2026-09-14): the file name is fixed, the algorithm declares itself in the
`method` field inside, and consumers — rendering, real-time compute, view
derivation — dispatch on that name. This module owns the name->fitter
registry; the live loop and this backfill CLI read the SAME registry and
call the SAME writer (`record/writer.py:write_step_decomposition`), so a
backfilled dataset and a live run are byte-compatible by construction.

Input per step is the recorded accumulated cloud `steps/NNN/fused/cloud.npy`
(StepRecord v2) — exactly what the loop held at that step. v1 steps without
the pair are reported missing, never guessed.

    p vbti/inspection/logic/fit/apply.py run 1009-cam1
    p vbti/inspection/logic/fit/apply.py run 1009-cup1 --method boxguard --overwrite
    p vbti/inspection/logic/fit/apply.py sweep 1009-cam1 1009-lan1 1009-lan2 1009-lan3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from inspection.fit.box import boxguard_fit
from inspection.fit.contract import Decomposition
from inspection.record.run import Run
from inspection.record.schema import DecompositionRecord
from inspection.record.writer import write_step_decomposition

DATA = Path(__file__).resolve().parents[2] / "data"

#: name -> pure fitter `(N,3) points -> Decomposition`. The ONLY place an
#: algorithm's name is bound to code; live loop and backfill both read it.
METHODS = {
    "boxguard": boxguard_fit,
}

SOURCE = "fused/cloud.npy"


def _run_dir(run: str) -> Path:
    for root in (DATA / "datasets", DATA / "runs"):
        d = root / run
        if (d / "run.json").exists():
            return d
    raise SystemExit(f"no run.json for {run!r} in {DATA}/datasets or {DATA}/runs")


def record_of(step_id: int, n: int, d: Decomposition) -> DecompositionRecord:
    """Fit `Decomposition` -> the on-disk `DecompositionRecord`, in ONE place.

    Public on purpose: the live loop (`run/machine.py`) and this backfill
    both build their record through this exact function, so the two paths
    cannot drift a field apart — byte-compatible by construction.
    """
    return DecompositionRecord(
        step_id=step_id, source=SOURCE, n_points=n,
        method=d.method, method_version=d.version,
        abstain=d.abstain, reason=d.reason,
        primitives=[p.to_json() for p in d.primitives])


def apply_run(run_dir: Path, method: str = "boxguard",
              overwrite: bool = False) -> dict[str, int]:
    """Fit every fused step's recorded cloud; write `decomposition.json`.

    Idempotent: an existing slot is skipped unless `overwrite` (re-running
    with a new method/version deliberately replaces only this file — its
    own product; capture bytes are never touched). A fitter exception is
    recorded as an abstain with the error as reason — LOUDLY printed, one
    bad step must not kill a 30-step backfill but must never pass silent.
    """
    fitter = METHODS[method]
    run = Run.load(run_dir)
    stats = {"fitted": 0, "abstained": 0, "skipped": 0, "missing": 0}
    for step in run.steps:
        if step.record.phase != "fused":
            continue                     # rejected/unfused steps own no slot
        out = step.dir / "decomposition.json"
        if out.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        cp = step.dir / SOURCE
        if not cp.exists():
            stats["missing"] += 1
            continue
        P = np.load(cp)
        try:
            d = fitter(P)
        except Exception as e:  # noqa: BLE001 — recorded, screamed, counted
            print(f"!! {run_dir.name} step {step.id:03d}: {method} raised "
                  f"{type(e).__name__}: {e}")
            d = Decomposition(method=method, version="error", abstain=True,
                              reason=f"fitter error: {type(e).__name__}: {e}")
        if d.method != method:
            raise SystemExit(f"registry name {method!r} != produced "
                             f"Decomposition.method {d.method!r}")
        write_step_decomposition(run_dir, record_of(step.id, len(P), d))
        stats["abstained" if d.abstain else "fitted"] += 1
    return stats


# ---------------------------------------------------------------- CLI
def run(run_id: str, method: str = "boxguard", overwrite: bool = False):
    """Backfill one run/dataset; prints the per-step outcome tally."""
    d = _run_dir(str(run_id))
    stats = apply_run(d, method=method, overwrite=overwrite)
    print(f"{d.name} [{method}]: " +
          "  ".join(f"{k} {v}" for k, v in stats.items()))


def sweep(*run_ids: str, method: str = "boxguard", overwrite: bool = False):
    """Backfill several runs in one call."""
    for rid in run_ids:
        run(str(rid), method=method, overwrite=overwrite)


def methods():
    """List registered fitting methods."""
    for name in METHODS:
        print(name)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": run, "sweep": sweep, "methods": methods})
