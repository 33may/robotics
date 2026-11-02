"""Migration gate — the ONE legal writer of a closed run (add-only).

A migration is registered, versioned, idempotent, recorded:
  - add-only: writes NEW files/fields; never modifies existing bytes
  - backfill-or-declared-missing: old runs get the value computed or an
    explicit marker — never silently absent
  - recorded: run.json `migrations` lists applied ids (the record update is
    itself additive: appending to a list field)
Always dry-run first over the archive; apply only after reviewing the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inspection.record.schema import RunRecord
from inspection.record.writer import _write_json


@dataclass
class Migration:
    id: str  # e.g. "m001-add-extra" — ordered, unique, immutable once shipped
    description: str
    applies: Callable[[Path], bool]  # does this run still need it?
    apply: Callable[[Path], list[str]]  # ADD-ONLY; returns created/extended files


def migrate_run(run_dir: Path, migrations: list[Migration],
                dry_run: bool = True) -> list[tuple[str, str]]:
    """Returns [(migration_id, "pending"|"applied"|"skipped"), ...]."""
    run_dir = Path(run_dir)
    run = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    report: list[tuple[str, str]] = []
    for m in migrations:
        if m.id in run.migrations or not m.applies(run_dir):
            report.append((m.id, "skipped"))
            continue
        if dry_run:
            report.append((m.id, "pending"))
            continue
        m.apply(run_dir)
        run.migrations.append(m.id)
        _write_json(run_dir / "run.json", run)
        report.append((m.id, "applied"))
    return report


def migrate_archive(root: Path, migrations: list[Migration],
                    dry_run: bool = True) -> dict[str, list[tuple[str, str]]]:
    return {d.name: migrate_run(d, migrations, dry_run=dry_run)
            for d in sorted(Path(root).iterdir())
            if d.is_dir() and (d / "run.json").exists()}
