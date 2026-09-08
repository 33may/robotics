"""catalog — cards and filters over run.json files. Files are the truth.

Pure-python scan (fast at lab scale, zero deps); the escalation path when it
gets slow is DuckDB over the same files, then a rebuildable SQLite — never a
server DB. Read-only: never mutates a run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from inspection.record.schema import Manifest, RunRecord
from inspection.record.validate import STALE_S

# The archive is two sibling folders, split by what a run IS (Anton
# 2026-09-08): `runs/` holds real inspection runs — a question asked, a
# verdict reached — and `datasets/` holds data-collection sweeps (Flow B,
# source="data-engine"), which are inputs to training, not inspections.
# The run.json `source` field says the same thing from inside the run; the
# folder says it from the outside, so `ls` can list one world at a time.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RUNS_ROOT = DATA_DIR / "runs"
DATASETS_ROOT = DATA_DIR / "datasets"


@dataclass
class Card:
    id: str
    name: str
    object: str | None
    object_instance: str | None
    source: str
    rig: str
    tags: list[str]
    status: str
    effective_status: str
    question: str | None
    verdict: str | None
    n_steps: int
    size_bytes: int
    created_at: float
    path: Path


def _card(run_dir: Path, stale_s: float, now: float) -> Card | None:
    try:
        run = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    except Exception:
        try:  # pre-schema layout -> adapt-on-read
            from inspection.record.legacy import adapt_run, is_legacy
            if not is_legacy(run_dir):
                return None
            run = adapt_run(run_dir).run
        except Exception:
            return None

    effective = run.status
    if run.status == "running":
        newest = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()),
                     default=0.0)
        if now - newest > stale_s:
            effective = "crashed"

    size = 0
    man_path = run_dir / "manifest.json"
    if man_path.exists():
        try:
            man = Manifest.model_validate_json(man_path.read_text())
            size = sum(e.bytes for e in man.files.values())
        except Exception:
            pass
    if size == 0:  # no/empty manifest (legacy, running) -> sum the tree
        size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())

    verdict = None
    for ans_path in (run_dir / "answer.json", run_dir / "eyes" / "answer.json"):
        if ans_path.exists():
            try:
                verdict = json.loads(ans_path.read_text()).get("verdict")
            except Exception:
                pass
            break

    return Card(id=run.id, name=run.name, object=run.object,
                object_instance=run.object_instance, source=run.source,
                rig=run.rig, tags=run.tags, status=run.status,
                effective_status=effective, question=run.question,
                verdict=verdict, n_steps=len(run.steps), size_bytes=size,
                created_at=run.created_at, path=run_dir)


def runs(root: Path, *, object: str | None = None, source: str | None = None,
         status: str | None = None, rig: str | None = None,
         tag: str | None = None, stale_s: float = STALE_S,
         now: float | None = None) -> list[Card]:
    now = now if now is not None else time.time()
    cards = []
    if not Path(root).is_dir():  # a root that has seen no runs yet is empty
        return cards
    for d in sorted(Path(root).iterdir()):
        if not (d.is_dir() and (d / "run.json").exists()):
            continue
        c = _card(d, stale_s, now)
        if c is None:
            continue
        if object is not None and c.object != object:
            continue
        if source is not None and c.source != source:
            continue
        if status is not None and c.status != status:
            continue
        if rig is not None and c.rig != rig:
            continue
        if tag is not None and tag not in c.tags:
            continue
        cards.append(c)
    return sorted(cards, key=lambda c: c.created_at)


def card(root: Path, run_id: str, stale_s: float = STALE_S,
         now: float | None = None) -> Card:
    now = now if now is not None else time.time()
    c = _card(Path(root) / run_id, stale_s, now)
    if c is None:
        raise FileNotFoundError(f"no valid run.json under {root}/{run_id}")
    return c
