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

    verdict = None
    ans_path = run_dir / "answer.json"
    if ans_path.exists():
        try:
            verdict = json.loads(ans_path.read_text()).get("verdict")
        except Exception:
            pass

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
