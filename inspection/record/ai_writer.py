"""AIRunWriter — the brain thread's own pen, `ai/<seq:03d>/` subtree only.

Owns every contract JSON under one orchestrator session: airun.json,
transcripts/tNNN.json, menu/menu_def.json, menu/inputs.jsonl, answer.json.
Layout is the read-side contract fixed by `record/run.py`'s `AIRun` (task-4):

    a = AIRunWriter.create(run_dir, orchestrator_model=..., menu_id=..., menu_hash=...)
    a.transcript(rec)          # transcripts/tNNN.json
    a.menu_def(m)               # menu/menu_def.json
    a.menu_input(mi)            # menu/inputs.jsonl append
    a.answer({...})             # answer.json
    a.usage(output_tokens=4231) # merge into airun.json usage
    a.event("approved", step_id=3)   # SHARED <run>/events.jsonl, same as RunWriter.event

Every JSON write goes through the same `_write_json` (validate -> temp ->
os.replace) that `RunWriter` uses. `event()` appends to the run-level shared
file with the identical open(..., "a") O_APPEND pattern — one events.jsonl
per run, written by both writers, never torn.
"""
from __future__ import annotations

import time
from pathlib import Path

from inspection.record.schema import (
    AIRunRecord,
    AnswerRecord,
    MenuDef,
    MenuInput,
    OperatorEvent,
    TranscriptRecord,
    validate_for_write,
)
from inspection.record.writer import _write_json


def next_transcript_id(tdir: Path) -> str:
    """Next dense transcript id in a transcripts dir: t000, t001, ...

    The naming is the read door's (`record/run.py:AIRun.transcripts` globs
    `t*.json`), so it lives beside the writer that owns it and is shared with
    every producer — nobody invents a second convention.
    """
    n = len(list(Path(tdir).glob("t*.json"))) if Path(tdir).is_dir() else 0
    return f"t{n:03d}"


def next_seq(run_dir: Path) -> int:
    """Next unused ai/ sequence number: max existing + 1, 0 when none."""
    root = Path(run_dir) / "ai"
    if not root.is_dir():
        return 0
    seqs = [int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(seqs) + 1 if seqs else 0


class AIRunWriter:
    def __init__(self, run_dir: Path, ai_dir: Path, record: AIRunRecord):
        self.run_dir = Path(run_dir)
        self._dir = Path(ai_dir)
        self._record = record

    @classmethod
    def create(cls, run_dir: Path, *, seq: int = 0, mode: str = "live",
               orchestrator_model: str, menu_id: str, menu_hash: str,
               opening_prompt: str | None = None) -> "AIRunWriter":
        run_dir = Path(run_dir)
        ai_dir = run_dir / "ai" / f"{seq:03d}"
        ai_dir.mkdir(parents=True, exist_ok=False)  # a session's evidence is never overwritten
        (ai_dir / "transcripts").mkdir()
        (ai_dir / "menu").mkdir()
        rec = AIRunRecord(seq=seq, mode=mode, orchestrator_model=orchestrator_model,
                          menu_id=menu_id, menu_hash=menu_hash,
                          opening_prompt=opening_prompt)
        w = cls(run_dir, ai_dir, rec)
        w._flush()
        return w

    @property
    def dir(self) -> Path:
        return self._dir

    # --- transcripts / menu / answer -------------------------------------------

    def transcript(self, rec: TranscriptRecord) -> Path:
        """transcripts/t{n:03d}.json — n is a collision-proof directory count,
        never caller-picked (mirrors RunWriter.begin_step's dense ids).

        The id is STAMPED onto the record here, for the same reason: the
        filename and `transcript_id` are one fact, only this method knows the
        number, and an `AnswerRecord.evidence_images[].transcript_id` that
        does not name a file on disk is a citation to nothing.
        """
        tdir = self._dir / "transcripts"
        rec.transcript_id = next_transcript_id(tdir)
        path = tdir / f"{rec.transcript_id}.json"
        _write_json(path, rec)
        return path

    def menu_def(self, m: MenuDef) -> None:
        """menu/menu_def.json — snapshot written once per menu version."""
        _write_json(self._dir / "menu" / "menu_def.json", m)

    def menu_input(self, mi: MenuInput) -> None:
        """menu/inputs.jsonl append — one MenuInput per line, O_APPEND."""
        with (self._dir / "menu" / "inputs.jsonl").open("a") as f:
            f.write(mi.model_dump_json() + "\n")

    def answer(self, a: AnswerRecord | dict) -> None:
        """answer.json — accepts a dict and validates it into AnswerRecord."""
        rec = a if isinstance(a, AnswerRecord) else validate_for_write(AnswerRecord, a)
        _write_json(self._dir / "answer.json", rec)

    def usage(self, **fields) -> None:
        """Merge kwargs into airun.json usage (tokens/cost/duration); flush."""
        self._record.usage = {**(self._record.usage or {}), **fields}
        self._flush()

    # --- events (SHARED with RunWriter) -----------------------------------------

    def event(self, kind: str, step_id: int | None = None,
              detail: str | None = None, now: float | None = None) -> None:
        ev = OperatorEvent(t=now if now is not None else time.time(),
                           kind=kind, step_id=step_id, detail=detail)
        with (self.run_dir / "events.jsonl").open("a") as f:
            f.write(ev.model_dump_json() + "\n")

    # --- internals ---------------------------------------------------------------

    def _flush(self) -> None:
        _write_json(self._dir / "airun.json", self._record)
