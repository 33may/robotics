#!/usr/bin/env python3
"""Run structure for the image machinery — the facts tier.

Three trust levels (design 2026-08-20): deterministic code writes geometry
facts via FactWriter; agents get their own writers (Task 2) that CANNOT
touch views/visited/coverage. Flushed to `store.json` on every write.
Never imports motion/run — the image tier does not move the robot.

WHERE it flushes is the caller's to say, and in a recorded run that is the
orchestrator session's own directory (`ai/<seq>/`, from `AIRunWriter.dir`):
this is the AI tier's mutable state — plan, hypothesis, findings — and it
belongs to the session that produced it, not to the run. Views are no longer
here at all; they are steps (`record/run.py`).
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ViewRecord:
    cell: tuple | None
    pose_id: int
    cap_dir: str
    t: float
    T_base_cam: np.ndarray


class RunStore:
    """Owns the dict + disk flush. Read API only — writes go through writers."""

    def __init__(self, store_dir: Path, data: dict):
        self.path = Path(store_dir)
        self._d = data

    @classmethod
    def create(cls, store_dir, h_bins, v_elevs, r):
        d = {"grid": {"h_bins": h_bins, "v_elevs": list(v_elevs), "r": r},
             "views": [], "notes": [], "hypothesis": None, "plan": None,
             "findings": []}
        store = cls(store_dir, d)
        store._flush()
        return store

    @classmethod
    def open(cls, store_dir):
        p = Path(store_dir) / "store.json"
        return cls(store_dir, json.loads(p.read_text()))

    def _flush(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "store.json").write_text(
            json.dumps(self._d, indent=1) + "\n")

    def views(self, cell=None):
        out = []
        for v in self._d["views"]:
            c = tuple(v["cell"]) if v["cell"] is not None else None
            if cell is None or c == tuple(cell):
                out.append(ViewRecord(c, v["pose_id"], v["cap_dir"], v["t"],
                                      np.asarray(v["T_base_cam"], float)))
        return out

    def visited(self):
        return {v.cell for v in self.views() if v.cell is not None}

    def coverage(self):
        g = self._d["grid"]
        cov = np.zeros((len(g["v_elevs"]), g["h_bins"]), dtype=bool)
        for (h, v) in self.visited():
            cov[v, h] = True
        return cov

    def notes(self):
        return list(self._d["notes"])

    def findings(self):
        return list(self._d["findings"])

    @property
    def hypothesis(self):
        return self._d["hypothesis"]

    @property
    def plan(self):
        return self._d["plan"]


class FactWriter:
    """Deterministic-code tier: the ONLY writer of geometric ground truth."""

    def __init__(self, store: RunStore):
        self._s = store

    def add_view(self, cell, pose_id, cap_dir, T_base_cam, t):
        self._s._d["views"].append(
            {"cell": list(cell) if cell is not None else None,
             "pose_id": int(pose_id), "cap_dir": str(cap_dir),
             "t": float(t),
             "T_base_cam": np.asarray(T_base_cam, float).tolist()})
        self._s._flush()


class PlanWriter:
    """Orchestrator tier: reasoning state only. No geometry verbs exist here."""

    def __init__(self, store: RunStore):
        self._s = store

    def set_plan(self, text):
        self._s._d["plan"] = str(text); self._s._flush()

    def set_hypothesis(self, text):
        self._s._d["hypothesis"] = str(text); self._s._flush()

    def note(self, text, cell=None):
        _note(self._s, "plan", text, cell)


class FindingWriter:
    """Inspection-subagent tier: per-view findings + verbatim transcripts.

    `ai` is the session's `AIRunWriter` when there is one. Transcripts are
    `TranscriptRecord`s either way — same name (`transcripts/tNNN.json`), same
    shape, same id — because that is what the one read door reads
    (`record/run.py:AIRun.transcripts`) and what `validate_run` checks. A
    transcript the reader cannot see is a transcript nobody will ever read.
    """

    def __init__(self, store: RunStore, ai=None):
        self._s = store
        self._ai = ai

    @property
    def artifacts_dir(self) -> Path:
        """Where the images a subagent actually looked at are persisted —
        `ai/<seq>/artifacts/`, the home the schema gives them, so an answer's
        `evidence_images[].artifact` is a path under its own AI session."""
        d = (self._ai.dir if self._ai is not None else self._s.path) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_finding(self, cell, summary, record):
        """Persist one subagent transcript; return its store-relative path.

        `record` is a `TranscriptRecord`; the writer stamps its id.
        """
        from inspection.record.ai_writer import next_transcript_id
        # Two relativities, one boundary. The loop reports artifact paths
        # RUN-relative because that is what the trace and the UI's capture
        # mount speak (`ui/publisher.py:_asset_url`); the record declares them
        # relative to the AI session (`schema.py:TranscriptRecord.artifacts`).
        # They all live in `artifacts_dir`, so the session-relative form is
        # exactly that directory plus the file name.
        record.artifacts = [f"artifacts/{Path(a).name}" for a in record.artifacts]
        if self._ai is not None:
            path = self._ai.transcript(record)
        else:
            # No AI session (unit tests, bench tools): same record, same
            # naming, written beside the store. NOT a second format.
            from inspection.record.writer import _write_json
            tdir = self._s.path / "transcripts"
            tdir.mkdir(parents=True, exist_ok=True)
            record.transcript_id = next_transcript_id(tdir)
            path = tdir / f"{record.transcript_id}.json"
            _write_json(path, record)
        rel = f"transcripts/{path.name}"
        self._s._d["findings"].append(
            # None is the survey: its declaration is a finding like any other,
            # it just has no cell address (eyes/agents/survey_agent.py).
            {"cell": None if cell is None else list(cell),
             "summary": str(summary), "transcript": rel,
             "transcript_id": record.transcript_id,
             "t": time.time()})
        self._s._flush()
        return rel

    def note(self, text, cell=None):
        _note(self._s, "finding", text, cell)


def _note(store, who, text, cell):
    store._d["notes"].append(
        {"who": who, "cell": list(cell) if cell is not None else None,
         "text": str(text), "t": time.time()})
    store._flush()
