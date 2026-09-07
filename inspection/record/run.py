"""Run — the one read object over a recorded run (spec: flows-design.md §4).

Read-only and lazy: records validated eagerly (small JSON), binaries only on
call. Files are the truth; this object never writes a byte into the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from inspection.record.schema import (ConfigSnapshot, OperatorEvent, RunRecord,
                                      SessionRecord, StepRecord, ViewState)
from inspection.record.validate import Report, validate_run


class Step:
    def __init__(self, record: StepRecord, step_dir: Path):
        self.record, self.dir = record, step_dir
        self.id = record.step_id

    @property
    def T_base_cam(self) -> np.ndarray | None:
        t = self.record.T_base_cam
        return None if t is None else np.asarray(t, float)

    @property
    def view_state(self) -> ViewState | None:
        p = self.dir / "view_state.json"
        return ViewState.model_validate_json(p.read_text()) if p.exists() else None


class Run:
    def __init__(self, path: Path, record: RunRecord, steps: dict[int, Step],
                 provenance: str = "native"):
        self.path, self.record, self.provenance = path, record, provenance
        self._steps = steps

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        run_dir = Path(run_dir)
        rp = run_dir / "run.json"
        if not rp.exists():
            raise FileNotFoundError(rp)
        record = RunRecord.model_validate_json(rp.read_text())
        steps = {}
        for sid in record.steps:
            sp = run_dir / "steps" / f"{sid:03d}" / "step.json"
            if sp.exists():
                steps[sid] = Step(StepRecord.model_validate_json(sp.read_text()),
                                  sp.parent)
        return cls(run_dir, record, steps)

    # steps -----------------------------------------------------------------
    @property
    def steps(self) -> list[Step]:
        return [self._steps[i] for i in sorted(self._steps)]

    def step(self, step_id: int) -> Step:
        return self._steps[step_id]

    @property
    def survey(self) -> Step | None:
        return self._steps.get(0)

    @property
    def captured(self) -> list[Step]:
        return [s for s in self.steps if s.record.outcome == "captured"]

    def at(self, address) -> Step | None:
        addr = list(address)
        hits = [s for s in self.captured if s.record.view.address == addr]
        return hits[-1] if hits else None

    # run-level records -------------------------------------------------------
    @property
    def session(self) -> SessionRecord | None:
        p = self.path / "session.json"
        return SessionRecord.model_validate_json(p.read_text()) if p.exists() else None

    @property
    def config(self) -> ConfigSnapshot | None:
        p = self.path / "config.json"
        return ConfigSnapshot.model_validate_json(p.read_text()) if p.exists() else None

    @property
    def events(self) -> list[OperatorEvent]:
        p = self.path / "events.jsonl"
        if not p.exists():
            return []
        return [OperatorEvent.model_validate_json(line)
                for line in p.read_text().splitlines() if line.strip()]

    def fused(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Points + row-aligned colors (or None); new layout wins over legacy."""
        for cand in (self.path / "fused" / "cloud.npy", self.path / "fused_cloud.npy"):
            if cand.exists():
                pts = np.load(cand)
                colors = None
                cpath = cand.with_name(cand.name.replace("cloud", "colors"))
                if cpath.exists():
                    c = np.load(cpath)
                    if len(c) == len(pts):
                        colors = c
                return pts, colors
        return None

    def validate(self, deep: bool = False) -> Report:
        return validate_run(self.path, deep=deep)

    def derived(self, method: str, variant: str | None = None) -> Path | None:
        """Existing derivation dir, if any: derived/<id> sits beside runs/<id>.

        Probes the legacy `fits/<method>[-<variant>]` bucket first, then the
        flat `derive.py` layout `<method>[/<variant>]`.
        """
        droot = self.path.parent.parent / "derived" / self.record.id
        fits_name = f"{method}-{variant}" if variant else method
        flat = droot / method / variant if variant else droot / method
        for cand in (droot / "fits" / fits_name, flat):
            if cand.is_dir():
                return cand
        return None
