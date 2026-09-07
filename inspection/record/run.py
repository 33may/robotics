"""Run — the one read object over a recorded run (spec: flows-design.md §4).

Read-only and lazy: records validated eagerly (small JSON), binaries only on
call. Files are the truth; this object never writes a byte into the run.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inspection.cell.geometry import rotate180
from inspection.record.legacy import adapt_run, is_legacy
from inspection.record.schema import (AIRunRecord, AnswerRecord, ConfigSnapshot,
                                      MenuDef, MenuInput, OperatorEvent, RunRecord,
                                      SessionRecord, StepRecord, TranscriptRecord,
                                      ViewState)
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

    # binaries ----------------------------------------------------------------
    # Convention (schema.py:Conventions.rotated_artifacts + capture.py:45-59,
    # Anton 2026-08-24): rgb.png, depth_aligned.npy AND mask.png are all
    # stored UPRIGHT on disk while the pose (T_base_cam etc.) stays RAW —
    # capture.py rotates all three 180° at write time when the viewsphere
    # roll is a half turn, so every reader gets an upright frame for free.
    # `upright=True` (default) therefore returns the file as stored;
    # `upright=False` rotates it back to the pose's RAW orientation — the
    # shape reconstruction path needs (`step_replay.py`'s "Frames" note).
    # One rotation policy, applied identically to all three artifacts here.
    def _to_pose_frame(self, arr: np.ndarray, upright: bool) -> np.ndarray:
        if not upright and self.record.rgb_rotation_deg == 180:
            return rotate180(arr)
        return arr

    def rgb(self, upright: bool = True) -> np.ndarray | None:
        """RGB order, from rgb.png."""
        p = self.dir / "rgb.png"
        if not p.exists():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._to_pose_frame(img, upright)

    def depth(self, upright: bool = True) -> np.ndarray | None:
        p = self.dir / "depth_aligned.npy"
        if not p.exists():
            return None
        return self._to_pose_frame(np.load(p), upright)

    def cloud(self) -> np.ndarray | None:
        p = self.dir / "cloud.ply"
        if not p.exists():
            return None
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(str(p)).points)

    def mask(self, upright: bool = True) -> np.ndarray | None:
        p = self.dir / "mask.png"
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return self._to_pose_frame(img > 0, upright)


class AIRun:
    """One orchestrator session: ai/<seq:03d>/ (spec: task-4 brief, binding
    layout — airun.json, transcripts/tNNN.json, menu/menu_def.json,
    menu/inputs.jsonl, answer.json, trace.jsonl)."""

    def __init__(self, record: AIRunRecord, ai_dir: Path):
        self.record, self.dir = record, ai_dir
        self.seq = record.seq

    @property
    def transcripts(self) -> list[TranscriptRecord]:
        tdir = self.dir / "transcripts"
        if not tdir.is_dir():
            return []
        recs = [TranscriptRecord.model_validate_json(p.read_text())
                for p in sorted(tdir.glob("t*.json"))]
        return sorted(recs, key=lambda r: r.t)

    @property
    def answer(self) -> AnswerRecord | None:
        p = self.dir / "answer.json"
        return AnswerRecord.model_validate_json(p.read_text()) if p.exists() else None

    @property
    def menu(self) -> MenuDef | None:
        p = self.dir / "menu" / "menu_def.json"
        return MenuDef.model_validate_json(p.read_text()) if p.exists() else None

    @property
    def menu_inputs(self) -> list[MenuInput]:
        p = self.dir / "menu" / "inputs.jsonl"
        if not p.exists():
            return []
        return [MenuInput.model_validate_json(line)
                for line in p.read_text().splitlines() if line.strip()]

    @property
    def trace_path(self) -> Path:
        return self.dir / "trace.jsonl"


class Run:
    def __init__(self, path: Path, record: RunRecord, steps: dict[int, Step],
                 provenance: str = "native"):
        self.path, self.record, self.provenance = path, record, provenance
        self._steps = steps

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        run_dir = Path(run_dir)
        record, steps, provenance = cls._read(run_dir)
        return cls(run_dir, record, steps, provenance=provenance)

    @staticmethod
    def _read(run_dir: Path) -> tuple[RunRecord, dict[int, "Step"], str]:
        """Load body shared by `load` and `refresh` — one door, native or legacy."""
        run_dir = Path(run_dir)
        if is_legacy(run_dir):
            a = adapt_run(run_dir)
            steps = {sid: Step(rec, run_dir / f"{sid:03d}")
                     for sid, rec in a.steps.items()}
            return a.run, steps, "legacy"
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
        return record, steps, "native"

    def refresh(self) -> "Run":
        """Re-read run.json + steps in place (mid-run readers); returns self.

        No lazy caches to drop today: session/config/events/view_state are
        all read fresh from disk on every call already.
        """
        self.record, self._steps, self.provenance = self._read(self.path)
        return self

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

    @property
    def ai(self) -> list[AIRun]:
        """AIRuns under ai/*/airun.json, sorted by seq; [] when absent (also
        legacy — legacy AI stays reachable via record/story.py, not here)."""
        root = self.path / "ai"
        if not root.is_dir():
            return []
        runs = []
        for d in sorted(root.iterdir()):
            p = d / "airun.json"
            if d.is_dir() and p.exists():
                runs.append(AIRun(AIRunRecord.model_validate_json(p.read_text()), d))
        return sorted(runs, key=lambda a: a.seq)

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
