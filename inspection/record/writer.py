"""RunWriter — the single writer for recording runs (live loop + data engine).

Owns every contract JSON in a run directory; capture code keeps writing its
bytes (rgb/depth/ply) into the step dir it gets from `begin_step`. Protocol:

    w = RunWriter.create(root, run_id=..., name=..., source=..., config=...)
    sid, sdir = w.begin_step(view)          # dense ids, survey = 0
    w.write_capture(sid, ...)               # step.json phase="captured"
    w.write_fused(sid, geometry, vstate)    # phase="fused" + view_state.json
    w.mark_step(sid, outcome="rejected", detail=...)   # pose-less failure entry
    w.event("approved", step_id=sid)        # events.jsonl append
    w.close("completed")                    # manifest(binaries) + final run.json

Every JSON write: validate_for_write -> temp-in-same-dir -> os.replace().
The steps roster in run.json flushes at the FIRST write of each step —
a crash never leaves a step dir the roster doesn't know about.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from inspection.record.schema import (
    ConfigSnapshot,
    FileEntry,
    GeometryStats,
    Manifest,
    OperatorEvent,
    RecordModel,
    RunRecord,
    Segmentation,
    SessionRecord,
    StepRecord,
    ViewState,
    validate_for_write,
)

BINARY_EXTS = {".png", ".npy", ".ply", ".obj"}


def default_id(name: str, now: float | None = None) -> str:
    """Run id in the house convention: DDMM-name (e.g. 3108-camdemo5)."""
    tm = time.localtime(now if now is not None else time.time())
    return f"{tm.tm_mday:02d}{tm.tm_mon:02d}-{name}"


def _write_json(path: Path, model: RecordModel) -> None:
    """Atomic, write-strict JSON write: validate -> temp -> rename."""
    data = model.model_dump(mode="json")
    validate_for_write(type(model), data)
    tmp = path.parent / f".tmp-{path.name}-{os.getpid()}"
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class RunWriter:
    def __init__(self, run_dir: Path, run: RunRecord,
                 config: ConfigSnapshot, session: SessionRecord | None):
        self.dir = run_dir
        self._run = run
        self._records: dict[int, StepRecord] = {}
        self._pending: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._closed = False
        _write_json(run_dir / "run.json", run)
        _write_json(run_dir / "config.json", config)
        if session is not None:
            _write_json(run_dir / "session.json", session)

    # --- lifecycle ------------------------------------------------------------

    @classmethod
    def create(cls, root: Path, *, run_id: str, name: str,
               source: str, config: ConfigSnapshot,
               session: SessionRecord | None = None,
               rig: str = "real", object: str | None = None,
               object_instance: str | None = None,
               question: str | None = None,
               view_methods: list[dict | Any] = (),
               q_survey: list[float] | None = None,
               tags: list[str] = (), now: float | None = None) -> "RunWriter":
        run_dir = Path(root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)  # evidence is never overwritten
        run = RunRecord(
            id=run_id, name=name, object=object,
            object_instance=object_instance, source=source, rig=rig,
            tags=list(tags), question=question, status="running",
            created_at=now if now is not None else time.time(),
            view_methods=list(view_methods), q_survey=q_survey,
        )
        return cls(run_dir, run, config, session)

    def close(self, status: str, now: float | None = None) -> None:
        """Manifest over binaries, final run.json. Valid-at-close or marked."""
        self._manifest()
        self._run.status = status  # type: ignore[assignment]
        self._run.closed_at = now if now is not None else time.time()
        self._flush_run()
        self._closed = True

    # --- steps ----------------------------------------------------------------

    def begin_step(self, view: dict | Any,
                   t_arrived: float | None = None) -> tuple[int, Path]:
        """Allocate the next dense step_id and its directory."""
        step_id = self._next_id
        self._next_id += 1
        sdir = self.dir / "steps" / f"{step_id:03d}"
        sdir.mkdir(parents=True, exist_ok=False)
        self._pending[step_id] = {
            "view": view,
            "t_arrived": t_arrived if t_arrived is not None else time.time(),
        }
        return step_id, sdir

    def write_capture(self, step_id: int, *, t_captured: float,
                      joints_rad: list[float], T_base_flange: list,
                      T_base_cam: list, rgb_rotation_deg: int,
                      segmentation: Segmentation | dict | None = None) -> None:
        pend = self._pending[step_id]
        rec = StepRecord(
            step_id=step_id, view=pend["view"], t_arrived=pend["t_arrived"],
            outcome="captured", phase="captured", t_captured=t_captured,
            joints_rad=joints_rad, T_base_flange=T_base_flange,
            T_base_cam=T_base_cam, rgb_rotation_deg=rgb_rotation_deg,
            segmentation=segmentation,
        )
        self._write_step(rec)

    def write_fused(self, step_id: int, geometry: GeometryStats | dict,
                    view_state: ViewState) -> None:
        rec = self._records[step_id].model_copy(update={
            "phase": "fused",
            "geometry": geometry if isinstance(geometry, GeometryStats)
            else GeometryStats.model_validate(geometry),
        })
        self._write_step(rec)
        _write_json(self._step_dir(step_id) / "view_state.json", view_state)

    def mark_step(self, step_id: int, *, outcome: str, detail: str) -> None:
        """Explicit failure entry — never a silent gap in the id space."""
        pend = self._pending[step_id]
        rec = StepRecord(
            step_id=step_id, view=pend["view"], t_arrived=pend["t_arrived"],
            outcome=outcome, phase="captured", detail=detail,
        )
        self._write_step(rec)

    # --- events ---------------------------------------------------------------

    def event(self, kind: str, step_id: int | None = None,
              detail: str | None = None, now: float | None = None) -> None:
        ev = OperatorEvent(t=now if now is not None else time.time(),
                           kind=kind, step_id=step_id, detail=detail)
        with (self.dir / "events.jsonl").open("a") as f:
            f.write(ev.model_dump_json() + "\n")

    # --- internals ------------------------------------------------------------

    def _step_dir(self, step_id: int) -> Path:
        return self.dir / "steps" / f"{step_id:03d}"

    def _write_step(self, rec: StepRecord) -> None:
        _write_json(self._step_dir(rec.step_id) / "step.json", rec)
        self._records[rec.step_id] = rec
        if rec.step_id not in self._run.steps:
            self._run.steps.append(rec.step_id)
        self._flush_run()  # roster flushed at first step write — crash-safe

    def _flush_run(self) -> None:
        _write_json(self.dir / "run.json", self._run)

    def _manifest(self) -> None:
        files: dict[str, FileEntry] = {}
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.suffix in BINARY_EXTS:
                st = p.stat()
                files[p.relative_to(self.dir).as_posix()] = FileEntry(
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    bytes=st.st_size, mtime=st.st_mtime,
                )
        _write_json(self.dir / "manifest.json", Manifest(files=files))
