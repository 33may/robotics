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
import threading
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


def git_sha() -> str:
    """HEAD of this checkout, or "unknown" — code identity for a record.

    Lives here because more than one record carries it (`ConfigSnapshot.git_sha`,
    `MenuDef.code_sha`) and a second implementation would eventually disagree
    with the first. Never raises: a run that cannot name its commit is worth
    recording anyway.
    """
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=Path(__file__).resolve().parents[2]
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — no git, no repo, no time: all "unknown"
        return "unknown"


def _write_json(path: Path, model: RecordModel) -> None:
    """Atomic, write-strict JSON write: validate -> temp -> rename.

    The temp name carries the THREAD id as well as the pid: run.json has more
    than one writer thread in a live run (the dispatcher flushes the steps
    roster while `brain/ask` sets the question from the command pump), and two
    threads sharing one temp path would have each other's half-written bytes
    renamed over the file. Same-directory temp keeps `os.replace` atomic.
    """
    data = model.model_dump(mode="json")
    validate_for_write(type(model), data)
    tmp = path.parent / f".tmp-{path.name}-{os.getpid()}-{threading.get_ident()}"
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
        self._check_open()
        self._manifest()
        self._run.status = status  # type: ignore[assignment]
        self._run.closed_at = now if now is not None else time.time()
        self._flush_run()
        self._closed = True

    # --- read-back ------------------------------------------------------------

    @property
    def question(self) -> str | None:
        """The run's question as it stands — None until (or unless) one is set.

        The writer already holds the run record, so a caller deciding on the
        close status (`run/app.py:finish_run`: a question-less run owes no
        answer) reads it from here rather than re-loading run.json from disk
        one line before overwriting it.
        """
        return self._run.question

    def run_meta(self) -> dict[str, Any]:
        """`source`/`name`/`object`/`question` — exactly `publish_run_meta`'s
        keyword arguments, so the retained UI topic cannot drift from the
        record it describes."""
        return {"source": self._run.source, "name": self._run.name,
                "object": self._run.object, "question": self._run.question}

    # --- steps ----------------------------------------------------------------

    def begin_step(self, view: dict | Any,
                   t_arrived: float | None = None) -> tuple[int, Path]:
        """Allocate the next dense step_id and its directory."""
        self._check_open()
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
        self._check_open()
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
        self._check_open()
        rec = self._records[step_id].model_copy(update={
            "phase": "fused",
            "geometry": geometry if isinstance(geometry, GeometryStats)
            else GeometryStats.model_validate(geometry),
        })
        self._write_step(rec)
        _write_json(self._step_dir(step_id) / "view_state.json", view_state)

    def mark_step(self, step_id: int, *, outcome: str, detail: str) -> None:
        """Explicit failure entry — never a silent gap in the id space."""
        self._check_open()
        pend = self._pending[step_id]
        rec = StepRecord(
            step_id=step_id, view=pend["view"], t_arrived=pend["t_arrived"],
            outcome=outcome, phase="captured", detail=detail,
        )
        self._write_step(rec)

    # --- late-arriving fields ---------------------------------------------------

    def set_view_method_params(self, method_id: str, **params) -> None:
        """Merge kwargs into run.view_methods[method_id].params; flush run.json.

        Survey-derived values (the viewsphere `r`) are only known after the
        run has already been created.
        """
        self._check_open()
        for vm in self._run.view_methods:
            if vm.id == method_id:
                vm.params = {**vm.params, **params}
                break
        else:
            raise KeyError(f"no view method '{method_id}' on this run")
        self._flush_run()

    def set_question(self, q: str) -> None:
        """Set run.question; flush run.json (Flow A's ask arrives after create)."""
        self._check_open()
        self._run.question = q
        self._flush_run()

    def write_session(self, session: SessionRecord) -> None:
        """Write session.json (rigs whose camera opens only after create)."""
        self._check_open()
        _write_json(self.dir / "session.json", session)

    # --- events ---------------------------------------------------------------

    def event(self, kind: str, step_id: int | None = None,
              detail: str | None = None, now: float | None = None) -> None:
        """Append one `OperatorEvent` line to events.jsonl.

        Safe from any thread, and from more than one at once: O_APPEND ("a")
        makes each single-line write atomic at these sizes, so a mover's
        approval beats on the driver thread cannot interleave with the
        dispatcher's stopped/fault rows mid-line.
        """
        self._check_open()
        ev = OperatorEvent(t=now if now is not None else time.time(),
                           kind=kind, step_id=step_id, detail=detail)
        with (self.dir / "events.jsonl").open("a") as f:
            f.write(ev.model_dump_json() + "\n")

    # --- internals ------------------------------------------------------------

    def _check_open(self) -> None:
        """Refuse every write verb once `close()` has run.

        A closed run has a manifest hashing every binary and a validated final
        run.json — a line that lands afterwards is a record its own receipt no
        longer describes. Raising (rather than dropping it quietly) is the
        point: the caller believes the write happened otherwise.
        """
        if self._closed:
            raise RuntimeError("run is closed")

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
