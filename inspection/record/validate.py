"""validate_run — integrity engine for recorded runs (user stories C2/F1).

Per-file schema validation + cross-file reference checks + manifest
verification + the detect-on-read crash rule. Read paths never mutate runs.

    rep = validate_run(run_dir)            # cheap: schema + refs + bytes/mtime
    rep = validate_run(run_dir, deep=True) # + full sha256 re-hash
    reports = validate_archive(root)

A Report is ok iff it has no error-severity problems; warns don't fail it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from inspection.record.schema import (
    AIRunRecord,
    AnswerRecord,
    ConfigSnapshot,
    Manifest,
    MenuDef,
    OperatorEvent,
    RunRecord,
    SessionRecord,
    StepRecord,
    TranscriptRecord,
    ViewState,
)

STALE_S = 3600.0  # running + no writes for this long -> effectively crashed


@dataclass
class Problem:
    severity: str  # "error" | "warn"
    where: str  # repo-relative-ish path or logical location
    what: str


@dataclass
class Report:
    run_id: str
    path: Path
    problems: list[Problem] = field(default_factory=list)
    effective_status: str | None = None

    @property
    def ok(self) -> bool:
        return not any(p.severity == "error" for p in self.problems)


def _load(report: Report, path: Path, model):
    """Schema-validate one file; record a problem on failure."""
    try:
        return model.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError, OSError) as e:
        report.problems.append(Problem("error", str(path.name), f"{model.__name__}: {e}"))
        return None


def validate_run(run_dir: Path, deep: bool = False,
                 stale_s: float = STALE_S, now: float | None = None) -> Report:
    run_dir = Path(run_dir)
    rep = Report(run_id=run_dir.name, path=run_dir)

    run = _load(rep, run_dir / "run.json", RunRecord)
    if run is None:
        return rep

    # --- required run-level files --------------------------------------------
    if not (run_dir / "config.json").exists():
        rep.problems.append(Problem("error", "config.json", "missing config snapshot"))
    else:
        _load(rep, run_dir / "config.json", ConfigSnapshot)
    if run.rig == "real":
        if not (run_dir / "session.json").exists():
            rep.problems.append(Problem("error", "session.json",
                                        "missing on a real-rig run"))
        else:
            _load(rep, run_dir / "session.json", SessionRecord)

    # --- steps: schema + roster + method refs --------------------------------
    method_ids = {m.id for m in run.view_methods}
    dir_ids: set[int] = set()
    for sdir in sorted((run_dir / "steps").iterdir()) if (run_dir / "steps").exists() else []:
        if not sdir.is_dir():
            continue
        step = _load(rep, sdir / "step.json", StepRecord)
        if step is None:
            continue
        dir_ids.add(step.step_id)
        if step.view.method not in method_ids:
            rep.problems.append(Problem(
                "error", f"steps/{sdir.name}",
                f"view.method '{step.view.method}' not in run.view_methods"))
        if step.outcome == "captured" and step.phase == "fused":
            if not (sdir / "view_state.json").exists():
                rep.problems.append(Problem("error", f"steps/{sdir.name}",
                                            "fused step lacks view_state.json"))
            else:
                _load(rep, sdir / "view_state.json", ViewState)
    roster = set(run.steps)
    for extra in sorted(dir_ids - roster):
        rep.problems.append(Problem("error", f"steps/{extra:03d}",
                                    f"step {extra} on disk but not in roster"))
    for missing in sorted(roster - dir_ids):
        rep.problems.append(Problem("error", "run.json",
                                    f"roster step {missing} has no directory"))

    # --- answer ---------------------------------------------------------------
    ans_path = run_dir / "answer.json"
    ans = _load(rep, ans_path, AnswerRecord) if ans_path.exists() else None
    if run.status == "completed" and ans is None:
        rep.problems.append(Problem("error", "answer.json",
                                    "completed run requires an answer"))
    if ans is not None:
        for sid in ans.step_ids:
            if sid not in roster:
                rep.problems.append(Problem("error", "answer.json",
                                            f"references unknown step {sid}"))

    # --- AI runs --------------------------------------------------------------
    for adir in sorted((run_dir / "ai").iterdir()) if (run_dir / "ai").exists() else []:
        airun = _load(rep, adir / "airun.json", AIRunRecord) \
            if (adir / "airun.json").exists() else None
        menu_path = adir / "menu" / "menu_def.json"
        menu = _load(rep, menu_path, MenuDef) if menu_path.exists() else None
        if airun is not None:
            if menu is not None:
                if airun.menu_hash != menu.content_hash:
                    rep.problems.append(Problem("error", f"ai/{adir.name}",
                                                "menu_hash != MenuDef.content_hash"))
            else:
                # No local menu snapshot to cross-check against — a no-op,
                # but flagged so a missing menu_def.json is never silent.
                rep.problems.append(Problem("warn", f"ai/{adir.name}",
                                            "no menu/menu_def.json — "
                                            "menu_hash cross-check skipped"))
        tdir = adir / "transcripts"
        for tpath in sorted(tdir.glob("*.json")) if tdir.exists() else []:
            tr = _load(rep, tpath, TranscriptRecord)
            if tr and tr.step_id not in roster:
                rep.problems.append(Problem(
                    "error", f"ai/{adir.name}/transcripts/{tpath.name}",
                    f"references unknown step {tr.step_id}"))

    # --- events ---------------------------------------------------------------
    ev_path = run_dir / "events.jsonl"
    if ev_path.exists():
        for i, line in enumerate(ev_path.read_text().splitlines()):
            try:
                OperatorEvent.model_validate_json(line)
            except (ValidationError, json.JSONDecodeError) as e:
                rep.problems.append(Problem("error", "events.jsonl",
                                            f"line {i}: {e}"))

    # --- manifest -------------------------------------------------------------
    man_path = run_dir / "manifest.json"
    man = _load(rep, man_path, Manifest) if man_path.exists() else None
    if man is not None:
        for rel, entry in man.files.items():
            p = run_dir / rel
            if not p.exists():
                rep.problems.append(Problem("error", rel,
                                            f"{rel} in manifest but missing on disk"))
                continue
            st = p.stat()
            if st.st_size != entry.bytes:
                rep.problems.append(Problem("error", rel,
                                            f"size {st.st_size} != recorded bytes {entry.bytes}"))
            elif st.st_mtime != entry.mtime:
                rep.problems.append(Problem("warn", rel, "mtime differs from record"))
            if deep:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                if digest != entry.sha256:
                    rep.problems.append(Problem("error", rel,
                                                "sha256 mismatch — bytes changed"))

    # --- crash detect-on-read -------------------------------------------------
    rep.effective_status = run.status
    if run.status == "running":
        now = now if now is not None else time.time()
        newest = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()),
                     default=0.0)
        if now - newest > stale_s:
            rep.effective_status = "crashed"
            rep.problems.append(Problem("warn", "run.json",
                                        f"running but no writes for {now - newest:.0f}s — "
                                        "effectively crashed"))
    return rep


def validate_archive(root: Path, deep: bool = False) -> list[Report]:
    root = Path(root)
    return [validate_run(d, deep=deep)
            for d in sorted(root.iterdir())
            if d.is_dir() and (d / "run.json").exists()]
