"""Legacy adapter — pre-schema runs (2108..3108 eras) as schema objects.

Adapt-on-read (decision 2026-09-03): reads the old layout, returns in-memory
records with provenance="legacy" and every fabricated field declared in
`synthesized`. NEVER writes a byte into the run.

Old layout facts (docs/data-engine/loop-data-audit.md §4):
  run.json = {q_survey, r, turns[{step,target,t,result,stopped}]}
  <NNN>/meta.json = {pose_id, timestamp, joints_rad, T_base_flange, T_base_cam,
                     [rgb_rotation_deg], [mask{score,box,px}|null]}
  session.json (absent on FakeRig/mock runs), eyes/answer.json (if completed)
The view<->cell join is positional: dir NNN's cell = N-th non-survey turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from inspection.record.schema import RunRecord, StepRecord


@dataclass
class AdaptedRun:
    run: RunRecord
    steps: dict[int, StepRecord] = field(default_factory=dict)
    path: Path | None = None


def is_legacy(run_dir: Path) -> bool:
    """Legacy run.json has no schema_version/id — new-schema ones always do."""
    try:
        raw = json.loads((Path(run_dir) / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "schema_version" not in raw


def adapt_run(run_dir: Path) -> AdaptedRun:
    run_dir = Path(run_dir)
    raw = json.loads((run_dir / "run.json").read_text())
    turns = raw.get("turns", [])
    cells = [t["target"] for t in turns if t.get("target") != "survey"]

    view_dirs = sorted(d for d in run_dir.iterdir()
                       if d.is_dir() and d.name.isdigit())
    steps: dict[int, StepRecord] = {}
    non_survey_i = 0
    for vdir in view_dirs:
        meta_path = vdir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        pose_id = meta["pose_id"]
        synthesized = ["t_arrived"]  # old runs have one timestamp, not two
        if pose_id == 0:
            address = None
        else:
            address = cells[non_survey_i] if non_survey_i < len(cells) else None
            if address is None:
                synthesized.append("view.address")
            non_survey_i += 1
        rot = meta.get("rgb_rotation_deg")
        if rot is None:
            rot = 0  # pre-2026-08-24 captures are raw on disk
            synthesized.append("rgb_rotation_deg")
        mask = meta.get("mask")
        steps[pose_id] = StepRecord(
            provenance="legacy", synthesized=synthesized,
            step_id=pose_id,
            view={"method": "vs-legacy", "address": address},
            t_arrived=meta["timestamp"], t_captured=meta["timestamp"],
            outcome="captured", phase="captured",
            joints_rad=meta["joints_rad"],
            T_base_flange=meta["T_base_flange"],
            T_base_cam=meta["T_base_cam"],
            rgb_rotation_deg=rot,
            segmentation={"score": mask["score"], "box": mask["box"],
                          "px": mask["px"]} if mask else None,
        )

    timestamps = [s.t_captured for s in steps.values()]
    # Status inference (Anton 2026-09-07): a capture-only run (no eyes/ at all)
    # that has steps COMPLETED its sweep — there was no verdict to reach.
    # Only an AI run that never answered, or an empty run, is aborted.
    ai_ran = any((run_dir / "eyes" / f).exists()
                 for f in ("store.json", "trace.jsonl", "transcripts"))
    if not steps:
        status = "aborted"
    elif not ai_ran:
        status = "completed"  # capture-only sweep (cup3/4/5, seeded pattern)
    elif (run_dir / "eyes" / "answer.json").exists():
        status = "completed"
    else:
        status = "aborted"  # AI ran, trace stopped, no verdict
    run = RunRecord(
        provenance="legacy",
        synthesized=["source", "status", "created_at", "view_methods"],
        id=run_dir.name, name=run_dir.name.split("-", 1)[-1],
        source="live", rig="real" if (run_dir / "session.json").exists() else "fake",
        status=status,
        created_at=min(timestamps) if timestamps else 0.0,
        view_methods=[{"id": "vs-legacy", "kind": "viewsphere-legacy",
                       "params": {"r": raw.get("r")}}],
        steps=sorted(steps),
        q_survey=raw.get("q_survey"),
    )
    return AdaptedRun(run=run, steps=steps, path=run_dir)


def load_any(run_dir: Path) -> AdaptedRun:
    """Uniform loader: new-schema run or legacy run -> AdaptedRun."""
    run_dir = Path(run_dir)
    if is_legacy(run_dir):
        return adapt_run(run_dir)
    run = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    steps = {}
    for sid in run.steps:
        sp = run_dir / "steps" / f"{sid:03d}" / "step.json"
        if sp.exists():
            steps[sid] = StepRecord.model_validate_json(sp.read_text())
    return AdaptedRun(run=run, steps=steps, path=run_dir)
