"""story — the AI narrative of a run, read from disk in order (user story C3).

Works on both generations: new-schema runs (ai/<seq>/ + answer.json) and
legacy runs (eyes/answer.json) via best-effort rendering. Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

from inspection.record.schema import AnswerRecord, TranscriptRecord


def story(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    lines: list[str] = [f"run: {run_dir.name}"]

    # AI runs (new schema)
    ai_root = run_dir / "ai"
    for adir in sorted(ai_root.iterdir()) if ai_root.exists() else []:
        lines.append(f"\n-- ai run {adir.name} --")
        tdir = adir / "transcripts"
        transcripts = []
        for tp in sorted(tdir.glob("*.json")) if tdir.exists() else []:
            try:
                transcripts.append(TranscriptRecord.model_validate_json(tp.read_text()))
            except Exception:
                lines.append(f"  [unreadable transcript {tp.name}]")
        for tr in sorted(transcripts, key=lambda t: t.t):
            ans = json.dumps(tr.answer) if not isinstance(tr.answer, str) else tr.answer
            lines.append(f"  [{tr.t:.0f}] {tr.kind} step {tr.step_id}: "
                         f"{tr.task} -> {ans}")

    # operator events
    ev = run_dir / "events.jsonl"
    if ev.exists():
        kinds = [json.loads(l)["kind"] for l in ev.read_text().splitlines() if l]
        if kinds:
            lines.append(f"\noperator events: {', '.join(kinds)}")

    # verdict — new schema at root, legacy under eyes/
    for ans_path in (run_dir / "answer.json", run_dir / "eyes" / "answer.json"):
        if ans_path.exists():
            try:
                a = AnswerRecord.model_validate_json(ans_path.read_text())
                lines.append(f"\nverdict: {a.verdict}\nreasoning: {a.reasoning}")
            except Exception:  # legacy answer shape — best effort
                raw = json.loads(ans_path.read_text())
                lines.append(f"\nverdict: {raw.get('verdict')}"
                             f"\nreasoning: {raw.get('reasoning', '')}")
            break
    return "\n".join(lines)
