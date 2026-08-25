#!/usr/bin/env python3
"""The run trace — every stage of the agentic loop, on disk.

`<run>/eyes/trace.jsonl`, one JSON object per line, append-only. This is the
observability artifact AND the reason the UI needs no live process: watching a
run is just reading a file that happens to still be growing, and re-opening a
finished run three days later uses the identical path. Anton 2026-08-25.

Kinds, in the order they typically appear:

    run       the question, model, run dir            — once, first
    thinking  orchestrator extended-thinking block
    text      orchestrator prose
    tool_call verb + args (the QUESTION and SCHEMA it asked for live here)
    tool_result what came back, plus images and the subagent's own transcript
    state     what the PostToolUse hook stapled on (context the model was
              given, not text it wrote — visible so behaviour is explicable)
    write     a store write, tagged with its trust tier (fact|plan|finding)
    answer    the verdict                             — once, last

Image paths are stored **relative to the run dir** (`"003/rgb.png"`), never as
URLs. The brain does not know the UI exists; the publisher maps paths onto the
`/captures` mount at publish time (`ui/publisher.py:_capture_url`).
"""
import json
import time
from pathlib import Path

TRACE_NAME = "trace.jsonl"


class TraceWriter:
    """Append-only. Never raises into the loop — a broken trace must not kill
    a run that is otherwise fine (same contract as the UI publisher)."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "eyes" / TRACE_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._seq = 0

    def event(self, kind, **fields):
        try:
            rec = {"seq": self._seq, "t": time.time(), "kind": kind, **fields}
            self._seq += 1
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            return rec
        except Exception:                       # noqa: BLE001 - see docstring
            return None


def read_trace(run_dir):
    """Every event of a run, oldest first. Partial last lines are skipped —
    the file is being appended to while this reads it."""
    p = Path(run_dir) / "eyes" / TRACE_NAME
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                            # a half-written tail line
    return out
