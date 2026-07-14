"""locbench/report.py — report.json, the dev loop's machine artifact (design.md D12).

The iterating agent's contract is `run → read report.json → fix → rerun`, so one file carries
everything: per-episode stats + verdicts (with human-readable failure strings naming the gate
that broke), the run verdict, and provenance (candidate, episode-set version/seed, adapter git
hash, lock/map hashes, timings) — enough to reproduce or disbelieve any number in it.

NaN stats (crashed/dead episodes) serialize as JSON null: `json` would otherwise emit the
non-standard `NaN` literal that plenty of parsers reject. Committed per run; the raw
`pairs.csv` next to it stays gitignored. Pure stdlib → `brain` env.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from .stats import EpisodeStats
from .verdict import EpisodeVerdict, RunVerdict


def build_report(
    *,
    candidate: str,
    scene: str,
    stats: Sequence[EpisodeStats],
    verdicts: Sequence[EpisodeVerdict],
    run: RunVerdict,
    provenance: Dict,
) -> Dict:
    """Assemble the report document (plain dict — `save_report` writes it)."""
    episodes: List[Dict] = []
    for st, v in zip(stats, verdicts):
        episodes.append({
            "stats": {k: _de_nan(x) for k, x in asdict(st).items()},
            "verdict": {"tier": v.tier, "failures": list(v.failures)},
        })
    return {
        "candidate": candidate,
        "scene": scene,
        "run": {"tier": run.tier, "passed": run.passed, "deployable": run.deployable,
                "failed_episodes": list(run.failed_episodes)},
        "episodes": episodes,
        "provenance": dict(provenance),
    }


def save_report(doc: Dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def load_report(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text())


def _de_nan(v):
    return None if isinstance(v, float) and math.isnan(v) else v
