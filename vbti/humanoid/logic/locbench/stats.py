"""locbench/stats.py — raw per-episode error statistics (design.md D10).

Per pair: position error = euclidean distance est↔GT [m]; yaw error = wrapped absolute
difference [deg]; both computed on RAW map-frame poses — no Umeyama, no anchor fit, nothing
that could eat a constant bias (E1: the measured nav-killing failure mode). Coverage counts
LOST/unanswered ticks against the candidate; error aggregates run over answered ticks only
(a LOST tick is a coverage problem, not a 0-meter success).

Aggregates per episode: mean / median / p95 / max, pose rate, and the episode outcome the
evaluator observed (arrived / timeout / crashed). Pure numpy/stdlib → `brain` env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .pairs import PosePair


@dataclass(frozen=True)
class EpisodeStats:
    episode_id: int
    outcome: str            # "arrived" | "timeout" | "crashed"
    n_ticks: int            # paired ticks (answered + LOST)
    n_answered: int
    coverage: float         # n_answered / n_ticks (0.0 for a dead episode)
    pos_mean: float         # [m] — NaN when nothing was answered
    pos_median: float
    pos_p95: float
    pos_max: float
    yaw_mean_deg: float
    yaw_median_deg: float
    yaw_p95_deg: float
    yaw_max_deg: float
    pose_rate_hz: float     # answered ticks per second of estimate span


def compute_stats(pairs: Sequence[PosePair], *, episode_id: int, outcome: str) -> EpisodeStats:
    n_ticks = len(pairs)
    answered = [p for p in pairs if p.est is not None]
    pos_err: List[float] = []
    yaw_err_deg: List[float] = []
    for p in answered:
        assert p.est is not None
        pos_err.append(math.dist(p.est[:2], p.gt[:2]))
        yaw_err_deg.append(abs(math.degrees(_wrap(p.est[2] - p.gt[2]))))

    if answered:
        span_s = max((answered[-1].stamp_ns - answered[0].stamp_ns) / 1e9, 1e-9)
        rate = len(answered) / span_s if len(answered) > 1 else 0.0
    else:
        rate = 0.0

    return EpisodeStats(
        episode_id=episode_id,
        outcome=outcome,
        n_ticks=n_ticks,
        n_answered=len(answered),
        coverage=(len(answered) / n_ticks) if n_ticks else 0.0,
        pos_mean=_agg(pos_err, np.mean),
        pos_median=_agg(pos_err, np.median),
        pos_p95=_agg(pos_err, lambda a: np.percentile(a, 95)),
        pos_max=_agg(pos_err, np.max),
        yaw_mean_deg=_agg(yaw_err_deg, np.mean),
        yaw_median_deg=_agg(yaw_err_deg, np.median),
        yaw_p95_deg=_agg(yaw_err_deg, lambda a: np.percentile(a, 95)),
        yaw_max_deg=_agg(yaw_err_deg, np.max),
        pose_rate_hz=rate,
    )


def _wrap(angle_rad: float) -> float:
    """Wrap to (-π, π] so a −179°-vs-+179° disagreement is a 2° error, not 358°."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _agg(values: List[float], fn) -> float:
    return float(fn(np.asarray(values))) if values else float("nan")
