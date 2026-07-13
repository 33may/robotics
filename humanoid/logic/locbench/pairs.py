"""locbench/pairs.py — pose-pair association + the raw per-episode CSV (design.md D10).

Two independent streams come out of an episode: the candidate's `LocalizationOut`s (W5
telemetry) and GT samples (W3 debug-pose). A `PosePair` joins one estimate with the GT sample
nearest in stamp — |Δt| ≤ 100 ms or the estimate is dropped as unpairable; the first 2 s
after episode start are warmup and excluded. LOST estimates ARE paired (est=None): the truth
existed and the candidate said nothing — the stats layer counts them against coverage.

No alignment, no anchoring, ever: pairs carry raw map-frame poses so a constant bias — the
measured nav-killing failure mode (E1) — stays visible all the way to the verdict.

CSV per episode is the gitignored raw artifact; `locbench score` recomputes stats/plots
offline from it. Pure stdlib → runs anywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..oli.reason.localization import LocalizationOut, LocalizationStatus

MAX_ASSOC_DT_NS: int = 100_000_000   # 100 ms — D10 association budget
WARMUP_NS: int = 2_000_000_000       # 2 s excluded after episode start

Pose3 = Tuple[float, float, float]   # (x, y, yaw), map frame
GtSample = Tuple[int, float, float, float]   # (stamp_ns, x, y, yaw) — debug-pose shape


@dataclass(frozen=True)
class PosePair:
    """One scored tick: the candidate's answer (None = LOST) against the nearest GT."""

    stamp_ns: int                    # the estimate's stamp
    est: Optional[Pose3]             # None ⇔ status is LOST
    gt: Pose3
    status: LocalizationStatus
    assoc_dt_ns: int                 # |est stamp − gt stamp| actually used


def associate(
    estimates: Sequence[LocalizationOut],
    gt_samples: Sequence[GtSample],
    *,
    episode_start_ns: int,
) -> List[PosePair]:
    """Pair each post-warmup estimate with the GT sample nearest in stamp (≤ 100 ms)."""
    gts = sorted(gt_samples)
    if not gts:
        return []
    stamps = [g[0] for g in gts]
    pairs: List[PosePair] = []
    for est in estimates:
        if est.stamp_ns - episode_start_ns < WARMUP_NS:
            continue
        idx = _nearest(stamps, est.stamp_ns)
        dt = abs(stamps[idx] - est.stamp_ns)
        if dt > MAX_ASSOC_DT_NS:
            continue
        g = gts[idx]
        pairs.append(PosePair(
            stamp_ns=est.stamp_ns,
            est=(est.pose.x, est.pose.y, est.pose.yaw) if est.pose is not None else None,
            gt=(g[1], g[2], g[3]),
            status=est.status,
            assoc_dt_ns=dt,
        ))
    return pairs


def _nearest(sorted_stamps: List[int], t: int) -> int:
    """Index of the stamp nearest to t (binary search on the sorted GT stream)."""
    import bisect

    i = bisect.bisect_left(sorted_stamps, t)
    if i == 0:
        return 0
    if i == len(sorted_stamps):
        return len(sorted_stamps) - 1
    return i if sorted_stamps[i] - t < t - sorted_stamps[i - 1] else i - 1


# ── CSV persistence (the gitignored raw artifact) ────────────────────────────

_FIELDS = ["stamp_ns", "est_x", "est_y", "est_yaw", "gt_x", "gt_y", "gt_yaw",
           "status", "assoc_dt_ns"]


def save_pairs_csv(pairs: Sequence[PosePair], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for p in pairs:
            ex, ey, eyaw = p.est if p.est is not None else ("", "", "")
            w.writerow([p.stamp_ns, ex, ey, eyaw, *p.gt, int(p.status), p.assoc_dt_ns])


def load_pairs_csv(path: str | Path) -> List[PosePair]:
    pairs: List[PosePair] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            est = (None if row["est_x"] == "" else
                   (float(row["est_x"]), float(row["est_y"]), float(row["est_yaw"])))
            pairs.append(PosePair(
                stamp_ns=int(row["stamp_ns"]),
                est=est,
                gt=(float(row["gt_x"]), float(row["gt_y"]), float(row["gt_yaw"])),
                status=LocalizationStatus(int(row["status"])),
                assoc_dt_ns=int(row["assoc_dt_ns"]),
            ))
    return pairs
