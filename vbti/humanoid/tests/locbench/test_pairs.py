"""TDD for pose-pair logging + association (logic/locbench/pairs.py) — D10.

The evaluator collects two independent streams during an episode: the candidate's estimates
(`LocalizationOut` off W5 telemetry) and GT poses (off the W3 debug-pose channel). Scoring
needs them as PAIRS: for each estimate, the GT sample nearest in stamp (|Δt| ≤ 100 ms, else
the estimate is unpaired), with the first 2 s after episode start excluded as warmup. Pairs
persist to CSV per episode (gitignored raw data; `locbench score` recomputes offline).
Pure stdlib → `brain` env.
"""

import pytest

from humanoid.logic.locbench.pairs import (
    MAX_ASSOC_DT_NS,
    WARMUP_NS,
    PosePair,
    associate,
    load_pairs_csv,
    save_pairs_csv,
)
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)

pytestmark = pytest.mark.brain

MS = 1_000_000
S = 1_000_000_000


def _est(stamp_ns, x, y, yaw=0.0, status=LocalizationStatus.TRACKING):
    if status is LocalizationStatus.LOST:
        return LocalizationOut(stamp_ns=stamp_ns, pose=None, status=status)
    return LocalizationOut(stamp_ns=stamp_ns, pose=RobotPose(stamp_ns, x, y, yaw),
                           status=status)


def _gt(stamp_ns, x, y, yaw=0.0):
    return (stamp_ns, x, y, yaw)


def test_pairs_nearest_gt_by_stamp():
    t0 = 10 * S
    ests = [_est(t0 + 3 * S, 1.0, 1.0), _est(t0 + 4 * S, 2.0, 2.0)]
    gts = [_gt(t0 + 3 * S - 20 * MS, 1.1, 1.0), _gt(t0 + 3 * S + 5 * MS, 1.05, 1.0),
           _gt(t0 + 4 * S + 1 * MS, 2.0, 2.1)]
    pairs = associate(ests, gts, episode_start_ns=t0)
    assert len(pairs) == 2
    assert pairs[0].gt == (1.05, 1.0, 0.0)      # nearest by |Δt|, not first-before
    assert pairs[1].gt == (2.0, 2.1, 0.0)


def test_estimate_without_close_gt_is_unpaired():
    t0 = 0
    ests = [_est(t0 + 5 * S, 1.0, 1.0)]
    gts = [_gt(t0 + 5 * S + MAX_ASSOC_DT_NS + 1, 1.0, 1.0)]
    pairs = associate(ests, gts, episode_start_ns=t0)
    assert pairs == []   # 100 ms is the D10 association budget — past it, no pair


def test_warmup_window_excluded():
    t0 = 100 * S
    inside = _est(t0 + WARMUP_NS - 1, 5.0, 5.0)
    after = _est(t0 + WARMUP_NS + 1, 6.0, 6.0)
    gts = [_gt(inside.stamp_ns, 5.0, 5.0), _gt(after.stamp_ns, 6.0, 6.0)]
    pairs = associate([inside, after], gts, episode_start_ns=t0)
    assert len(pairs) == 1 and pairs[0].est == (6.0, 6.0, 0.0)


def test_lost_estimates_kept_as_gap_markers():
    # LOST ticks matter for coverage: they pair with GT (the truth existed) but carry no
    # est pose — the stats layer counts them as unanswered.
    t0 = 0
    ests = [_est(3 * S, 0, 0, status=LocalizationStatus.LOST)]
    gts = [_gt(3 * S, 1.0, 2.0)]
    pairs = associate(ests, gts, episode_start_ns=t0)
    assert len(pairs) == 1
    assert pairs[0].est is None and pairs[0].gt == (1.0, 2.0, 0.0)
    assert pairs[0].status is LocalizationStatus.LOST


def test_csv_roundtrip(tmp_path):
    t0 = 0
    ests = [_est(3 * S, 1.0, 1.0, 0.5), _est(4 * S, 0, 0, status=LocalizationStatus.LOST)]
    gts = [_gt(3 * S, 1.1, 1.0, 0.45), _gt(4 * S, 2.0, 2.0, 0.0)]
    pairs = associate(ests, gts, episode_start_ns=t0)
    p = tmp_path / "pairs.csv"
    save_pairs_csv(pairs, p)
    assert load_pairs_csv(p) == pairs
    assert isinstance(load_pairs_csv(p)[0], PosePair)
