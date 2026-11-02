"""TDD for deploybench scoring (logic/deploybench/deploy_stats.py).

Turns the two raw streams a run collects — the algorithm's `LocalizationOut`s and GT samples —
into the deploy answer (FR5): did it ARRIVE, the localization error stats (reused from
locbench), the KIDNAPPED time-to-first-fix, and the worst dead-reckon gap (longest the pose
went without a map-anchored fix). Pure numpy/stdlib → `brain`.
"""

import math

import pytest

from humanoid.logic.deploybench.deploy_stats import compute_deploy_result, summary_line
from humanoid.logic.deploybench.scenario import Scenario, StartMode
from humanoid.logic.oli.reason.localization import (
    LocalizationOut,
    LocalizationStatus,
    RobotPose,
)

pytestmark = pytest.mark.brain

DT = 100_000_000  # 0.1 s in ns
N = 51            # 0 … 5.0 s


def _est(t_ns, x, y, yaw, status):
    pose = None if status is LocalizationStatus.LOST else RobotPose(stamp_ns=t_ns, x=x, y=y, yaw=yaw)
    return LocalizationOut(
        stamp_ns=t_ns, pose=pose, status=status,
        last_fix_stamp_ns=(t_ns if status is LocalizationStatus.TRACKING else None),
    )


def _known_scn():
    return Scenario(name="s", map_dir="m", start=(0.0, 0.0), goal=(3.0, 0.0),
                    start_mode=StartMode.KNOWN, start_yaw=0.0)


def _gt_stream():
    # GT marches along +x toward the goal; est (built by the caller) sits a fixed offset off it.
    return [(i * DT, i * 0.06, 0.0, 0.0) for i in range(N)]


def test_known_all_tracking_arrives_clean():
    scn = _known_scn()
    gts = _gt_stream()
    ests = [_est(i * DT, i * 0.06 + 0.02, 0.0, 0.0, LocalizationStatus.TRACKING)
            for i in range(N)]
    res = compute_deploy_result(scn, ests, gts, outcome="arrived", episode_start_ns=0)

    assert res.arrived is True
    assert res.ttff_s == pytest.approx(0.0, abs=0.11)          # fix from the first frame
    assert res.longest_dead_reckon_s == pytest.approx(0.1, abs=0.05)
    assert res.stats.pos_mean == pytest.approx(0.02, abs=1e-6)  # constant 2 cm offset, raw frame
    # GT ends at (3.0, 0.0) == the goal, so the physical arrival distance is ~0.
    assert res.final_goal_dist_m == pytest.approx(0.0, abs=1e-6)


def test_kidnapped_recovers_ttff_measured():
    scn = Scenario(name="k", map_dir="m", start=(0.0, 0.0), goal=(3.0, 0.0),
                   start_mode=StartMode.KIDNAPPED)
    gts = _gt_stream()
    ests = []
    for i in range(N):
        t = i * DT
        if t < 2_000_000_000:               # first 2 s: no fix (kidnapped, searching)
            ests.append(_est(t, 0, 0, 0, LocalizationStatus.LOST))
        else:                               # then map-anchored
            ests.append(_est(t, i * 0.06 + 0.02, 0.0, 0.0, LocalizationStatus.TRACKING))
    res = compute_deploy_result(scn, ests, gts, outcome="arrived", episode_start_ns=0)

    assert res.ttff_s == pytest.approx(2.0, abs=0.11)
    assert res.longest_dead_reckon_s == pytest.approx(2.0, abs=0.11)


def test_never_fixes_reports_none():
    scn = _known_scn()
    gts = _gt_stream()
    ests = [_est(i * DT, 0, 0, 0, LocalizationStatus.LOST) for i in range(N)]
    res = compute_deploy_result(scn, ests, gts, outcome="timeout", episode_start_ns=0)

    assert res.arrived is False
    assert res.ttff_s is None
    assert math.isnan(res.stats.pos_mean)       # nothing answered
    assert "timeout" in summary_line(res).lower()
