"""TDD for report.json (logic/locbench/report.py) — D12.

The machine artifact the dev loop reads: per-episode stats + verdicts, the run verdict, and
full provenance (candidate, episode-set version/seed, adapter git hash, lock/map hashes,
timings). An agent's iteration is strictly `run → read report.json → fix → rerun`, so the
report must carry everything needed to know WHAT failed and WHY — with no sim access.
Committed (small); the raw pairs.csv stays gitignored. Pure → `brain` env.
"""

import json

import pytest

from humanoid.logic.locbench.pairs import PosePair
from humanoid.logic.locbench.report import build_report, load_report, save_report
from humanoid.logic.locbench.stats import compute_stats
from humanoid.logic.locbench.verdict import episode_verdict, run_verdict
from humanoid.logic.oli.reason.localization import LocalizationStatus

pytestmark = pytest.mark.brain

S = 1_000_000_000


def _stats(episode_id, bias=0.0, outcome="arrived"):
    pairs = [PosePair(stamp_ns=(10 + i) * S, est=(i * 0.1 + bias, 0.0, 0.0),
                      gt=(i * 0.1, 0.0, 0.0), status=LocalizationStatus.TRACKING,
                      assoc_dt_ns=0)
             for i in range(50)]
    return compute_stats(pairs, episode_id=episode_id, outcome=outcome)


def _report():
    stats = [_stats(0, bias=0.02), _stats(1, bias=0.2)]
    verdicts = [episode_verdict(s) for s in stats]
    return build_report(
        candidate="reference",
        scene="warehouse",
        stats=stats,
        verdicts=verdicts,
        run=run_verdict(verdicts),
        provenance={"episode_set_version": 1, "episode_set_seed": 33,
                    "adapter_git": "abc1234", "lock_hash": None, "map_dir_hash": None,
                    "started_at": "2026-07-13T18:00:00", "wall_s": 812.4},
    )


def test_report_carries_the_dev_loop_essentials():
    doc = _report()
    assert doc["candidate"] == "reference" and doc["scene"] == "warehouse"
    assert doc["run"]["tier"] == "FAIL"
    assert doc["run"]["failed_episodes"] == [1]
    ep1 = doc["episodes"][1]
    assert ep1["verdict"]["tier"] == "FAIL"
    # The WHY, human-readable — what the iterating agent acts on:
    assert any("max pos" in f for f in ep1["verdict"]["failures"])
    assert ep1["stats"]["pos_mean"] == pytest.approx(0.2)
    assert doc["provenance"]["episode_set_seed"] == 33


def test_report_is_json_roundtrippable(tmp_path):
    doc = _report()
    p = tmp_path / "report.json"
    save_report(doc, p)
    assert load_report(p) == doc
    json.loads(p.read_text())   # plain JSON on disk, nothing exotic


def test_nan_stats_serialize_safely(tmp_path):
    # A crashed episode has NaN error stats — JSON has no NaN, so they must become null.
    stats = [compute_stats([], episode_id=0, outcome="crashed")]
    verdicts = [episode_verdict(s) for s in stats]
    doc = build_report(candidate="x", scene="warehouse", stats=stats, verdicts=verdicts,
                       run=run_verdict(verdicts), provenance={})
    p = tmp_path / "report.json"
    save_report(doc, p)
    loaded = json.loads(p.read_text())
    assert loaded["episodes"][0]["stats"]["pos_mean"] is None
