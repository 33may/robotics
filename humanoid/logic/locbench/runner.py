"""locbench/runner.py — `locbench run`: boot the stack, drive the episodes, write the run.

Single-entrypoint rule: the runner spawns the LAUNCHER (`logic/oli/launcher.py --service
--shadow <name> ...`) as one subprocess — the Supervisor inside it boots World + brain and
tears both down when the runner is done. The evaluator here is a pure client on the seam
(P3 in the design diagram).

GT source (Stage 1): telemetry's `pose` field IS ground truth in shadow mode — Nav drives on
the debug-pose channel, whose single reader is the brain. The evaluator therefore reads GT
off W5 and binds nothing extra. Stage 2 flips Nav onto the candidate's pose, so §10 adds a
dedicated GT feed for the judge (World-side, additive) — documented, not built here.

Run artifacts (D12), per `runs/<candidate>/<run-id>/`: `report.json` (committed),
`ep<k>_pairs.csv` (gitignored raw), plots (committed), `run.log`. Exit code: 0 = PASS or
DEPLOY, 1 = FAIL, 2 = infrastructure failure (stack died before any episode).
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .episodes import EpisodeSet, load_episode_set
from .evaluator import EvalConfig, Evaluator, EpisodeResult
from .pairs import associate, save_pairs_csv
from .report import build_report, save_report
from .stats import compute_stats
from .verdict import episode_verdict, run_verdict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO_ROOT / "humanoid" / "logic" / "oli" / "launcher.py"


def write_run_artifacts(
    run_dir: Path,
    *,
    candidate: str,
    episode_set: EpisodeSet,
    results: List[EpisodeResult],
    grid=None,
    provenance: Optional[dict] = None,
) -> dict:
    """Pairs → stats → verdicts → report.json (+ plots when a grid is given). Pure."""
    run_dir.mkdir(parents=True, exist_ok=True)
    all_pairs, stats = [], []
    for res in results:
        pairs = associate(res.ests, res.gts, episode_start_ns=res.episode_start_ns)
        save_pairs_csv(pairs, run_dir / f"ep{res.episode.id}_pairs.csv")
        all_pairs.append(pairs)
        stats.append(compute_stats(pairs, episode_id=res.episode.id, outcome=res.outcome))
    verdicts = [episode_verdict(s) for s in stats]
    run = run_verdict(verdicts)
    doc = build_report(
        candidate=candidate, scene=episode_set.scene, stats=stats, verdicts=verdicts,
        run=run, provenance={
            "episode_set_version": episode_set.version,
            "episode_set_seed": episode_set.seed,
            "episode_errors": {r.episode.id: r.error for r in results if r.error},
            **(provenance or {}),
        })
    save_report(doc, run_dir / "report.json")

    if grid is not None:
        from .plots import (
            plot_error_distribution,
            plot_error_timeline,
            plot_run_sheet,
        )
        episodes = [r.episode for r in results]
        tiers = [v.tier for v in verdicts]
        plot_run_sheet(grid, episodes, all_pairs, tiers, run_dir / "run_sheet.png")
        plot_error_timeline(all_pairs, run_dir / "error_timeline.png")
        plot_error_distribution(all_pairs, run_dir / "error_cdf.png")
    return doc


def score_run_dir(run_dir: Path, episode_set: EpisodeSet, grid=None) -> dict:
    """`locbench score` — recompute report + plots offline from the stored pairs."""
    from .pairs import load_pairs_csv
    from .report import load_report

    old = load_report(run_dir / "report.json")
    results = []
    for ep_doc in old["episodes"]:
        ep_id = ep_doc["stats"]["episode_id"]
        ep = next(e for e in episode_set.episodes if e.id == ep_id)
        pairs = load_pairs_csv(run_dir / f"ep{ep_id}_pairs.csv")
        res = EpisodeResult(episode=ep, outcome=ep_doc["stats"]["outcome"])
        # stored pairs are already associated — hand them through a passthrough result
        res._pairs = pairs  # type: ignore[attr-defined]
        results.append(res)
    # recompute stats directly from the stored pairs (no re-association)
    stats = [compute_stats(getattr(r, "_pairs"), episode_id=r.episode.id, outcome=r.outcome)
             for r in results]
    verdicts = [episode_verdict(s) for s in stats]
    doc = build_report(candidate=old["candidate"], scene=old["scene"], stats=stats,
                       verdicts=verdicts, run=run_verdict(verdicts),
                       provenance=old.get("provenance", {}))
    save_report(doc, run_dir / "report.json")
    if grid is not None:
        from .plots import plot_error_distribution, plot_error_timeline, plot_run_sheet
        all_pairs = [getattr(r, "_pairs") for r in results]
        plot_run_sheet(grid, [r.episode for r in results], all_pairs,
                       [v.tier for v in verdicts], run_dir / "run_sheet.png")
        plot_error_timeline(all_pairs, run_dir / "error_timeline.png")
        plot_error_distribution(all_pairs, run_dir / "error_cdf.png")
    return doc


def board(runs_root: Path) -> str:
    """`locbench board` — one MD table row per candidate's LATEST report."""
    rows = []
    for report_path in sorted(runs_root.glob("*/*/report.json")):
        doc = json.loads(report_path.read_text())
        rows.append((doc["candidate"], report_path.parent.name, doc))
    latest = {}
    for cand, run_id, doc in rows:
        if cand not in latest or run_id > latest[cand][0]:
            latest[cand] = (run_id, doc)
    lines = ["| candidate | run | tier | episodes pass | worst mean pos | worst max pos |",
             "|---|---|---|---|---|---|"]
    for cand in sorted(latest):
        run_id, doc = latest[cand]
        eps = doc["episodes"]
        n_pass = sum(1 for e in eps if e["verdict"]["tier"] != "FAIL")
        means = [e["stats"]["pos_mean"] for e in eps if e["stats"]["pos_mean"] is not None]
        maxes = [e["stats"]["pos_max"] for e in eps if e["stats"]["pos_max"] is not None]
        lines.append(
            f"| {cand} | {run_id} | **{doc['run']['tier']}** | {n_pass}/{len(eps)} | "
            f"{max(means):.3f} m | {max(maxes):.3f} m |" if means else
            f"| {cand} | {run_id} | **{doc['run']['tier']}** | {n_pass}/{len(eps)} | — | — |")
    return "\n".join(lines)


# ── the live run (spawns the launcher; smoked live, not unit-tested) ─────────


def run_bench(
    *,
    candidate: str,
    scene_cfg: dict,
    episodes_file: Path,
    runs_root: Path,
    n_episodes: Optional[int] = None,
    headless: bool = False,
    timeout_s: float = 90.0,
    shadow_config: Optional[str] = None,
    log=print,
) -> int:
    from humanoid.logic.locbench.envs import bench_env_name, env_exists
    from humanoid.logic.oli.reason.mapping import StaticMapping
    from humanoid.logic.oli.service import GoalChannelClient, LocCtrlClient, TelemetryClient

    # D8: the WHOLE brain boots inside the candidate's disposable env (Anton, 14-07-2026:
    # hard-error if it is missing — no silent fallback to `brain`, so a run is always
    # reproducible from its lock.yml). `locbench env create <candidate>` builds it.
    brain_env = bench_env_name(candidate)
    if not env_exists(candidate):
        log(f"[locbench] env {brain_env!r} not found — run `locbench env create {candidate}` first")
        return 3

    es = load_episode_set(episodes_file)
    episodes = list(es.episodes)[: n_episodes or len(es.episodes)]
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / candidate / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    argv = [sys.executable, str(_LAUNCHER), "--sim", "isaac", "--mode", "glide",
            "--service", "--shadow", candidate, "--brain-env", brain_env, "--cameras",
            "--debug-pose", "--map", str(_REPO_ROOT / "humanoid" / es.map_dir),
            "--log", str(run_dir / "stack.log")]
    if scene_cfg.get("scene_usd"):
        argv += ["--scene", str(scene_cfg["scene_usd"])]
    if scene_cfg.get("camera_every"):
        argv += ["--camera-every", str(scene_cfg["camera_every"])]
    if headless:
        argv.append("--headless")
    if shadow_config:
        argv += ["--shadow-config", shadow_config]

    telemetry = TelemetryClient()          # bind BEFORE the brain starts publishing
    log(f"[locbench] booting stack: {' '.join(argv)}")
    proc = subprocess.Popen(argv, cwd=str(_REPO_ROOT / "humanoid"))
    t0 = time.monotonic()
    try:
        # the brain is up when telemetry flows
        while telemetry.latest() is None:
            if proc.poll() is not None:
                log("[locbench] stack died during boot")
                return 2
            time.sleep(0.5)
        log(f"[locbench] stack up in {time.monotonic() - t0:.0f}s — running "
            f"{len(episodes)} episode(s)")

        goals = GoalChannelClient()
        ctrl = LocCtrlClient()
        # Bench GT feed: republish every new GT sample to a socket the candidate MAY read
        # (`Setup.calibration["gt_feed_socket"]`). The reference candidate consumes it (D13);
        # real SLAM candidates ignore the key. Fire-and-forget — no reader, no cost.
        from humanoid.logic.oli.comm.debug_pose import DebugPoseServer
        gt_feed = DebugPoseServer("/tmp/oli-gt-feed.sock")
        last_fed = [0]

        def gt_latest():
            sample = _gt_from_telemetry(telemetry.latest())
            if sample is not None and sample[0] > last_fed[0]:
                last_fed[0] = sample[0]
                gt_feed.publish(*sample)
            return sample

        ev = Evaluator(
            send_goal=goals.send_goal, clear_goal=goals.clear_goal,
            send_start=ctrl.send_start, send_stop=ctrl.send_stop,
            gt_latest=gt_latest,
            telemetry_latest=telemetry.latest,
            stack_alive=lambda: proc.poll() is None,
            map_dir=str(_REPO_ROOT / "humanoid" / es.map_dir),
            calibration={"gt_feed_socket": "/tmp/oli-gt-feed.sock"},
            config=EvalConfig(timeout_s=timeout_s),
            log=log,
        )
        started = _dt.datetime.now().isoformat(timespec="seconds")
        results = ev.run(episodes)
        goals.close()
        ctrl.close()
        gt_feed.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        telemetry.close()

    grid = StaticMapping(str(_REPO_ROOT / "humanoid" / es.map_dir)).latest().grid
    doc = write_run_artifacts(
        run_dir, candidate=candidate, episode_set=es, results=results, grid=grid,
        provenance={"started_at": started, "wall_s": round(time.monotonic() - t0, 1),
                    "adapter_git": _git_head(), "timeout_s": timeout_s})
    log(f"[locbench] run {run_id}: {doc['run']['tier']} "
        f"(failed episodes: {doc['run']['failed_episodes'] or 'none'})")
    log(f"[locbench] artifacts → {run_dir}")
    return 0 if doc["run"]["passed"] else 1


def _gt_from_telemetry(snap):
    """Stage 1: telemetry `pose` is Nav's driving pose = GT in shadow mode (see module doc)."""
    if snap is None or snap.pose is None:
        return None
    return (snap.stamp_ns, snap.pose[0], snap.pose[1], snap.pose[2])


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=str(_REPO_ROOT)).stdout.strip() or None
    except OSError:
        return None
