"""
Training monitoring and log analysis.

Designed for both human CLI use and AI agent consumption.
Every function is highly configurable — the agent decides what range,
how many points, which metrics, what perspective.

Commands:
    snapshot  — live status (configurable detail level)
    logs      — view log entries (range, metric filter)
    trend     — loss trend analysis (configurable window, threshold)
    compare   — side-by-side version comparison
    history   — raw metric history (configurable sampling)
    metrics   — list available metrics in a run
"""

import json
from pathlib import Path

from vbti.logic.train.experiment_utils import (
    _resolve_experiment, _resolve_version, _version_dir, load_config,
)


# ── Raw log access ────────────────────────────────────────────────────────────

def _load_log(experiment: str, version: str) -> list[dict]:
    """Load all entries from training_log.jsonl."""
    log_path = _version_dir(experiment, version) / "metrics" / "training_log.jsonl"
    if not log_path.exists():
        return []
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _filter_entries(entries: list[dict], step_from: int = 0, step_to: int | None = None,
                    metric: str | None = None) -> list[dict]:
    """Filter entries by step range and optionally by metric presence."""
    filtered = [e for e in entries if e["step"] >= step_from]
    if step_to is not None:
        filtered = [e for e in filtered if e["step"] <= step_to]
    if metric:
        filtered = [e for e in filtered if metric in e]
    return filtered


def _extract_metric(entries: list[dict], key: str) -> list[tuple[int, float]]:
    """Extract (step, value) pairs for a metric."""
    return [(e["step"], e[key]) for e in entries if key in e]


def _sample(items: list, n: int) -> list:
    """Downsample a list to n items, preserving first and last."""
    if len(items) <= n:
        return items
    step = len(items) / (n - 1)
    indices = sorted(set([int(i * step) for i in range(n - 1)] + [len(items) - 1]))
    return [items[i] for i in indices]


# ── Metrics discovery ─────────────────────────────────────────────────────────

def metrics(experiment: str | None = None, version: str | None = None):
    """List all metrics available in a run with their step counts."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    entries = _load_log(experiment, version)

    if not entries:
        print("No log entries yet.")
        return {}

    skip = {"step", "timestamp"}
    metric_counts = {}
    for e in entries:
        for k in e:
            if k not in skip:
                metric_counts[k] = metric_counts.get(k, 0) + 1

    print(f"\n  {experiment}/{version} — {len(entries)} log entries, steps {entries[0]['step']}..{entries[-1]['step']}")
    print(f"  Available metrics:")
    for k, count in sorted(metric_counts.items()):
        print(f"    {k:20s}  {count} entries")

    return metric_counts


# ── Snapshot ──────────────────────────────────────────────────────────────────

def snapshot(points: int = 30, step_from: int = 0, step_to: int | None = None,
             include_config: bool = True, include_history: bool = True,
             experiment: str | None = None, version: str | None = None) -> dict:
    """Live snapshot of a training run.

    Args:
        points: how many history points to include (more = more context for AI)
        step_from: start of step range (0 = from beginning)
        step_to: end of step range (None = latest)
        include_config: include config summary in output
        include_history: include sampled metric history
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    entries = _load_log(experiment, version)
    entries = _filter_entries(entries, step_from, step_to)

    if not entries:
        return {"experiment": experiment, "version": version, "status": "no_data"}

    last = entries[-1]

    snap = {
        "experiment": experiment,
        "version": version,
        "step": last["step"],
        "total_entries": len(entries),
        "step_range": [entries[0]["step"], last["step"]],
        "timestamp": last.get("timestamp"),
    }

    # Per-metric summaries
    skip = {"step", "timestamp"}
    all_keys = set()
    for e in entries:
        all_keys.update(k for k in e if k not in skip)

    for key in sorted(all_keys):
        series = _extract_metric(entries, key)
        if not series:
            continue
        values = [v for _, v in series]
        is_loss = "loss" in key
        snap[key] = {
            "current": values[-1],
            "best": min(values) if is_loss else max(values),
            "worst": max(values) if is_loss else min(values),
            "mean": sum(values) / len(values),
            "entries": len(values),
            "trend": _compute_trend(values),
        }
        if is_loss:
            best_val = min(values)
            best_idx = values.index(best_val)
            snap[key]["best_at_step"] = series[best_idx][0]

    if include_history:
        snap["history"] = _build_history(entries, all_keys, points)

    if include_config:
        try:
            cfg = load_config(experiment, version)
            snap["config_summary"] = {
                "model": cfg.get("model", {}).get("type"),
                "dataset": cfg.get("dataset", {}).get("sources", [{}])[0].get("repo_id"),
                "lr": cfg.get("training", {}).get("lr"),
                "steps": cfg.get("training", {}).get("steps"),
                "batch_size": cfg.get("training", {}).get("batch_size"),
            }
        except Exception:
            pass

    return snap


def _build_history(entries: list[dict], keys: set[str], max_points: int) -> dict:
    """Build sampled history for all metrics."""
    sampled = _sample(entries, max_points)
    result = {"steps": [e["step"] for e in sampled]}
    for key in sorted(keys):
        values = [e.get(key) for e in sampled]
        if any(v is not None for v in values):
            result[key] = values
    return result


def _compute_trend(values: list[float], window: int = 5) -> str:
    """Classify trend from recent values."""
    if len(values) < 3:
        return "insufficient_data"

    recent = values[-window:] if len(values) >= window else values
    earlier = values[-2 * window:-window] if len(values) >= 2 * window else values[:len(values) // 2]

    if not earlier:
        return "insufficient_data"

    recent_avg = sum(recent) / len(recent)
    earlier_avg = sum(earlier) / len(earlier)
    change = (recent_avg - earlier_avg) / (earlier_avg + 1e-10)

    if change < -0.05:
        return "decreasing"
    elif change > 0.05:
        return "increasing"
    else:
        return "plateau"


# ── Logs ──────────────────────────────────────────────────────────────────────

def logs(last: int = 10, first: int = 0, step_from: int = 0, step_to: int | None = None,
         metric: str | None = None,
         experiment: str | None = None, version: str | None = None):
    """View log entries with flexible filtering.

    Args:
        last: show last N entries (0 = don't tail)
        first: show first N entries (0 = don't head)
        step_from: filter entries from this step
        step_to: filter entries up to this step
        metric: only show entries containing this metric
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    entries = _load_log(experiment, version)
    entries = _filter_entries(entries, step_from, step_to, metric)

    if not entries:
        print("No matching log entries.")
        return

    # Select range
    if last > 0 and first == 0:
        show = entries[-last:]
    elif first > 0 and last == 0:
        show = entries[:first]
    elif first > 0 and last > 0:
        show = entries[:first] + [{"_separator": True}] + entries[-last:]
    else:
        show = entries

    for entry in show:
        if entry.get("_separator"):
            print("  ...")
            continue

        step = entry.get("step", "?")
        parts = [f"Step {step:>6}"]

        if "train_loss" in entry:
            parts.append(f"loss={entry['train_loss']:.4f}")
        if "val_loss" in entry:
            parts.append(f"val={entry['val_loss']:.4f}")
        if "lr" in entry:
            parts.append(f"lr={entry['lr']:.2e}")

        skip = {"step", "timestamp", "train_loss", "val_loss", "lr", "loss", "_separator"}
        extras = {k: v for k, v in entry.items() if k not in skip and isinstance(v, (int, float))}
        for k, v in extras.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")

        print(" | ".join(parts))


# ── Trend analysis ────────────────────────────────────────────────────────────

def trend(metric: str = "train_loss", window: int = 10, threshold: float = 0.02,
          step_from: int = 0, step_to: int | None = None,
          experiment: str | None = None, version: str | None = None):
    """Analyze metric trend with configurable parameters.

    Args:
        metric: which metric to analyze
        window: number of recent points for trend calculation
        threshold: plateau threshold — values within this % range = plateau
        step_from: analyze from this step
        step_to: analyze up to this step
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    entries = _load_log(experiment, version)
    entries = _filter_entries(entries, step_from, step_to, metric)
    series = _extract_metric(entries, metric)

    if len(series) < 3:
        print(f"Not enough data for trend analysis ({len(series)} points)")
        return {}

    steps, values = zip(*series)
    steps = list(steps)
    values = list(values)

    # Direction
    direction = _compute_trend(values, window)

    # Improvement rate
    total_change = (values[-1] - values[0]) / (values[0] + 1e-10) * 100
    steps_elapsed = steps[-1] - steps[0]
    rate_per_100 = total_change / (steps_elapsed / 100) if steps_elapsed > 0 else 0

    # Plateau detection
    recent = values[-window:]
    recent_mean = sum(recent) / len(recent)
    recent_range = (max(recent) - min(recent)) / (recent_mean + 1e-10)
    is_plateau = recent_range < threshold

    # Best value
    is_loss = "loss" in metric
    best_val = min(values) if is_loss else max(values)
    best_step = steps[values.index(best_val)]
    steps_since_best = steps[-1] - best_step

    result = {
        "metric": metric,
        "direction": direction,
        "points": len(values),
        "step_range": [steps[0], steps[-1]],
        "current": values[-1],
        "best": best_val,
        "best_at_step": best_step,
        "steps_since_best": steps_since_best,
        "total_change_pct": round(total_change, 3),
        "rate_per_100_steps": round(rate_per_100, 4),
        "is_plateau": is_plateau,
        "plateau_range": round(recent_range, 5),
    }

    print(f"\n  Metric: {metric} (steps {steps[0]}..{steps[-1]})")
    print(f"  Current: {values[-1]:.6f}  Best: {best_val:.6f} (step {best_step})")
    print(f"  Direction: {direction}")
    print(f"  Change: {total_change:+.2f}% total, {rate_per_100:+.3f}%/100steps")
    print(f"  Plateau: {'yes' if is_plateau else 'no'} (range: {recent_range:.4f}, threshold: {threshold})")

    if is_plateau and steps_since_best > steps_elapsed * 0.3:
        print(f"  >> Plateaued, best was {steps_since_best} steps ago — consider stopping or adjusting LR")

    return result


# ── History: raw metric export ────────────────────────────────────────────────

def history(metric: str = "train_loss", points: int = 50,
            step_from: int = 0, step_to: int | None = None,
            experiment: str | None = None, version: str | None = None) -> dict:
    """Get sampled metric history as {steps: [...], values: [...]}.

    Args:
        metric: which metric
        points: how many points to sample (more = finer resolution)
        step_from: start step
        step_to: end step
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    entries = _load_log(experiment, version)
    entries = _filter_entries(entries, step_from, step_to, metric)
    series = _extract_metric(entries, metric)
    series = _sample(series, points)

    result = {
        "metric": metric,
        "points": len(series),
        "steps": [s for s, _ in series],
        "values": [v for _, v in series],
    }

    if series:
        values = result["values"]
        result["min"] = min(values)
        result["max"] = max(values)
        result["current"] = values[-1]

    return result


# ── Compare versions ──────────────────────────────────────────────────────────

def compare(v1: str, v2: str, metric: str = "train_loss",
            experiment: str | None = None):
    """Side-by-side comparison of two versions."""
    experiment = _resolve_experiment(experiment)

    # Config diff
    from vbti.logic.train.config_utils import TrainConfig
    cfg1 = TrainConfig.from_dict(load_config(experiment, v1))
    cfg2 = TrainConfig.from_dict(load_config(experiment, v2))
    diff = cfg1.diff(cfg2)

    print(f"\n  Comparing {experiment}: {v1} vs {v2}")
    print("  " + "=" * 50)

    if diff:
        print("\n  Config differences:")
        for section, fields in diff.items():
            for field, vals in fields.items():
                print(f"    {section}.{field}: {vals['old']} → {vals['new']}")
    else:
        print("\n  Configs are identical")

    # Metric comparison
    for label, version in [(v1, v1), (v2, v2)]:
        entries = _load_log(experiment, version)
        series = _extract_metric(entries, metric)
        val_series = _extract_metric(entries, "val_loss")

        print(f"\n  {label}:")
        print(f"    Steps: {entries[-1]['step'] if entries else 0}")
        if series:
            values = [v for _, v in series]
            print(f"    {metric}: {values[-1]:.4f} (best: {min(values):.4f})")
        if val_series:
            val_values = [v for _, v in val_series]
            print(f"    val_loss: {val_values[-1]:.4f} (best: {min(val_values):.4f})")


# ── Update status.json (called from engine during training) ───────────────────

def update_status(step: int, metrics: dict,
                  experiment: str | None = None, version: str | None = None):
    """Update the AI-friendly status.json during training."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    status_path = _version_dir(experiment, version) / "metrics" / "status.json"

    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
    else:
        status = {
            "experiment": experiment, "version": version,
            "status": "running",
            "history": {"steps": []},
        }

    status["step"] = step
    status["status"] = "running"

    # Update per-metric tracking
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if key not in status or not isinstance(status.get(key), dict):
            status[key] = {}
        status[key]["current"] = value

        hist_vals = status["history"].get(key, [])
        all_vals = [v for v in hist_vals if v is not None]
        if all_vals:
            is_loss = "loss" in key
            status[key]["best"] = min(all_vals + [value]) if is_loss else max(all_vals + [value])
        status[key]["trend"] = _compute_trend(
            [v for v in all_vals + [value] if v is not None]
        )

    # Append to history
    hist = status["history"]
    hist["steps"].append(step)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            hist.setdefault(key, []).append(value)

    # Downsample if needed
    if len(hist["steps"]) > 50:
        sampled_indices = sorted(set(
            [int(i * len(hist["steps"]) / 30) for i in range(30)]
            + [len(hist["steps"]) - 1]
        ))
        for k in list(hist.keys()):
            hist[k] = [hist[k][i] for i in sampled_indices if i < len(hist[k])]

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "snapshot": snapshot,
        "logs":     logs,
        "trend":    trend,
        "compare":  compare,
        "history":  history,
        "metrics":  metrics,
    })
