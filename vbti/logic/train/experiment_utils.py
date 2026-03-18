"""
Experiment management utilities.

Name-based API with active experiment context. State tracked in
experiments/state.json so both humans and AI agents can interact naturally.

    experiments/
    ├── state.json                          # active experiment + version
    └── {experiment_name}/
        ├── experiment.md
        ├── base_config.yaml
        ├── v001/
        │   ├── config.yaml
        │   ├── run.md
        │   ├── notes.md
        │   ├── metrics/
        │   │   ├── training_log.jsonl
        │   │   └── summary.json
        │   ├── checkpoints/
        │   │   ├── best/
        │   │   └── final/
        │   └── eval/
        │       ├── sim_results.json
        │       └── videos/
        ├── v002/
        └── compare.md

Usage:
    create_experiment("lift_cube_smolvla", hypothesis="SmolVLA can learn lift from 50 demos")
    use("lift_cube_smolvla")
    create_version(config={...}, notes="baseline LR 1e-5")
    log_metrics(step=100, metrics={"train_loss": 0.42})
    complete_version()
"""

from pathlib import Path
from datetime import datetime
import json
import yaml


EXPERIMENTS_ROOT = Path("~/projects/ml_portfolio/robotics/vbti/experiments").expanduser()
STATE_FILE = EXPERIMENTS_ROOT / "state.json"


# ── State management ──────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"active_experiment": None, "active_version": None}


def _save_state(state: dict):
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def use(experiment_name: str, version: str | None = None):
    """Set the active experiment (and optionally version)."""
    exp_dir = EXPERIMENTS_ROOT / experiment_name
    if not exp_dir.exists():
        raise ValueError(f"Experiment '{experiment_name}' does not exist")
    if version:
        if not (exp_dir / version).exists():
            raise ValueError(f"Version '{version}' not found in '{experiment_name}'")
    state = _load_state()
    state["active_experiment"] = experiment_name
    state["active_version"] = version
    _save_state(state)
    print(f"Active: {experiment_name}" + (f" / {version}" if version else ""))


def active() -> tuple[str | None, str | None]:
    """Return (active_experiment, active_version)."""
    state = _load_state()
    return state.get("active_experiment"), state.get("active_version")


def _resolve_experiment(name: str | None = None) -> str:
    """Resolve experiment name — use active if not provided."""
    if name:
        return name
    exp, _ = active()
    if not exp:
        raise ValueError("No active experiment. Call use('name') or pass experiment name.")
    return exp


def _resolve_version(experiment: str, version: str | None = None) -> str:
    """Resolve version — use active if not provided."""
    if version:
        return version
    _, v = active()
    if not v:
        raise ValueError(f"No active version. Call use('{experiment}', 'v001') or pass version.")
    return v


def _experiment_dir(name: str) -> Path:
    return EXPERIMENTS_ROOT / name


def _version_dir(experiment: str, version: str) -> Path:
    return EXPERIMENTS_ROOT / experiment / version


# ── Experiment CRUD ───────────────────────────────────────────────────────────

def create_experiment(name: str, hypothesis: str = "", goal: str = "") -> str:
    """Create a new experiment. Returns the experiment name."""
    exp_dir = _experiment_dir(name)

    if exp_dir.exists():
        print(f"Experiment '{name}' already exists")
        return name

    exp_dir.mkdir(parents=True)

    md_content = f"""# {name}

**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Hypothesis
{hypothesis or "TODO"}

## Goal
{goal or "TODO"}

## Versions
| Version | Date | Status | Val Loss | Notes |
|---------|------|--------|----------|-------|
"""
    (exp_dir / "experiment.md").write_text(md_content)
    (exp_dir / "base_config.yaml").write_text("# Base config shared across versions\n")
    (exp_dir / "compare.md").write_text(f"# {name} — Version Comparison\n\n")

    print(f"Created experiment: {name}")
    return name


def list_experiments() -> list[str]:
    """List all experiment names."""
    if not EXPERIMENTS_ROOT.exists():
        return []
    return sorted(
        d.name for d in EXPERIMENTS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


# ── Version CRUD ──────────────────────────────────────────────────────────────

def _get_next_version(experiment: str) -> str:
    """Return the next version string (v001, v002, ...)."""
    exp_dir = _experiment_dir(experiment)
    existing = sorted(
        d.name for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    )
    if not existing:
        return "v001"
    last_num = int(existing[-1][1:])
    return f"v{last_num + 1:03d}"


def create_version(config: dict | str, notes: str = "", experiment: str | None = None) -> str:
    """Create a new version in the experiment. Sets it as active. Returns version id.

    config can be:
        - dict: raw config data
        - "base": load base_config.yaml from the experiment folder
        - str path: load from YAML file
    """
    experiment = _resolve_experiment(experiment)

    # Resolve config shortcuts
    if isinstance(config, str):
        if config == "base":
            base_path = _experiment_dir(experiment) / "base_config.yaml"
            with open(base_path) as f:
                config = yaml.safe_load(f)
        else:
            with open(config) as f:
                config = yaml.safe_load(f)
    version_id = _get_next_version(experiment)
    version_dir = _version_dir(experiment, version_id)
    version_dir.mkdir()

    # Subdirectories
    (version_dir / "metrics").mkdir()
    (version_dir / "checkpoints" / "best").mkdir(parents=True)
    (version_dir / "checkpoints" / "final").mkdir(parents=True)
    (version_dir / "eval" / "videos").mkdir(parents=True)

    # Frozen config snapshot
    with open(version_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # run.md
    run_md = f"""# {version_id}

**Started**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status**: running
**Hardware**: TODO
**Duration**: —
"""
    (version_dir / "run.md").write_text(run_md)

    # notes.md
    notes_md = f"""# {version_id} — Notes

{notes or "TODO: why this run, what changed from previous version"}
"""
    (version_dir / "notes.md").write_text(notes_md)

    # Set as active version
    use(experiment, version_id)

    print(f"Created version: {experiment}/{version_id}")
    return version_id


def list_versions(experiment: str | None = None) -> list[str]:
    """List all versions for an experiment."""
    experiment = _resolve_experiment(experiment)
    exp_dir = _experiment_dir(experiment)
    return sorted(
        d.name for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    )


# ── Metrics & lifecycle ───────────────────────────────────────────────────────

def log_metrics(step: int, metrics: dict, experiment: str | None = None, version: str | None = None):
    """Append a metrics entry to training_log.jsonl."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    log_path = _version_dir(experiment, version) / "metrics" / "training_log.jsonl"
    entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_summary(summary: dict, experiment: str | None = None, version: str | None = None):
    """Write the final metrics summary."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    summary_path = _version_dir(experiment, version) / "metrics" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def complete_version(status: str = "completed", duration_str: str = "",
                     experiment: str | None = None, version: str | None = None):
    """Mark a version as finished — updates run.md."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    run_md_path = _version_dir(experiment, version) / "run.md"
    content = run_md_path.read_text()
    content = content.replace("**Status**: running", f"**Status**: {status}")
    if duration_str:
        content = content.replace("**Duration**: —", f"**Duration**: {duration_str}")
    content += f"\n**Finished**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    run_md_path.write_text(content)
    print(f"Completed: {experiment}/{version} — {status}")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_config(experiment: str | None = None, version: str | None = None) -> dict:
    """Load the frozen config for a version."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    with open(_version_dir(experiment, version) / "config.yaml") as f:
        return yaml.safe_load(f)


def load_summary(experiment: str | None = None, version: str | None = None) -> dict | None:
    """Load the metrics summary, or None if not yet available."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    summary_path = _version_dir(experiment, version) / "metrics" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def get_version_dir(experiment: str | None = None, version: str | None = None) -> Path:
    """Get the path to a version directory (for checkpoint access etc)."""
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    return _version_dir(experiment, version)


def list_checkpoints(experiment: str | None = None, version: str | None = None,
                     include_named: bool = True) -> list[dict]:
    """List all checkpoints for a version, sorted by step number.

    Returns list of dicts: [{"name": "step_001000", "step": 1000, "path": Path(...)}, ...]
    Named checkpoints (best, final) are appended at the end if include_named=True.
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    ckpt_dir = _version_dir(experiment, version) / "checkpoints"

    if not ckpt_dir.exists():
        return []

    step_ckpts = []
    named_ckpts = []

    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue
        # Must have model.safetensors to be a valid checkpoint
        if not (d / "model.safetensors").exists():
            continue

        if d.name.startswith("step_") and d.name[5:].isdigit():
            step_ckpts.append({
                "name": d.name,
                "step": int(d.name[5:]),
                "path": d,
            })
        elif include_named:
            named_ckpts.append({
                "name": d.name,
                "step": None,
                "path": d,
            })

    step_ckpts.sort(key=lambda c: c["step"])
    named_ckpts.sort(key=lambda c: c["name"])
    return step_ckpts + named_ckpts


def print_checkpoints(experiment: str | None = None, version: str | None = None):
    """Print all checkpoints for a version in a readable format."""
    ckpts = list_checkpoints(experiment, version)
    if not ckpts:
        print("No checkpoints found.")
        return
    for c in ckpts:
        print(f"---")
        print(f"  name: {c['name']}")
        print(f"  path: {c['path']}")
    print("---")


def resolve_checkpoint(checkpoint: str, experiment: str | None = None,
                       version: str | None = None) -> list[Path]:
    """Resolve a checkpoint specifier to one or more paths.

    checkpoint can be:
        - "all": all step checkpoints (no best/final)
        - "best", "final", "step_002000": specific checkpoint by name
        - "2000": shorthand for step_002000
    """
    experiment = _resolve_experiment(experiment)
    version = _resolve_version(experiment, version)
    ckpt_dir = _version_dir(experiment, version) / "checkpoints"

    if checkpoint == "all":
        ckpts = list_checkpoints(experiment, version, include_named=False)
        if not ckpts:
            raise ValueError(f"No checkpoints found in {ckpt_dir}")
        return [c["path"] for c in ckpts]

    # Direct name match (best, final, step_002000)
    direct = ckpt_dir / checkpoint
    if direct.exists() and (direct / "model.safetensors").exists():
        return [direct]

    # Shorthand: "2000" -> "step_002000"
    if checkpoint.isdigit():
        step_name = f"step_{int(checkpoint):06d}"
        step_path = ckpt_dir / step_name
        if step_path.exists() and (step_path / "model.safetensors").exists():
            return [step_path]

    raise ValueError(f"Checkpoint '{checkpoint}' not found in {ckpt_dir}")


def status(experiment: str | None = None) -> dict:
    """Get a quick overview: active experiment, version, all versions with status."""
    experiment = _resolve_experiment(experiment)
    versions = list_versions(experiment)
    overview = {"experiment": experiment, "versions": []}
    for v in versions:
        run_md = _version_dir(experiment, v) / "run.md"
        s = "unknown"
        if run_md.exists():
            for line in run_md.read_text().splitlines():
                if line.startswith("**Status**:"):
                    s = line.split(":", 1)[1].strip()
                    break
        overview["versions"].append({"version": v, "status": s})
    return overview


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "create":     create_experiment,
        "use":        use,
        "active":     active,
        "version":    create_version,
        "lsv":        list_versions,
        "lse":        list_experiments,
        "status":     status,
        "complete":   complete_version,
        "log":        log_metrics,
        "summary":    save_summary,
        "config":     load_config,
        "dir":        get_version_dir,
        "ckpts":      print_checkpoints,
    })
