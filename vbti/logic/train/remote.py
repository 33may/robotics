"""
Remote training CLI — launch and manage training jobs on a remote GPU machine.

The remote machine runs lerobot-train inside a tmux session.
Local machine handles experiment versioning, config, and checkpoint storage.

Config: vbti/remote.yaml

Usage:
    # Sync a dataset to remote (once per dataset):
    python -m vbti.logic.train.remote push-data --repo_id=eternalmay33/08-merged

    # Launch training — run_name sets the output subdir (mirrors local lerobot_output_rN naming):
    python -m vbti.logic.train.remote train --run_name=lerobot_output_r3
    python -m vbti.logic.train.remote train --run_name=lerobot_output_r3 --resume_from=lerobot_output_r2/checkpoints/030000
    python -m vbti.logic.train.remote train --run_name=lerobot_output_r3 --experiment=duck_cup_smolvla --version=v007

    # Check if training is running:
    python -m vbti.logic.train.remote status

    # Pull run outputs (checkpoints + logs) back to local version dir:
    python -m vbti.logic.train.remote pull
    python -m vbti.logic.train.remote pull --checkpoint=step_010000

    # Stream live training logs:
    python -m vbti.logic.train.remote logs
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml


# ── Config loading ────────────────────────────────────────────────────────────

REMOTE_CONFIG_PATH = Path(__file__).parents[3] / "vbti" / "remote.yaml"


def _load_remote_config() -> dict:
    if not REMOTE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Remote config not found: {REMOTE_CONFIG_PATH}")
    with open(REMOTE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["local_hf_cache"] = str(Path(cfg["local_hf_cache"]).expanduser())
    return cfg


def _ssh(remote_cfg: dict, cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on remote via ssh."""
    full_cmd = [
        "sshpass", "-p", remote_cfg["password"],
        "ssh", "-o", "StrictHostKeyChecking=no",
        remote_cfg["host"], cmd
    ]
    if capture:
        return subprocess.run(full_cmd, capture_output=True, text=True)
    else:
        return subprocess.run(full_cmd)


def _rsync_to_remote(remote_cfg: dict, local_path: str, remote_path: str, delete: bool = False):
    """Rsync local → remote."""
    cmd = [
        "sshpass", "-p", remote_cfg["password"],
        "rsync", "-avzL", "--progress", "--partial",
        "-e", "ssh -o StrictHostKeyChecking=no",
    ]
    if delete:
        cmd.append("--delete")
    cmd += [local_path, f"{remote_cfg['host']}:{remote_path}"]
    subprocess.run(cmd, check=True)


def _rsync_from_remote(remote_cfg: dict, remote_path: str, local_path: str):
    """Rsync remote → local."""
    cmd = [
        "sshpass", "-p", remote_cfg["password"],
        "rsync", "-avz", "--progress",
        "-e", "ssh -o StrictHostKeyChecking=no",
        f"{remote_cfg['host']}:{remote_path}", local_path,
    ]
    subprocess.run(cmd, check=True)


# ── Experiment utils ──────────────────────────────────────────────────────────

def _resolve_version_dir(experiment: str | None, version: str | None) -> Path:
    from vbti.logic.train.experiment_utils import active, use, get_version_dir
    if experiment and version:
        use(experiment, version)
    elif experiment or version:
        raise ValueError("Provide both --experiment and --version, or neither (uses active)")
    return get_version_dir()


# ── Commands ──────────────────────────────────────────────────────────────────

def push_data(repo_id: str, remote_cfg: dict = None):
    """Rsync a local dataset to the remote machine.

    The dataset is expected at: {local_hf_cache}/{org}/{name}/
    It lands at:                {remote.data_dir}/{org}/{name}/

    Example:
        python -m vbti.logic.train.remote push-data --repo_id=eternalmay33/08-merged
    """
    if remote_cfg is None:
        remote_cfg = _load_remote_config()

    local_root = Path(remote_cfg["local_hf_cache"])
    local_dataset = local_root / repo_id
    if not local_dataset.exists():
        raise FileNotFoundError(f"Dataset not found locally: {local_dataset}")

    # Mirror the org/name structure on remote
    org, name = repo_id.split("/", 1)
    remote_org_dir = f"{remote_cfg['data_dir']}/{org}"

    # Ensure remote org dir exists
    _ssh(remote_cfg, f"mkdir -p {remote_org_dir}")

    print(f"Syncing {local_dataset} → remote:{remote_org_dir}/{name}")
    _rsync_to_remote(remote_cfg, str(local_dataset) + "/", f"{remote_org_dir}/{name}/")
    print(f"Done. Remote path: {remote_org_dir}/{name}")


def train(
    run_name: str = "lerobot_output",
    resume_from: str = None,
    experiment: str = None,
    version: str = None,
    config: str = None,
    notes: str = "",
    dry_run: bool = False,
):
    """Launch training on remote machine in a tmux session.

    Creates {remote_version_dir}/{run_name}/ mirroring local lerobot_output_rN structure.
    Auto-pushes dataset if not present on remote.
    If resume_from is given (relative path from version dir, e.g. lerobot_output_r2/checkpoints/030000),
    pushes that checkpoint to remote and uses it as --policy.path.

    A remote_session.json is saved to the local version dir so `pull`, `status`, `logs` know where to look.

    Example:
        python -m vbti.logic.train.remote train --run_name=lerobot_output_r3
        python -m vbti.logic.train.remote train --run_name=lerobot_output_r3 --resume_from=lerobot_output_r2/checkpoints/030000
    """
    from vbti.logic.train.engine import _build_lerobot_command, resolve_config_and_version
    from vbti.logic.train.experiment_utils import active

    remote_cfg = _load_remote_config()

    # ── Resolve config + version ──────────────────────────────────────────────
    cfg, version_id, version_dir = resolve_config_and_version(config, experiment, version, notes)
    exp_name, _ = active()

    # ── Remote paths ──────────────────────────────────────────────────────────
    repo_id = cfg.dataset.sources[0].repo_id
    org, name = repo_id.split("/", 1)
    remote_data_root = f"{remote_cfg['data_dir']}/{org}/{name}"
    remote_version_dir = f"{remote_cfg['experiments_dir']}/{exp_name}/{version_id}"
    remote_run_dir = f"{remote_version_dir}/{run_name}"
    remote_ckpt_dir = f"{remote_run_dir}/checkpoints"
    job_name = f"{exp_name}_{version_id}_{run_name}"
    tmux_session = f"train_{exp_name}_{version_id}_{run_name}".replace("/", "_")
    log_file = f"{remote_version_dir}/train_{run_name}.log"

    # ── Build lerobot-train command ───────────────────────────────────────────
    args = _build_lerobot_command(cfg, remote_run_dir, job_name)

    # Substitute remote dataset root
    args = [a for a in args if not a.startswith("--dataset.root=")]
    args.append(f"--dataset.root={remote_data_root}")

    # Auto-detect local pretrained path and rewrite for remote
    local_pretrained = Path(cfg.model.pretrained)
    if local_pretrained.exists() and local_pretrained.is_dir():
        remote_pretrained_dir = f"{remote_version_dir}/pretrained_base"
        args = [a for a in args if not a.startswith("--policy.path=")]
        args.append(f"--policy.path={remote_pretrained_dir}")
        # Will be rsynced below alongside resume checkpoint

    # Resume: swap --policy.path to point at the remote checkpoint
    if resume_from:
        remote_resume_path = f"{remote_version_dir}/{resume_from}/pretrained_model"
        args = [a for a in args if not a.startswith("--policy.path=")]
        args.append(f"--policy.path={remote_resume_path}")

    # Escape inner double quotes so the command survives bash -c "..." inside tmux
    def _shell_quote(a: str) -> str:
        if " " in a or '"' in a:
            return "'" + a.replace("'", "'\\''") + "'"
        return a

    lerobot_cmd = " ".join(_shell_quote(a) for a in args)

    print(f"\nExperiment:   {exp_name} / {version_id}")
    print(f"Run:          {run_name}")
    print(f"Remote dir:   {remote_run_dir}")
    print(f"Tmux session: {tmux_session}")
    if resume_from:
        print(f"Resume from:  {resume_from}")
    print(f"\nCommand:\n  {lerobot_cmd}\n")

    # ── Receipt ───────────────────────────────────────────────────────────────
    session_info = {
        "run_name": run_name,
        "tmux_session": tmux_session,
        "remote_version_dir": remote_version_dir,
        "remote_run_dir": remote_run_dir,
        "remote_ckpt_dir": remote_ckpt_dir,
        "remote_data_root": remote_data_root,
        "job_name": job_name,
        "repo_id": repo_id,
        "host": remote_cfg["host"],
    }
    receipt_path = version_dir / "remote_session.json"

    if dry_run:
        print("(dry run — not launching)")
        print(f"\nWould save receipt to: {receipt_path}")
        print(json.dumps(session_info, indent=2))
        return

    # ── Ensure dataset is complete on remote ─────────────────────────────────
    local_dataset = Path(remote_cfg["local_hf_cache"]) / repo_id
    local_count = sum(1 for _ in local_dataset.rglob("*") if _.is_file())
    result = _ssh(remote_cfg, f"find {remote_data_root} -type f | wc -l", capture=True)
    remote_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0

    if remote_count < local_count:
        print(f"Dataset incomplete on remote ({remote_count}/{local_count} files) — syncing...")
        push_data(repo_id, remote_cfg)
    else:
        print(f"Dataset complete on remote ({remote_count} files)")

    # ── Push local pretrained model if needed ──────────────────────────────
    if not resume_from and local_pretrained.exists() and local_pretrained.is_dir():
        print(f"Pushing local pretrained model → remote:{remote_pretrained_dir}")
        _ssh(remote_cfg, f"mkdir -p {remote_pretrained_dir}")
        _rsync_to_remote(remote_cfg, str(local_pretrained) + "/", remote_pretrained_dir + "/")

    # ── Push resume checkpoint if needed ─────────────────────────────────────
    if resume_from:
        local_resume = version_dir / resume_from
        if not local_resume.exists():
            print(f"[ERR] resume_from path not found locally: {local_resume}")
            sys.exit(1)
        remote_resume_dir = f"{remote_version_dir}/{resume_from}"
        print(f"Pushing resume checkpoint → remote:{remote_resume_dir}")
        _ssh(remote_cfg, f"mkdir -p {remote_resume_dir}")
        _rsync_to_remote(remote_cfg, str(local_resume) + "/", remote_resume_dir + "/")

    # ── Copy config to version dir (don't create run_dir — lerobot-train does that) ─
    _ssh(remote_cfg, f"mkdir -p {remote_version_dir}")
    _rsync_to_remote(remote_cfg, str(version_dir / "config.yaml"), f"{remote_version_dir}/config.yaml")

    # ── Write a train script on remote and launch it in tmux ─────────────────
    # Writing to a script avoids all shell quoting issues with complex args (e.g. rename_map JSON)
    train_script = f"{remote_version_dir}/train_{run_name}.sh"
    script_lines = [
        "#!/bin/bash",
        "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}",
        "source /home/vbti/anton/env/bin/activate",
        lerobot_cmd,
    ]
    script_content = "\n".join(script_lines)

    # Write script to remote via heredoc
    write_script_cmd = f"cat > {train_script} << 'TRAIN_EOF'\n{script_content}\nTRAIN_EOF\nchmod +x {train_script}"
    _ssh(remote_cfg, write_script_cmd)

    tmux_launch = (
        f"tmux new-session -d -s {tmux_session} "
        f"'bash -c \"{train_script} 2>&1 | tee {log_file}; echo EXIT_CODE=$? >> {log_file}\"'"
    )

    _ssh(remote_cfg, f"tmux kill-session -t {tmux_session} 2>/dev/null || true")
    result = _ssh(remote_cfg, tmux_launch)
    if result.returncode != 0:
        print("[ERR] Failed to launch training session")
        sys.exit(1)

    receipt_path.write_text(json.dumps(session_info, indent=2))
    print(f"Training launched. Streaming logs (Ctrl+C to detach)...\n")

    # Stream logs immediately
    tail_cmd = [
        "sshpass", "-p", remote_cfg["password"],
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-t", remote_cfg["host"],
        f"tail -n 50 -f --retry {log_file}"
    ]
    try:
        subprocess.run(tail_cmd)
    except KeyboardInterrupt:
        print("\n\nDetached from logs.")
        print(f"  Re-attach: python -m vbti.logic.train.remote logs")
        print(f"  Pull:      python -m vbti.logic.train.remote pull")


def status(experiment: str = None, version: str = None):
    """Check if remote training session is running.

    Example:
        python -m vbti.logic.train.remote status
    """
    remote_cfg = _load_remote_config()
    version_dir = _resolve_version_dir(experiment, version)
    receipt_path = version_dir / "remote_session.json"

    if not receipt_path.exists():
        print("No remote_session.json found — has training been launched?")
        return

    session = json.loads(receipt_path.read_text())
    tmux_session = session["tmux_session"]

    result = _ssh(remote_cfg, f"tmux ls 2>/dev/null", capture=True)
    running = tmux_session in result.stdout

    print(f"Session: {tmux_session}")
    print(f"Status:  {'RUNNING' if running else 'NOT RUNNING'}")

    if running:
        log_file = f"{session.get('remote_run_dir', session['remote_version_dir'])}/train.log"
        result = _ssh(remote_cfg, f"tail -5 {log_file} 2>/dev/null", capture=True)
        if result.stdout.strip():
            print(f"\nLast log lines:\n{result.stdout}")


def logs(experiment: str = None, version: str = None, lines: int = 50):
    """Stream live training logs from remote.

    Example:
        python -m vbti.logic.train.remote logs
        python -m vbti.logic.train.remote logs --lines=100
    """
    remote_cfg = _load_remote_config()
    version_dir = _resolve_version_dir(experiment, version)
    receipt_path = version_dir / "remote_session.json"

    if not receipt_path.exists():
        print("No remote_session.json — has training been launched?")
        return

    session = json.loads(receipt_path.read_text())
    log_file = f"{session.get('remote_run_dir', session['remote_version_dir'])}/train.log"

    print(f"Tailing {log_file} (Ctrl+C to stop)\n")
    cmd = [
        "sshpass", "-p", remote_cfg["password"],
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-t", remote_cfg["host"],
        f"tail -n {lines} -f --retry {log_file}"
    ]
    subprocess.run(cmd)


def pull(
    experiment: str = None,
    version: str = None,
    checkpoint: str = "all",
):
    """Pull run outputs (checkpoints + logs) from remote → local version dir/{run_name}/.

    Args:
        checkpoint: "all" (default) syncs entire run dir, or specific step e.g. "step_010000"

    Example:
        python -m vbti.logic.train.remote pull
        python -m vbti.logic.train.remote pull --checkpoint=step_010000
    """
    remote_cfg = _load_remote_config()
    version_dir = _resolve_version_dir(experiment, version)
    receipt_path = version_dir / "remote_session.json"

    if not receipt_path.exists():
        print("No remote_session.json — don't know where to pull from.")
        print("Re-run with --experiment and --version, or ensure the receipt exists.")
        return

    session = json.loads(receipt_path.read_text())
    run_name = session.get("run_name", "lerobot_output")
    remote_run_dir = session.get("remote_run_dir") or f"{session['remote_version_dir']}/{run_name}"
    remote_ckpt_dir = session.get("remote_ckpt_dir") or f"{remote_run_dir}/checkpoints"
    local_run_dir = version_dir / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint == "all":
        print(f"Pulling {run_name}/ from remote → {local_run_dir}")
        _rsync_from_remote(remote_cfg, remote_run_dir + "/", str(local_run_dir) + "/")
    else:
        ckpt_name = checkpoint.replace("step_", "") if checkpoint.startswith("step_") else checkpoint
        remote_src = f"{remote_ckpt_dir}/{ckpt_name}/"
        local_dst = local_run_dir / "checkpoints" / ckpt_name
        local_dst.mkdir(parents=True, exist_ok=True)
        print(f"Pulling checkpoint '{ckpt_name}' from remote")
        _rsync_from_remote(remote_cfg, remote_src, str(local_dst) + "/")

    print(f"Done. Local run dir: {local_run_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "push-data": push_data,
        "train":     train,
        "status":    status,
        "logs":      logs,
        "pull":      pull,
    })
