"""
Remote Sequential Training Chain.

Generates a single bash script that runs all versions sequentially on the remote,
pushes it via SSH, and launches it in a detached tmux session.

Features:
- "Fire and forget" execution entirely on the remote machine.
- Fault-tolerant: If one version crashes, it logs the failure and moves to the next.
- Scheduling: Can wait for an existing tmux session to finish before starting,
  with a hard fallback timeout to start anyway if the previous run hangs.

Usage:
    # Run immediately:
    python -m vbti.logic.train.chain_remote --versions v021,v022,v023

    # Schedule to wait for an existing run to finish first (4 hour max wait):
    python -m vbti.logic.train.chain_remote --versions v021,v022 --wait_for_session train_duck_cup_smolvla_v020_lerobot_output
"""

import argparse
import json
import os
import sys
from pathlib import Path

from vbti.logic.train.chain import derive_from_config, MIN_SAFETENSORS_BYTES
from vbti.logic.train.engine import _build_lerobot_command, resolve_config_and_version
from vbti.logic.train.experiment_utils import active
from vbti.logic.train.remote import _load_remote_config, _ssh, _rsync_to_remote, push_data


def _shell_quote(a: str) -> str:
    if " " in a or '"' in a:
        return "'" + a.replace("'", "'\\''") + "'"
    return a


def run_remote_chain(
    versions: list[str], 
    run_name: str = "lerobot_output_r1", 
    expandable_segments: bool = True,
    wait_for_session: str | None = None
):
    remote_cfg = _load_remote_config()
    exp_name, _ = active()
    
    remote_exp_dir = f"{remote_cfg['experiments_dir']}/{exp_name}"
    chain_script = f"{remote_exp_dir}/chain_{run_name}.sh"
    log_file = f"{remote_exp_dir}/chain_{run_name}.log"
    tmux_session = f"chain_{exp_name}_{run_name}".replace("/", "_")

    # ── 1. Init the bash script (Notice: no 'set -e' to ensure fault tolerance) ─
    script_lines = [
        "#!/bin/bash",
        f"export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${{LD_LIBRARY_PATH:-}}",
        "source /home/vbti/anton/env/bin/activate",
        "",
        f"exec > >(tee -a {log_file}) 2>&1",
        f"echo '===================================================='",
        f"echo 'CHAIN SCRIPT INITIATED: $(date)'",
        f"echo 'VERSIONS QUEUED: {', '.join(versions)}'",
        f"echo '===================================================='",
        "",
    ]
    
    if expandable_segments:
        script_lines.insert(2, "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # ── 1.5 Inject scheduling loop with 4-hour timeout ─────────────────────────
    if wait_for_session:
        # 240 checks * 60 seconds = 4 hours
        script_lines.extend([
            f"echo '⏳ SCHEDULING: Waiting for active tmux session \"{wait_for_session}\" to finish...'",
            f"WAIT_CHECKS=0",
            f"MAX_CHECKS=240", # 4 hours max wait (240 mins)
            f"while tmux has-session -t {wait_for_session} 2>/dev/null; do",
            f"  if [ $WAIT_CHECKS -ge $MAX_CHECKS ]; then",
            f"    echo '⚠️ TIMEOUT REACHED! 4 hours passed. Forcing start of new chain at $(date).'",
            f"    break",
            f"  fi",
            f"  sleep 60", # Check every minute
            f"  WAIT_CHECKS=$((WAIT_CHECKS+1))",
            f"done",
            f"if [ $WAIT_CHECKS -lt $MAX_CHECKS ]; then",
            f"  echo '✅ Target session \"{wait_for_session}\" finished normally! Proceeding with chain at $(date).'",
            f"fi",
            f"echo '===================================================='",
            "",
        ])

    # ── 2. Process each version locally & append to script ────────────────────
    print(f"Preparing chain for {len(versions)} versions: {versions}")
    for v in versions:
        print(f"\n[chain] Validating and syncing {v}...")
        cfg, version_id, version_dir = resolve_config_and_version(config=None, experiment=exp_name, version=v, notes="")
        derived = derive_from_config(v)

        repo_id = derived["repo_id"]
        remote_data_root = f"{remote_cfg['data_dir']}/{repo_id.split('/')[0]}/{repo_id.split('/')[1]}"
        remote_version_dir = f"{remote_exp_dir}/{version_id}"
        remote_run_dir = f"{remote_version_dir}/{run_name}"
        remote_ckpt_dir = f"{remote_run_dir}/checkpoints"
        job_name = f"{exp_name}_{version_id}_{run_name}"
        
        # 2a. Sync dataset to remote — unless it's a remote-only dataset.
        #     The UVA-baked `_uva` copy is built on the remote and never exists
        #     locally; push_data would raise FileNotFoundError. Skip the push
        #     when the dataset is already on the remote, or when it will be
        #     produced by an awaited session (--wait_for_session) before the
        #     chain script actually runs.
        local_ds = Path(remote_cfg["local_hf_cache"]) / repo_id
        if local_ds.exists():
            push_data(repo_id, remote_cfg)
        else:
            remote_ds = f"{remote_cfg['data_dir']}/{repo_id}"
            on_remote = _ssh(
                remote_cfg, f"test -d {remote_ds} && echo yes", capture=True
            ).stdout.strip() == "yes"
            if on_remote:
                print(f"[chain] {repo_id}: remote-only dataset at {remote_ds} — skipping push_data")
            elif wait_for_session:
                print(f"[chain] {repo_id}: not present yet — expected from awaited "
                      f"session '{wait_for_session}'. Skipping push_data; "
                      f"lerobot-train resolves --dataset.root at run time.")
            else:
                raise FileNotFoundError(
                    f"Dataset '{repo_id}' not found locally ({local_ds}) "
                    f"or on the remote ({remote_ds})."
                )

        # 2b. Sync local pretrained base if it exists.
        #     os.path.isdir (not Path.exists) — when `pretrained` is a remote-only
        #     absolute path, stat'ing it on the laptop can raise PermissionError
        #     (Python 3.12 propagates it); os.path.isdir swallows OSError -> False,
        #     so a remote path correctly reads as "not local" and is passed through.
        local_pretrained = Path(cfg.model.pretrained)
        remote_pretrained_dir = f"{remote_version_dir}/pretrained_base"
        pretrained_is_local = os.path.isdir(local_pretrained)
        if pretrained_is_local:
            print(f"[chain] Syncing local pretrained base → {remote_pretrained_dir}")
            _ssh(remote_cfg, f"mkdir -p {remote_pretrained_dir}")
            _rsync_to_remote(remote_cfg, str(local_pretrained) + "/", remote_pretrained_dir + "/")

        # 2c. Sync config.yaml
        _ssh(remote_cfg, f"mkdir -p {remote_version_dir}")
        _rsync_to_remote(remote_cfg, str(version_dir / "config.yaml"), f"{remote_version_dir}/config.yaml")

        # 2d. Write local receipt so `remote.py pull` works natively later
        session_info = {
            "run_name": run_name,
            "tmux_session": tmux_session,  # Master chain tmux session
            "remote_version_dir": remote_version_dir,
            "remote_run_dir": remote_run_dir,
            "remote_ckpt_dir": remote_ckpt_dir,
            "remote_data_root": remote_data_root,
            "job_name": job_name,
            "repo_id": repo_id,
            "host": remote_cfg["host"],
        }
        receipt_path = version_dir / "remote_session.json"
        receipt_path.write_text(json.dumps(session_info, indent=2))
        print(f"[chain] Wrote local receipt for {v}")

        # 2e. Build command arguments
        args = _build_lerobot_command(cfg, remote_run_dir, job_name)
        args = [a for a in args if not a.startswith("--dataset.root=")]
        args.append(f"--dataset.root={remote_data_root}")
        
        if pretrained_is_local:
            # Local base was rsynced (2b) — point --policy.path at its remote copy.
            # Otherwise --policy.path keeps whatever _build_lerobot_command set
            # (e.g. a remote-only path like /home/vbti/anton/data/smolvla_uva_base).
            args = [a for a in args if not a.startswith("--policy.path=")]
            args.append(f"--policy.path={remote_pretrained_dir}")

        cmd = " ".join(_shell_quote(a) for a in args)

        # Prepend inline env variables for lerobot-train (e.g. gradient checkpointing)
        env_prefix = ""
        if not getattr(cfg.training, "gradient_checkpoint", True):
            env_prefix = "VBTI_GRAD_CHECKPOINT=0 "

        # 2f. Append the execution & fault-tolerant predicate check
        final_step = derived["final_step"]
        expected_safetensors = f"{remote_run_dir}/checkpoints/{final_step:06d}/pretrained_model/model.safetensors"

        script_lines.extend([
            f"echo ''",
            f"echo '=== {v} STARTING: $(date) ==='",
            f"{env_prefix}{cmd}",
            f"echo '=== {v} FINISHED TRAINING, CHECKING PREDICATE ==='",
            f"if [ ! -f \"{expected_safetensors}\" ] || [ $(stat -c%s \"{expected_safetensors}\") -lt {MIN_SAFETENSORS_BYTES} ]; then",
            f"  echo '❌ FAIL: {v} predicate failed! Model missing or too small. Moving to next version.'",
            f"else",
            f"  echo '✅ SUCCESS: {v} PASSED PREDICATE: $(date)'",
            f"fi",
        ])

    # ── 3. Finalize and push the script ───────────────────────────────────────
    script_lines.extend([
        "",
        f"echo '===================================================='",
        f"echo 'CHAIN SCRIPT FULLY COMPLETED: $(date)'",
        f"echo '===================================================='",
    ])
    script_content = "\n".join(script_lines)

    print(f"\n[chain] Pushing execution script to remote: {chain_script}")
    write_script_cmd = f"cat > {chain_script} << 'TRAIN_EOF'\n{script_content}\nTRAIN_EOF\nchmod +x {chain_script}"
    _ssh(remote_cfg, write_script_cmd)

    # ── 4. Launch Tmux ────────────────────────────────────────────────────────
    tmux_launch = (
        f"tmux new-session -d -s {tmux_session} "
        f"'bash -c \"{chain_script}\"'"
    )
    
    print(f"[chain] Killing old chain tmux session if exists: {tmux_session}")
    _ssh(remote_cfg, f"tmux kill-session -t {tmux_session} 2>/dev/null || true")
    
    print(f"[chain] Launching remote chain in tmux...")
    result = _ssh(remote_cfg, tmux_launch)
    if result.returncode != 0:
        print("[ERR] Failed to launch training session on remote")
        sys.exit(1)

    print(f"\n✅ Remote chain successfully dispatched!")
    if wait_for_session:
        print(f"⏳ It is currently paused and waiting for '{wait_for_session}' to finish.")
        print(f"⏰ If it doesn't finish within 4 hours, it will force-start anyway.")
    else:
        print(f"🚀 It is running now.")
        
    print(f"Log file on remote: {log_file}")
    print(f"Tmux session:       {tmux_session}")
    print(f"\nWhen you return, pull your results with:")
    print(f"  python -m vbti.logic.train.remote pull --checkpoint=all --run_name={run_name} --version=vXXX")


def main():
    p = argparse.ArgumentParser(description="Generate and deploy a sequential remote training chain.")
    p.add_argument("--versions", required=True, help="comma-separated, e.g. v021,v022,v023,v024")
    p.add_argument("--run_name", default="lerobot_output_r1", help="directory name for remote logs/checkpoints")
    p.add_argument("--wait_for_session", default=None, help="If provided, wait for this tmux session to close before starting")
    args = p.parse_args()
    
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    run_remote_chain(versions, args.run_name, wait_for_session=args.wait_for_session)


if __name__ == "__main__":
    main()