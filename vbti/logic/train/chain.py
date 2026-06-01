"""Sequential training chain — orchestrator only.

Given a list of versions whose configs already exist on disk, train them on remote
back-to-back. Halt on first failure of the done-predicate. Pulls checkpoints to
local after each version succeeds.

Does NOT generate configs. Each version directory must already have a `config.yaml`
written by a sweep-specific script (e.g. `scripts/gen_data_eff_configs.py`).

Per-version flow:
  1. Load <version>/config.yaml → derive expected_final_step
  2. Launch remote training (non-blocking tmux)
  3. Poll tmux until it dies
  4. Rsync checkpoints to local
  5. Check the done-predicate; halt if it fails

Usage:
    python -m vbti.logic.train.chain --versions v021,v022,v023,v024
    nohup python -m vbti.logic.train.chain --versions v021 > chain.log 2>&1 &
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from vbti.logic.train.remote import _load_remote_config, _rsync_from_remote, _ssh


# ── Constants ────────────────────────────────────────────────────────────────

EXPERIMENT      = "duck_cup_smolvla"
REPO_ROOT       = Path("/home/may33/projects/ml_portfolio/robotics")
EXP_ROOT        = REPO_ROOT / "vbti/experiments" / EXPERIMENT

POLL_INTERVAL_SEC = 300   # 5 min — overridable via --poll_interval

# SmolVLA bf16 model.safetensors is 906,712,520 bytes in v020 (verified across all
# ckpts). 800 MB floor = ~12% margin below the true size; catches torn/empty/corrupt
# files without being brittle to a small lerobot save-format addition.
MIN_SAFETENSORS_BYTES: int = 800_000_000


# ── Derivation helpers (read from per-version config) ────────────────────────

def n_frames_for(repo_id: str, episodes: list[int] | None) -> int:
    """Sum frame counts over the kept episodes. If episodes is None, returns total."""
    import pyarrow.parquet as pq
    meta = Path.home() / ".cache/huggingface/lerobot" / repo_id / \
           "meta/episodes/chunk-000/file-000.parquet"
    df = pq.read_table(meta, columns=["episode_index", "length"]).to_pandas()
    if episodes is None:
        return int(df["length"].sum())
    return int(df[df.episode_index.isin(episodes)].length.sum())


def expected_final_step(total_steps: int, save_freq: int) -> int:
    """Largest save_freq-aligned step ≤ total_steps."""
    return (total_steps // save_freq) * save_freq


def derive_from_config(version: str) -> dict:
    """Read <version>/config.yaml and return everything the chain needs to know."""
    config_path = EXP_ROOT / version / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    src = cfg["dataset"]["sources"][0]
    repo_id   = src["repo_id"]
    episodes  = src.get("episodes")

    epochs    = cfg["training"].get("epochs")
    bs        = cfg["training"]["batch_size"]
    save_freq = cfg["logging"]["save_freq"]
    if epochs is not None:
        # epochs-based: must count frames, which needs the dataset cached locally
        n_frames    = n_frames_for(repo_id, episodes)
        total_steps = int(epochs * n_frames / bs)
    else:
        # explicit `steps`: no frame count needed — works for datasets that
        # exist only on the remote (e.g. the UVA-baked _uva copy).
        n_frames    = None
        total_steps = int(cfg["training"]["steps"])
    final_step  = expected_final_step(total_steps, save_freq)

    return {
        "repo_id":     repo_id,
        "episodes":    episodes,
        "n_episodes":  len(episodes) if episodes else None,
        "n_frames":    n_frames,
        "epochs":      epochs,
        "batch_size":  bs,
        "save_freq":   save_freq,
        "total_steps": total_steps,
        "final_step":  final_step,
    }


# ── Remote orchestration ─────────────────────────────────────────────────────

def launch_remote(version: str, run_name: str) -> dict:
    """Invoke remote.py train via subprocess. Returns parsed remote_session.json."""
    cmd = [
        sys.executable, "-m", "vbti.logic.train.remote", "train",
        f"--version={version}",
        f"--run_name={run_name}",
        "--stream=false",   # non-blocking launch; chain owns polling
    ]
    print(f"\n[chain] launch: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    receipt = EXP_ROOT / version / "remote_session.json"
    return json.loads(receipt.read_text())


def tmux_alive(remote_cfg: dict, session: str) -> bool:
    r = _ssh(remote_cfg, f"tmux has-session -t {session} 2>/dev/null && echo alive",
             capture=True)
    return r.stdout.strip() == "alive"


def wait_until_dead(remote_cfg: dict, session: str, version: str, run_name: str,
                    poll_interval: int = POLL_INTERVAL_SEC):
    """Poll every poll_interval sec. Log latest local checkpoint name while waiting."""
    ckpt_dir = EXP_ROOT / version / run_name / "checkpoints"
    while tmux_alive(remote_cfg, session):
        if ckpt_dir.exists():
            ckpts = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
            latest = ckpts[-1] if ckpts else "—"
        else:
            latest = "—"
        print(f"  [chain] {time.strftime('%H:%M:%S')} {version} alive | latest local ckpt={latest}",
              flush=True)
        time.sleep(poll_interval)
    print(f"  [chain] {time.strftime('%H:%M:%S')} {version} tmux dead", flush=True)


def pull_checkpoints(remote_cfg: dict, version: str, run_name: str, session_info: dict):
    """Rsync remote checkpoints/ to local."""
    remote_ckpt = session_info["remote_ckpt_dir"]
    local_ckpt  = EXP_ROOT / version / run_name / "checkpoints"
    local_ckpt.mkdir(parents=True, exist_ok=True)
    print(f"[chain] rsync {remote_ckpt}/ → {local_ckpt}/", flush=True)
    _rsync_from_remote(remote_cfg, remote_ckpt + "/", str(local_ckpt) + "/")


# ── Done-predicate ───────────────────────────────────────────────────────────

def training_is_done(version: str, run_name: str, expected_step: int) -> tuple[bool, str]:
    """Predicate A — strict.

    A version is "really done" iff ALL of:
      1. The final checkpoint directory exists at the expected step.
      2. It contains pretrained_model/model.safetensors.
      3. That file is at least MIN_SAFETENSORS_BYTES (sanity floor).
    """
    ckpt_dir = EXP_ROOT / version / run_name / "checkpoints" / f"{expected_step:06d}"
    if not ckpt_dir.is_dir():
        return False, f"missing checkpoint dir: {ckpt_dir}"
    safetensors = ckpt_dir / "pretrained_model" / "model.safetensors"
    if not safetensors.exists():
        return False, f"missing model.safetensors: {safetensors}"
    size = safetensors.stat().st_size
    if size < MIN_SAFETENSORS_BYTES:
        return False, f"model.safetensors too small: {size:,} bytes < floor {MIN_SAFETENSORS_BYTES:,}"
    return True, "ok"


# ── Main loop ────────────────────────────────────────────────────────────────

def run_chain(versions: list[str], run_name: str = "lerobot_output_r1",
              poll_interval: int = POLL_INTERVAL_SEC):
    """Train each version sequentially. Halt on first done-predicate failure."""
    # Validate everything upfront so we fail fast before any compute
    derived = {}
    for v in versions:
        d = derive_from_config(v)
        derived[v] = d
        n_eps = d["n_episodes"] if d["n_episodes"] is not None else "ALL"
        n_frames = d["n_frames"] if d["n_frames"] is not None else "?"
        print(f"[chain] queued {v}: {n_eps} eps | {n_frames} frames "
              f"| steps={d['total_steps']} | save_freq={d['save_freq']} "
              f"| final_step={d['final_step']}", flush=True)

    remote_cfg  = _load_remote_config()
    status_path = EXP_ROOT / "chain_status.json"
    status = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "versions": {}, "run_name": run_name}

    for v in versions:
        d = derived[v]
        print(f"\n=== {v} (final_step={d['final_step']}) ===", flush=True)

        session_info = launch_remote(v, run_name)
        wait_until_dead(remote_cfg, session_info["tmux_session"], v, run_name, poll_interval)
        pull_checkpoints(remote_cfg, v, run_name, session_info)

        ok, reason = training_is_done(v, run_name, d["final_step"])
        status["versions"][v] = {
            "ok": ok, "reason": reason,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **d,
        }
        status_path.write_text(json.dumps(status, indent=2))

        if not ok:
            print(f"\n=== HALT: {v} failed done-predicate: {reason} ===", flush=True)
            return
        print(f"=== {v} OK — next ===", flush=True)

    print("\n=== chain complete ===", flush=True)


def main():
    p = argparse.ArgumentParser(description="Run a sequential training chain.")
    p.add_argument("--versions", required=True,
                   help="comma-separated, e.g. v021,v022,v023,v024")
    p.add_argument("--run_name", default="lerobot_output_r1")
    p.add_argument("--poll_interval", type=int, default=POLL_INTERVAL_SEC,
                   help="seconds between tmux-alive polls (smoke tests: 20-30; real: 300)")
    args = p.parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    run_chain(versions, args.run_name, args.poll_interval)


if __name__ == "__main__":
    main()
