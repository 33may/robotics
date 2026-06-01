"""Generate configs for the Phase-1 data-efficiency sweep.

Clones v020/config.yaml four times — once per data slice — overriding only:
  - dataset.sources[0].episodes : which episodes to include
  - logging.save_freq           : tuned so each run saves 6 checkpoints (1 per 2 epochs)

Everything else (vision unfrozen, aug, BS=12, epochs=12, etc.) is inherited from v020.

Run once; review the generated config.yaml files; commit them; then launch the chain:
    python scripts/gen_data_eff_configs.py
    python -m vbti.logic.train.chain --versions v021,v022,v023,v024
"""

from __future__ import annotations

import yaml

from vbti.logic.train.chain import EXP_ROOT, n_frames_for


BASE_VERSION   = "v020"
TOTAL_EPISODES = 765
N_CKPTS        = 6   # save every 2 epochs for a 12-epoch run

# (version, stride). v020 anchor at 100% is already trained — not regenerated.
PLAN = [
    ("v021", 16),   # 6.25% — 48 ep
    ("v022", 8),    # 12.5% — 96 ep
    ("v023", 4),    # 25%   — 192 ep
    ("v024", 2),    # 50%   — 383 ep
]


def episodes_for(stride: int) -> list[int]:
    return list(range(0, TOTAL_EPISODES, stride))


def write_one(version: str, stride: int) -> dict:
    base = EXP_ROOT / BASE_VERSION / "config.yaml"
    cfg  = yaml.safe_load(base.read_text())

    eps         = episodes_for(stride)
    repo_id     = cfg["dataset"]["sources"][0]["repo_id"]
    n_frames    = n_frames_for(repo_id, eps)
    epochs      = cfg["training"]["epochs"]
    bs          = cfg["training"]["batch_size"]
    total_steps = int(epochs * n_frames / bs)
    save_freq   = total_steps // N_CKPTS

    cfg["dataset"]["sources"][0]["episodes"] = eps
    cfg["logging"]["save_freq"] = save_freq

    dst_dir = EXP_ROOT / version
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))

    return {
        "version": version, "stride": stride, "n_episodes": len(eps),
        "n_frames": n_frames, "total_steps": total_steps,
        "save_freq": save_freq, "path": str(dst),
    }


def main():
    print(f"{'version':<8} {'stride':<7} {'eps':<5} {'frames':<8} {'steps':<7} {'save_freq':<10}")
    for v, s in PLAN:
        r = write_one(v, s)
        print(f"{r['version']:<8} {r['stride']:<7} {r['n_episodes']:<5} "
              f"{r['n_frames']:<8} {r['total_steps']:<7} {r['save_freq']:<10}  → {r['path']}")


if __name__ == "__main__":
    main()
