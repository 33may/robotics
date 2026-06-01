"""Generate v025-v029 configs — the SmolVLA-UVA data-efficiency sweep.

Sibling of gen_data_eff_configs.py (which generates the vanilla v021-v024).
Clones v020/config.yaml once per data slice, overriding only:
  - model.type                  : smolvla -> smolvla_uva
  - model.pretrained            : -> smolvla_uva_base (type=smolvla_uva ckpt)
  - dataset.sources[0].repo_id  : -> the baked _uva dataset copy
  - dataset.sources[0].episodes : which episodes to include (stride slice)
  - logging.save_freq           : tuned so each run saves ~6 checkpoints

Everything else (vision unfrozen, aug, BS, epochs, lr schedule, ...) is
inherited from v020 — same as the v021-v024 sweep, so the UVA curve is
directly comparable.

The UVA aux-loss hyperparams (aux_weight, t_future, teacher_features_key, ...)
are NOT in config.yaml — they live in the smolvla_uva_base checkpoint's
config.json and ride into lerobot-train via --policy.path.

Slice -> version (small first; v029 adds the full-dataset point the
vanilla sweep never had):
    v025 = 1/16 (stride 16)   v026 = 1/8   v027 = 1/4   v028 = 1/2
    v029 = full dataset (no episode filter)

n_frames is read from the SOURCE dataset (duck_cup_v020_all, present in the
local cache) because the baked _uva copy lives only on the remote — the bake
adds a column, not rows, so per-episode frame counts are identical.

Run once; review the generated config.yaml files; then launch the chain:
    python scripts/gen_uva_data_eff_configs.py
    python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029
"""
from __future__ import annotations

import yaml

from vbti.logic.train.chain import EXP_ROOT, n_frames_for
from vbti.logic.train.config_utils import TrainConfig

BASE_VERSION   = "v020"
SOURCE_REPO    = "eternalmay33/duck_cup_v020_all"       # local — for frame counts
UVA_REPO       = "eternalmay33/duck_cup_v020_all_uva"   # baked copy (remote-only)
UVA_PRETRAINED = "/home/vbti/anton/data/smolvla_uva_base"
TOTAL_EPISODES = 765
N_CKPTS        = 6   # ~1 checkpoint per 2 epochs for a 12-epoch-equivalent run

# (version, stride). stride=None => full dataset, no episode filter.
PLAN = [
    ("v025", 16),    # 1/16 — 48 ep
    ("v026", 8),     # 1/8  — 96 ep
    ("v027", 4),     # 1/4  — 192 ep
    ("v028", 2),     # 1/2  — 383 ep
    ("v029", None),  # full — 765 ep
]


def episodes_for(stride: int | None) -> list[int] | None:
    if stride is None:
        return None
    return list(range(0, TOTAL_EPISODES, stride))


def write_one(version: str, stride: int | None) -> dict:
    base = EXP_ROOT / BASE_VERSION / "config.yaml"
    cfg  = yaml.safe_load(base.read_text())

    eps         = episodes_for(stride)
    n_frames    = n_frames_for(SOURCE_REPO, eps)
    epochs      = cfg["training"]["epochs"]
    bs          = cfg["training"]["batch_size"]
    total_steps = int(epochs * n_frames / bs)
    save_freq   = max(1, total_steps // N_CKPTS)

    cfg["model"]["type"]                    = "smolvla_uva"
    cfg["model"]["pretrained"]              = UVA_PRETRAINED
    cfg["dataset"]["sources"][0]["repo_id"] = UVA_REPO
    cfg["dataset"]["sources"][0]["episodes"] = eps
    cfg["logging"]["save_freq"]             = save_freq

    # Bake `steps` in explicitly (drop `epochs`). The baked _uva dataset lives
    # only on the remote, so the chain can't count frames locally to derive
    # steps from epochs. total_steps here == the exact step counts v021-v024 /
    # v020 actually ran (epochs=12 at bs=12 -> steps == n_frames; verified W&B).
    cfg["training"]["epochs"] = None
    cfg["training"]["steps"]  = total_steps

    # Fail fast: the generated config must parse through the typed schema
    # (this is what catches a bad `smolvla_uva` enum / model block).
    TrainConfig.from_dict(yaml.safe_load(yaml.safe_dump(cfg)))

    dst_dir = EXP_ROOT / version
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))

    return {
        "version": version, "stride": stride,
        "n_episodes": len(eps) if eps else TOTAL_EPISODES,
        "n_frames": n_frames, "total_steps": total_steps,
        "save_freq": save_freq, "path": str(dst),
    }


def main():
    print(f"{'version':<8} {'stride':<7} {'eps':<5} {'frames':<8} {'steps':<7} {'save_freq':<10}")
    for v, s in PLAN:
        r = write_one(v, s)
        stride_str = "full" if s is None else str(s)
        print(f"{r['version']:<8} {stride_str:<7} {r['n_episodes']:<5} "
              f"{r['n_frames']:<8} {r['total_steps']:<7} {r['save_freq']:<10}  → {r['path']}")


if __name__ == "__main__":
    main()
