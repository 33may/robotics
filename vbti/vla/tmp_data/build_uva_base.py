"""Build smolvla_uva_base = lerobot/smolvla_base weights + a config.json patched
to type=smolvla_uva. Used as --policy.path for the v025-v029 UVA sweep so
lerobot-train builds a SmolVLAUVAPolicy and warm-starts from smolvla_base
(video_head stays fresh-init via strict=False loading).

Runs on the remote (smolvla_base already cached there from v021-v024).
"""
import json
import os
import shutil

from huggingface_hub import snapshot_download

DST = "/home/vbti/anton/data/smolvla_uva_base"

# UVA fields — must match the bake (siglip_output, 4x4 grid, 960-d, t_future=4).
UVA_FIELDS = {
    "teacher_features_key": "observation.video_features.siglip_output_4x4",
    "teacher_feature_dim": 960,
    "teacher_spatial_size": 4,
    "t_future": 4,
    "aux_weight": 0.3,
    "video_head_hidden": 720,
    "enable_aux_loss": True,
}

snap = snapshot_download("lerobot/smolvla_base")
print("smolvla_base snapshot:", snap)

os.makedirs(DST, exist_ok=True)
shutil.copy(os.path.join(snap, "model.safetensors"), os.path.join(DST, "model.safetensors"))

with open(os.path.join(snap, "config.json")) as f:
    cfg = json.load(f)
assert cfg["type"] == "smolvla", f"expected base type smolvla, got {cfg['type']}"
cfg["type"] = "smolvla_uva"
cfg.update(UVA_FIELDS)
with open(os.path.join(DST, "config.json"), "w") as f:
    json.dump(cfg, f, indent=4)
print("wrote patched config.json -> type=smolvla_uva +", list(UVA_FIELDS))

# Verify it round-trips through the registered SmolVLAUVAConfig.
# `import lerobot.policies` runs policies/__init__.py, which fires the
# @register_subclass("smolvla_uva") decorator. lerobot-train gets this for
# free (it imports lerobot.policies.factory); a bare script must do it.
import lerobot.policies  # noqa: E402,F401
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

loaded = PreTrainedConfig.from_pretrained(DST)
assert loaded.type == "smolvla_uva", loaded.type
assert getattr(loaded, "enable_aux_loss") is True
assert getattr(loaded, "t_future") == 4
sz = os.path.getsize(os.path.join(DST, "model.safetensors"))
print(f"VERIFY OK: {DST} | type={loaded.type} | model.safetensors={sz/1e6:.1f} MB")
