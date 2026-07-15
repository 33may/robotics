"""Empirical check: does the SmolVLA training preprocessing path ever resize
camera frames to 256x256? Replicates lerobot_train.py's preprocessing on a real
batch and traces image shapes at every stage.
"""
import torch
from pprint import pprint

import lerobot.policies.smolvla.modeling_smolvla as msv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType

CKPT = "vbti/experiments/duck_cup_smolvla/v019/lerobot_output_r1/checkpoints/005000/pretrained_model"
REPO = "eternalmay33/duck_cup_v019_all"
RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.left": "observation.images.camera2",
    "observation.images.right": "observation.images.camera3",
    "observation.images.gripper": "observation.images.empty_camera_0",
    "observation.images.gripper_depth": "observation.images.empty_camera_1",
}
device = torch.device("cuda")

# --- instrument the ONLY image-resize fn in the SmolVLA path ----------------
RESIZE_LOG = []
_orig_resize = msv.resize_with_pad
def _traced_resize(img, width, height, pad_value=-1):
    out = _orig_resize(img, width, height, pad_value=pad_value)
    RESIZE_LOG.append((tuple(img.shape), tuple(out.shape)))
    return out
msv.resize_with_pad = _traced_resize

# --- load policy from the actual checkpoint --------------------------------
policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.to(device).eval()
cfg = policy.config
print("policy.config.resize_imgs_with_padding =", cfg.resize_imgs_with_padding)
print("policy.config input image features (declared/provenance):")
for k, f in cfg.input_features.items():
    if f.type is FeatureType.VISUAL:
        print(f"   {k}: {tuple(f.shape)}")

# --- build dataset + dataloader the training way ---------------------------
fps = LeRobotDataset(REPO, episodes=[0]).meta.fps
obs_idx = list(range(1 - cfg.n_obs_steps, 1))          # [0]
act_idx = list(range(cfg.chunk_size))                  # [0..49]
img_keys_raw = list(RENAME_MAP.keys())
delta = {"observation.state": [i / fps for i in obs_idx],
         "action": [i / fps for i in act_idx]}
for k in img_keys_raw:
    delta[k] = [i / fps for i in obs_idx]

ds = LeRobotDataset(REPO, episodes=[0], delta_timestamps=delta)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))

print("\n=== STAGE 1: raw frames straight from the dataset (training input) ===")
for k in img_keys_raw:
    print(f"   {k}: {tuple(batch[k].shape)}  dtype={batch[k].dtype} "
          f"min={batch[k].min():.3f} max={batch[k].max():.3f}")

# --- build preprocessor exactly like lerobot_train.py ----------------------
pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CKPT,
    preprocessor_overrides={
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": ds.meta.stats,
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": RENAME_MAP},
    },
)

processed = pre(batch)
print("\n=== STAGE 2: after FULL training preprocessor (rename+norm+device) ===")
for k in sorted(processed):
    if "image" in k and hasattr(processed[k], "shape"):
        print(f"   {k}: {tuple(processed[k].shape)}")

print("\n=== STAGE 3: run the real training forward (triggers resize_with_pad) ===")
with torch.no_grad():
    loss, out = policy.forward(processed)
print(f"   forward OK, loss={loss.item():.4f}")

print("\n=== resize_with_pad call log  (input_shape -> output_shape) ===")
for i, (a, b) in enumerate(RESIZE_LOG):
    print(f"   call {i}: {a} -> {b}")

shapes_in = {s[0][-2:] for s in RESIZE_LOG}
shapes_out = {s[1][-2:] for s in RESIZE_LOG}
print("\n=== VERDICT ===")
print(f"   distinct INPUT H,W to resize : {shapes_in}")
print(f"   distinct OUTPUT H,W of resize: {shapes_out}")
print(f"   256x256 ever appears anywhere: "
      f"{any(256 in s for s in shapes_in | shapes_out)}")
