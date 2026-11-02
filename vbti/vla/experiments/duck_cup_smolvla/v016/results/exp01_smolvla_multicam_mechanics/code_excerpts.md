# exp01 — verbatim source excerpts

All paths under `/home/may33/projects/ml_portfolio/robotics/lerobot/src/lerobot/`.

## `policies/smolvla/modeling_smolvla.py:403-443` — prepare_images
```python
def prepare_images(self, batch):
    images = []
    img_masks = []
    present_img_keys = [key for key in self.config.image_features if key in batch]
    missing_img_keys = [key for key in self.config.image_features if key not in batch]

    if len(present_img_keys) == 0:
        raise ValueError(
            f"All image features are missing from the batch. ..."
        )
    for key in present_img_keys:
        img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
        if self.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
        img = img * 2.0 - 1.0
        bsize = img.shape[0]
        device = img.device
        if f"{key}_padding_mask" in batch:
            mask = batch[f"{key}_padding_mask"].bool()
        else:
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
        images.append(img)
        img_masks.append(mask)

    for num_empty_cameras in range(len(missing_img_keys)):
        if num_empty_cameras >= self.config.empty_cameras:
            break
        img = torch.ones_like(img) * -1
        mask = torch.zeros_like(mask)
        images.append(img)
        img_masks.append(mask)
    return images, img_masks
```

## `policies/smolvla/modeling_smolvla.py:634-664` — embed_prefix per-camera ViT pass
```python
for _img_idx, (img, img_mask,) in enumerate(zip(images, img_masks, strict=False)):
    ...
    img_emb = self.vlm_with_expert.embed_image(img)
    img_emb_dim = img_emb.shape[-1]
    img_emb = img_emb * torch.tensor(img_emb_dim**0.5, ...)
    bsize, num_img_embs = img_emb.shape[:2]
    img_mask = img_mask[:, None].expand(bsize, num_img_embs)
    embs.append(img_emb)
    pad_masks.append(img_mask)
    att_masks += [0] * (num_img_embs)
```

## `policies/smolvla/modeling_smolvla.py:704` — sequence concat
```python
embs = torch.cat(embs, dim=1)
```

## `policies/smolvla/smolvlm_with_expert.py:179-192` — embed_image
```python
def embed_image(self, image: torch.Tensor):
    patch_attention_mask = None
    image_hidden_states = (
        self.get_vlm_model().vision_model(
            pixel_values=image.to(dtype=...),
            patch_attention_mask=patch_attention_mask,
        ).last_hidden_state
    )
    image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
    return image_hidden_states
```

## `policies/smolvla/configuration_smolvla.py:53` — empty_cameras default
```python
empty_cameras: int = 0
```

## `policies/factory.py:458-472` — image keys frozen at make_policy
```python
features = dataset_to_policy_features(ds_meta.features)
...
cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
if not cfg.input_features:
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
```

## `configs/policies.py:148-152` — image_features property
```python
@property
def image_features(self) -> dict[str, PolicyFeature]:
    if not self.input_features:
        return {}
    return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}
```

## `datasets/aggregate.py:64-78` — strict feature-equality check at merge
```python
fps = all_metadata[0].fps
robot_type = all_metadata[0].robot_type
features = all_metadata[0].features

for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
    if fps != meta.fps:
        raise ValueError(...)
    if robot_type != meta.robot_type:
        raise ValueError(...)
    if features != meta.features:
        raise ValueError(f"Same features is expected, but got features={meta.features} instead of {features}.")
```

## `datasets/factory.py:115` — MultiLeRobotDataset disabled
```python
raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
```

## `datasets/utils.py:715-720` — image must be 3-D
```python
for key, ft in features.items():
    shape = ft["shape"]
    if ft["dtype"] in ["image", "video"]:
        type = FeatureType.VISUAL
        if len(shape) != 3:
            raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
```

## `datasets/lerobot_dataset.py:246-258` — image/video/camera key derivation
```python
@property
def image_keys(self) -> list[str]:
    return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

@property
def video_keys(self) -> list[str]:
    return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

@property
def camera_keys(self) -> list[str]:
    return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]
```
