"""L2 extractor: SigLIP final-layer patch tokens from the teacher SmolVLA model.

For each input image batch:
  1. Forward through teacher.vlm_with_expert.embed_image -> (B, N_patches, feat_dim)
  2. Reshape to spatial grid (B, sqrt(N), sqrt(N), feat_dim)
  3. Avg-pool to (B, spatial_size, spatial_size, feat_dim)
"""
import math

import torch
import torch.nn.functional as F

from vbti.logic.dataset.target_extractors import register


@register("siglip_output")
def siglip_output_extractor(
    teacher,  # SmolVLAPolicy
    images: torch.Tensor,  # (B, 3, H, W) — pre-resized to teacher's expected input
    spatial_size: int = 4,
    **kwargs,  # noqa: ARG001 — accepted for registry uniformity
) -> torch.Tensor:
    """Returns (B, spatial_size, spatial_size, feat_dim) features, no_grad.

    Args:
        teacher: SmolVLAPolicy instance (frozen, eval mode).
        images: (B, 3, H, W) tensor matching teacher's SigLIP input convention.
                Caller is responsible for [-1, 1] normalization and resize_with_pad.
        spatial_size: side length of output grid (k x k). Must divide the native
                      SigLIP patch grid side.
    """
    with torch.no_grad():
        # embed_image returns (B, N_patches, feat_dim) — raw SigLIP output
        feats = teacher.model.vlm_with_expert.embed_image(images)  # (B, N, D)

        B, N, D = feats.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise ValueError(
                f"SigLIP output has {N} patches; expected square grid. "
                f"Teacher model may have non-square patch layout."
            )

        # Reshape (B, N, D) -> (B, D, side, side) for avg_pool2d
        feats_2d = feats.transpose(1, 2).reshape(B, D, side, side)

        if spatial_size != side:
            if side % spatial_size != 0:
                raise ValueError(
                    f"spatial_size={spatial_size} must divide native patch grid side={side}"
                )
            kernel = side // spatial_size
            feats_2d = F.avg_pool2d(feats_2d, kernel_size=kernel)

        # (B, D, spatial_size, spatial_size) -> (B, spatial_size, spatial_size, D)
        return feats_2d.permute(0, 2, 3, 1).contiguous()
