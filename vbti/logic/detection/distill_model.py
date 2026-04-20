"""Tiny distilled detector — MobileNetV3-Small backbone, 6-output sigmoid head.

Output layout: [duck_cx, duck_cy, duck_conf, cup_cx, cup_cy, cup_conf]
All outputs in [0, 1] (sigmoid).

Designed to run 4 independent per-camera instances in parallel on GPU.
Input: (B, 3, 224, 224) float32 normalized by ImageNet mean/std.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DistilledDetector(nn.Module):
    """MobileNetV3 feature extractor + small regression head.

    backbone ∈ {mobilenet_v3_small, mobilenet_v3_large}.
    forward(x) expects x in [0, 1] float32, shape (B, 3, 224, 224).
    Internally normalizes with ImageNet stats.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if backbone == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_small(weights=weights)
            feat_dim = 576  # mobilenetv3-small final conv channels
        elif backbone == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_large(weights=weights)
            feat_dim = 960  # mobilenetv3-large final conv channels
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.features = net.features
        self.avgpool = net.avgpool  # AdaptiveAvgPool2d((1,1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 6),
        )

        # ImageNet normalization buffers
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) in [0, 1] — returns (B, 6) in [0, 1]."""
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = self.head(x)
        return torch.sigmoid(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    m = DistilledDetector()
    x = torch.rand(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
    print("param count:", count_params(m))
    print("sample output:", y[0].detach().numpy())
