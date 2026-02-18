from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

class VisualEncoderBase(ABC):
    """
    Abstract base class for visual encoders.
    All custom encoders must inherit from this class and implement the required methods.
    """

    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def load_model(self):
        """
        Loads the pretrained model.
        """
        pass

    @abstractmethod
    def preprocess(self, image):
        """
        Preprocesses a single image or a batch of images for the encoder.
        Returns a tensor ready to be passed to the model.
        """
        pass

    @abstractmethod
    def encode(self, image_tensor):
        """
        Runs the model on the input tensor and returns the embeddings.
        """
        pass

    @abstractmethod
    def get_output_shape(self):
        """
        Runs the model on the input tensor and returns the embeddings.
        """
        pass

    def to(self, device):
        """
        Moves the model to the specified device.
        """
        self.device = device
        return self


class CLIPVisualEncoder(VisualEncoderBase):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)

    def preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        return inputs["pixel_values"].to(self.device)

    def encode(self, image_tensor):
        inputs = self.preprocess(image_tensor)
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=inputs)

    def get_output_shape(self):
        return self.model.config.projection_dim




# ─────────────────────── helper utilities
def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    func    = getattr(torchvision.models, name)
    resnet  = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()           # global-avg-pool → 512-d (for resnet18/34)
    return resnet

def replace_submodules(root_module: nn.Module,
                       predicate,
                       func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    bn_paths = [k.split('.') for k, m in
                root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, key in bn_paths:
        parent_mod = root_module.get_submodule('.'.join(parent)) if parent else root_module
        src = parent_mod[int(key)] if isinstance(parent_mod, nn.Sequential) else getattr(parent_mod, key)
        tgt = func(src)
        if isinstance(parent_mod, nn.Sequential):
            parent_mod[int(key)] = tgt
        else:
            setattr(parent_mod, key, tgt)
    assert not any(predicate(m) for m in root_module.modules())
    return root_module

def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16):
    return replace_submodules(
        root_module,
        predicate=lambda m: isinstance(m, nn.BatchNorm2d),
        func=lambda m: nn.GroupNorm(m.num_features // features_per_group, m.num_features)
    )


class CNNVisualEncoder(VisualEncoderBase, nn.Module):
    """
    * Принимает тензор (B,3,H,W) в диапазоне [0,1] БЕЗ ресайза и нормализации.
    * Возвращает признаки после global-pool, размер 512 для resnet18/34,
      2048 для resnet50.
    * BatchNorm → GroupNorm для совместимости с EMA.
    """

    def __init__(
        self,
        resnet_name: str = "resnet18",
        device: str = "cuda",
        feats_per_group: int = 16,
    ):
        nn.Module.__init__(self)
        VisualEncoderBase.__init__(self, device)

        backbone = get_resnet(resnet_name)
        backbone = replace_bn_with_gn(backbone, feats_per_group)
        self.model = backbone.to(self.device)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96, device=self.device)
            self._feat_dim = self.model(dummy).shape[-1]

    # --------------------------------------------------------------------- API
    def load_model(self):
        pass

    def preprocess(self, image: Union[np.ndarray, Image, torch.Tensor]):
        """
        Оставляем пустым, потому что вы уже подаёте готовые тензоры.
        """
        raise NotImplementedError(
            "В данной версии preprocess не используется – "
            "передавайте сразу тензор (B,3,H,W) в encode/forward."
        )

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(image_tensor)

    def get_output_shape(self) -> int:
        return self._feat_dim

    # ------------------------- forward – main entry --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor (B,3,H,W)  в диапазоне [0,1].
        Никаких преобразований размера или нормализации не выполняется.
        """
        return self.model(x.to(self.device))