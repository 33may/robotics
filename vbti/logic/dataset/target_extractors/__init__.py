"""Registry of target-feature extractors for UVA aux-loss bake.

Each extractor maps (teacher_model, image_batch, **kwargs) -> (B, S, S, feat_dim)
feature tensor. Decorate with @register("name") to add a new extractor.
"""
from collections.abc import Callable

import torch

_EXTRACTORS: dict[str, Callable[..., torch.Tensor]] = {}


def register(name: str):
    """Decorator: register an extractor under a unique name.

    Raises ValueError if a name is registered twice (catches accidental shadowing).
    """
    def decorator(fn: Callable[..., torch.Tensor]):
        if name in _EXTRACTORS:
            raise ValueError(f"target extractor '{name}' already registered")
        _EXTRACTORS[name] = fn
        return fn
    return decorator


def get(name: str) -> Callable[..., torch.Tensor]:
    """Look up an extractor by name. Raises KeyError if not found."""
    if name not in _EXTRACTORS:
        raise KeyError(
            f"target extractor '{name}' not registered. "
            f"Available: {sorted(_EXTRACTORS.keys())}"
        )
    return _EXTRACTORS[name]


def list_available() -> list[str]:
    """List all registered extractor names."""
    return sorted(_EXTRACTORS.keys())
