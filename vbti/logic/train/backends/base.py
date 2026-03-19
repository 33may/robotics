"""
Abstract training backend.

Each model (SmolVLA, GR00T, ...) implements this interface.
The training engine calls these methods — the loop itself is shared.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ModelBundle:
    """Everything needed to train and save a model."""
    model: torch.nn.Module
    preprocessor: object    # model-specific, called on each batch
    postprocessor: object   # model-specific, used for inference only
    dataset_meta: object    # dataset metadata (stats, features, fps)


class TrainingBackend(ABC):
    """Interface that each model backend implements."""

    @abstractmethod
    def load_model(self, config) -> ModelBundle:
        """Load pretrained model + preprocessor/postprocessor from config.

        Returns a ModelBundle with everything needed for training.
        """

    @abstractmethod
    def make_dataloaders(self, config, dataset_meta) -> tuple:
        """Create train and val dataloaders from config.

        Returns (train_loader, val_loader). val_loader may be None.
        """

    @abstractmethod
    def train_step(self, bundle: ModelBundle, batch, optimizer) -> tuple[torch.Tensor, dict]:
        """Run one training step.

        Args:
            bundle: ModelBundle from load_model()
            batch: raw batch from dataloader
            optimizer: the optimizer (for grad clipping etc)

        Returns:
            (loss, metrics_dict) where loss is a scalar tensor
            and metrics_dict has string keys with float values.
        """

    @abstractmethod
    def validate(self, bundle: ModelBundle, val_loader, n_batches: int) -> dict:
        """Run validation and return metrics dict.

        Args:
            bundle: ModelBundle
            val_loader: validation dataloader
            n_batches: max batches to evaluate

        Returns:
            dict with at least "val_loss" key.
        """

    @abstractmethod
    def save_checkpoint(self, bundle: ModelBundle, path: Path, is_best: bool = False,
                        optimizer=None, scheduler=None, step: int = None):
        """Save model checkpoint to path.

        Should save: model weights, preprocessor, postprocessor, config.
        If optimizer/scheduler/step provided, also save training state for resume.
        """

    @abstractmethod
    def get_optimizer(self, bundle: ModelBundle, config) -> tuple:
        """Create optimizer and scheduler from config.

        Returns (optimizer, scheduler).
        """
