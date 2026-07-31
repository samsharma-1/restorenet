"""Dataset loaders and augmentation pipelines."""

from src.datasets.base import discover_images, load_image
from src.datasets.dataloader import build_dataloaders
from src.datasets.degradation import (
    DegradationConfig,
    apply_downscale,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_speckle_noise,
    apply_synthetic_degradation,
)
from src.datasets.restoration_dataset import RestorationDataset

__all__ = [
    "DegradationConfig",
    "RestorationDataset",
    "apply_downscale",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_speckle_noise",
    "apply_synthetic_degradation",
    "build_dataloaders",
    "discover_images",
    "load_image",
]
