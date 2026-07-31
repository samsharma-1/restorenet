"""PyTorch dataset for image restoration with synthetic degradation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.base import discover_images, load_image, resize_to_min_side
from src.datasets.degradation import (
    apply_downscale,
    apply_synthetic_degradation,
    sample_degradation_params,
)
from src.datasets.transforms import build_train_transforms, build_val_transforms


class RestorationDataset(Dataset):
    """Dataset for 2× super-resolution and denoising/deblurring.

    **Directory layout (on-the-fly mode, default)**

    Place clean high-resolution images directly under ``root_dir``::

        data/train/
            img001.png
            img002.tif

    **Paired mode (optional)**

    Provide aligned clean/degraded folders::

        data/train/
            clean/img001.png
            degraded/img001.png

    **Mixed-resolution strategy**

    Images smaller than ``min_crop_size`` are upsampled so the shorter side
    equals ``min_crop_size``. Larger images keep aspect ratio and are resized
    so the shorter side is at least ``min_crop_size`` before cropping. This
    avoids letterbox padding while keeping batch shapes consistent after crop.

    Returns a dict with:

    - ``lr``: degraded low-resolution input (C, H/f, W/f)
    - ``hr``: clean high-resolution target (C, H, W)
    - ``metadata``: source path and degradation parameters
    """

    def __init__(
        self,
        root_dir: str | Path,
        config: dict[str, Any],
        split: Literal["train", "val"] = "train",
        mode: Literal["on_the_fly", "paired"] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.config = config
        self.split = split
        data_cfg = config.get("data", {})
        self.image_size = int(data_cfg.get("image_size", 256))
        self.downscale_factor = int(
            config.get("degradation", {}).get(
                "downscale_factor",
                config.get("model", {}).get("scale_factor", 2),
            )
        )
        self.mode = mode or data_cfg.get("mode", "on_the_fly")
        self.degradation_enabled = bool(
            config.get("degradation", {}).get("enabled", True)
        )

        if self.mode == "paired":
            clean_dir = self.root_dir / "clean"
            degraded_dir = self.root_dir / "degraded"
            self.image_paths = discover_images(clean_dir, recursive=False)
            self.degraded_paths = {
                path.name: degraded_dir / path.name for path in self.image_paths
            }
            missing = [
                name
                for name, deg_path in self.degraded_paths.items()
                if not deg_path.exists()
            ]
            if missing:
                raise ValueError(
                    f"Missing degraded pairs for: {', '.join(missing[:5])}"
                    + (" ..." if len(missing) > 5 else "")
                )
        else:
            self.image_paths = discover_images(self.root_dir)
            self.degraded_paths = {}

        if split == "train":
            self.hr_transform: A.Compose | None = build_train_transforms(config)
            self._rng = np.random.default_rng(int(config.get("seed", 42)))
        else:
            self.hr_transform = build_val_transforms(config)
            self._rng = np.random.default_rng(
                int(config.get("seed", 42)) + 1_000_003
            )

        self._deterministic_val = bool(
            config.get("degradation", {}).get("deterministic_val", True)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _prepare_hr_image(self, image: np.ndarray) -> np.ndarray:
        """Resize for mixed-resolution training and apply HR augmentations."""
        min_side = self.image_size
        prepared = resize_to_min_side(image, min_side=min_side)
        if self.hr_transform is not None:
            transformed = self.hr_transform(image=prepared)
            return transformed["image"]
        return prepared

    def _degrade(
        self,
        hr_image: np.ndarray,
        index: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Create LR degraded view and collect metadata."""
        if self.mode == "paired":
            path = self.image_paths[index]
            lr_image = load_image(self.degraded_paths[path.name])
            target_lr_h = hr_image.shape[0] // self.downscale_factor
            target_lr_w = hr_image.shape[1] // self.downscale_factor
            if lr_image.shape[:2] != (target_lr_h, target_lr_w):
                lr_image = cv2.resize(
                    lr_image,
                    (target_lr_w, target_lr_h),
                    interpolation=cv2.INTER_CUBIC,
                )
            metadata = {
                "path": str(path),
                "mode": "paired",
            }
            return lr_image.astype(np.float32), metadata

        if not self.degradation_enabled:
            lr_image = apply_downscale(hr_image, factor=self.downscale_factor)
            return lr_image, {"mode": "on_the_fly", "degradation": "disabled"}

        if self.split == "val" and self._deterministic_val:
            rng = np.random.default_rng(int(self.config.get("seed", 42)) + index)
        else:
            rng = self._rng

        params = sample_degradation_params(
            self.config.get("degradation", {}),
            rng,
        )

        lr_image, _ = apply_synthetic_degradation(hr_image, params, rng=rng)
        metadata = {
            "path": str(self.image_paths[index]),
            "mode": "on_the_fly",
            "degradation": params.__dict__,
        }
        return lr_image, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        image = load_image(path)
        hr_image = self._prepare_hr_image(image)
        lr_image, metadata = self._degrade(hr_image, index)

        lr_tensor = torch.from_numpy(lr_image.transpose(2, 0, 1).copy()).float()
        hr_tensor = torch.from_numpy(hr_image.transpose(2, 0, 1).copy()).float()

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "metadata": metadata,
        }
