"""Albumentations transform pipelines for restoration training."""

from __future__ import annotations

from typing import Any

import albumentations as A


def build_train_transforms(config: dict[str, Any]) -> A.Compose:
    """Build stochastic augmentations for training.

    Normalization choice: pixel values remain in **[0, 1]** (no ImageNet stats).
    Scientific microscopy/SAR images are not ImageNet-distributed; keeping [0, 1]
    float tensors simplifies loss computation and metric evaluation.

    Args:
        config: Full training config; reads ``data.image_size`` and
            ``augmentation.train``.

    Returns:
        Albumentations compose pipeline operating on HR images before degradation.
    """
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {}).get("train", {})
    crop_size = int(aug_cfg.get("crop_size", data_cfg.get("image_size", 256)))

    transforms: list[A.BasicTransform] = [
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
    ]

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))
    if aug_cfg.get("rotate90", True):
        transforms.append(A.RandomRotate90(p=0.5))

    return A.Compose(transforms)


def build_val_transforms(config: dict[str, Any]) -> A.Compose:
    """Build deterministic augmentations for validation.

    Resizes the shorter side to ``image_size`` then center-crops to a fixed
    square so batches are collate-friendly.

    Args:
        config: Full training config; reads ``data.image_size`` and
            ``augmentation.val``.

    Returns:
        Albumentations compose pipeline for validation HR images.
    """
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {}).get("val", {})
    crop_size = int(aug_cfg.get("crop_size", data_cfg.get("image_size", 256)))

    return A.Compose(
        [
            A.SmallestMaxSize(max_size=crop_size, interpolation=1),
            A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        ]
    )
