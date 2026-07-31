"""DataLoader factory for restoration training."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.restoration_dataset import RestorationDataset


def _collate_restoration_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate restoration samples into batched tensors."""
    return {
        "lr": torch.stack([sample["lr"] for sample in batch], dim=0),
        "hr": torch.stack([sample["hr"] for sample in batch], dim=0),
        "metadata": [sample["metadata"] for sample in batch],
    }


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a training config.

    Args:
        config: Parsed ``configs/train.yaml`` dictionary.

    Returns:
        Tuple of ``(train_loader, val_loader)``.

    Raises:
        FileNotFoundError: If dataset directories are missing.
        ValueError: If directories contain no images.
    """
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    train_dataset = RestorationDataset(
        root_dir=paths_cfg.get("train_dir", "data/train"),
        config=config,
        split="train",
    )
    val_dataset = RestorationDataset(
        root_dir=paths_cfg.get("val_dir", "data/validation"),
        config=config,
        split="val",
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(training_cfg.get("batch_size", 8)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": _collate_restoration_batch,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
