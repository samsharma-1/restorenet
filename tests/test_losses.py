"""Tests for restoration loss functions."""

from __future__ import annotations

import pytest
import torch

from src.losses import RestorationLoss


def test_restoration_loss_returns_scalar_and_parts() -> None:
    """Combined loss should return a scalar and named components."""
    criterion = RestorationLoss(
        {"weights": {"l1": 1.0, "ms_ssim": 0.1, "edge": 0.05, "fft": 0.05, "lpips": 0.0}}
    )
    pred = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)

    loss, parts = criterion(pred, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {"l1", "ms_ssim", "edge", "fft", "total"}.issubset(parts)


def test_restoration_loss_identical_tensors_is_small() -> None:
    """Identical tensors should produce near-zero pixel/edge/frequency loss."""
    criterion = RestorationLoss(
        {"weights": {"l1": 1.0, "ms_ssim": 0.0, "edge": 0.1, "fft": 0.1, "lpips": 0.0}}
    )
    target = torch.rand(1, 3, 16, 16)

    loss, _ = criterion(target, target)

    assert loss.item() < 1e-5


def test_restoration_loss_rejects_shape_mismatch() -> None:
    """Predictions and targets must have matching shapes."""
    criterion = RestorationLoss({"weights": {"l1": 1.0}})
    with pytest.raises(ValueError, match="Prediction shape"):
        criterion(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 16, 16))
