"""Tests for model construction and forward passes."""

from __future__ import annotations

import pytest
import torch

from src.models import NAFNet, build_model


def test_nafnet_outputs_2x_restoration_shape() -> None:
    """NAFNet should map LR tensors to 2x HR tensors."""
    model = NAFNet(width=8, num_blocks=2, scale_factor=2)
    inputs = torch.rand(2, 3, 16, 20)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 3, 32, 40)
    assert outputs.min() >= 0.0
    assert outputs.max() <= 1.0


def test_build_model_from_config() -> None:
    """Model factory should honor the YAML model section."""
    config = {
        "model": {
            "name": "NAFNet",
            "in_channels": 3,
            "out_channels": 3,
            "width": 8,
            "num_blocks": 1,
            "scale_factor": 2,
        }
    }

    model = build_model(config)

    assert isinstance(model, NAFNet)
    assert model.scale_factor == 2


def test_build_model_rejects_unknown_model() -> None:
    """Unsupported model names should fail early."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model({"model": {"name": "UnknownNet"}})
