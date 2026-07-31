"""Unit tests for dataset loading, degradation, and dataloaders."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.datasets.base import discover_images, load_image, resize_to_min_side
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
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _write_rgb_image(path: Path, array: np.ndarray) -> None:
    """Save an RGB float or uint8 array to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


@pytest.fixture
def sample_images(tmp_path: Path) -> Path:
    """Create tiny RGB images for dataset tests."""
    rng = np.random.default_rng(0)
    sizes = [(64, 64), (80, 48), (48, 96)]
    for index, (height, width) in enumerate(sizes):
        gradient = np.linspace(0.2, 0.9, width, dtype=np.float32)
        image = np.stack(
            [
                np.tile(gradient, (height, 1)),
                np.full((height, width), 0.3 + 0.1 * index, dtype=np.float32),
                rng.random((height, width), dtype=np.float32) * 0.5 + 0.2,
            ],
            axis=-1,
        )
        _write_rgb_image(tmp_path / f"sample_{index}.png", image)
    return tmp_path


@pytest.fixture
def dataset_config(sample_images: Path) -> dict:
    """Minimal config pointing at temporary sample images."""
    base = load_config(PROJECT_ROOT / "configs" / "train.yaml")
    base["paths"]["train_dir"] = str(sample_images)
    base["paths"]["val_dir"] = str(sample_images)
    base["training"]["batch_size"] = 2
    base["data"]["num_workers"] = 0
    base["data"]["image_size"] = 32
    base["augmentation"]["train"]["crop_size"] = 32
    base["augmentation"]["val"]["crop_size"] = 32
    base["degradation"]["speckle_var"] = 0.04
    base["degradation"]["blur_sigma"] = 1.0
    base["degradation"]["gaussian_noise_std"] = 0.01
    return base


class TestDegradation:
    """Tests for synthetic degradation helpers."""

    @pytest.fixture
    def clean_patch(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.random((64, 64, 3), dtype=np.float32) * 0.8 + 0.1

    def test_speckle_is_multiplicative(self, clean_patch: np.ndarray) -> None:
        variance = 0.05
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, np.sqrt(variance), size=clean_patch.shape).astype(
            np.float32
        )
        expected = np.clip(clean_patch * (1.0 + noise), 0.0, 1.0)
        actual = apply_speckle_noise(
            clean_patch,
            variance,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_speckle_preserves_shape_and_range(self, clean_patch: np.ndarray) -> None:
        degraded = apply_speckle_noise(clean_patch, variance=0.05)
        assert degraded.shape == clean_patch.shape
        assert degraded.min() >= 0.0
        assert degraded.max() <= 1.0

    def test_gaussian_blur_preserves_shape(self, clean_patch: np.ndarray) -> None:
        blurred = apply_gaussian_blur(clean_patch, sigma=1.5)
        assert blurred.shape == clean_patch.shape
        assert blurred.dtype == np.float32

    def test_gaussian_noise_preserves_shape(self, clean_patch: np.ndarray) -> None:
        noisy = apply_gaussian_noise(clean_patch, std=0.03, rng=np.random.default_rng(0))
        assert noisy.shape == clean_patch.shape
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_downscale_factor_two(self, clean_patch: np.ndarray) -> None:
        downscaled = apply_downscale(clean_patch, factor=2)
        assert downscaled.shape == (32, 32, 3)

    def test_synthetic_pipeline_returns_lr_hr_pair(self, clean_patch: np.ndarray) -> None:
        params = DegradationConfig(
            downscale_factor=2,
            speckle_var=0.05,
            blur_sigma=1.0,
            gaussian_noise_std=0.02,
        )
        lr, hr = apply_synthetic_degradation(
            clean_patch,
            params,
            rng=np.random.default_rng(7),
        )
        assert hr.shape == clean_patch.shape
        assert lr.shape == (32, 32, 3)
        assert lr.dtype == np.float32


class TestBaseUtilities:
    """Tests for image discovery and loading."""

    def test_discover_images_finds_samples(self, sample_images: Path) -> None:
        paths = discover_images(sample_images)
        assert len(paths) == 3

    def test_discover_images_raises_on_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No images found"):
            discover_images(tmp_path)

    def test_load_image_returns_float_rgb(self, sample_images: Path) -> None:
        path = next(sample_images.glob("*.png"))
        image = load_image(path)
        assert image.shape[2] == 3
        assert image.dtype == np.float32
        assert image.max() <= 1.0

    def test_resize_to_min_side(self) -> None:
        image = np.zeros((40, 100, 3), dtype=np.float32)
        resized = resize_to_min_side(image, min_side=50)
        assert min(resized.shape[:2]) == 50


class TestRestorationDataset:
    """Tests for RestorationDataset and DataLoader factory."""

    def test_dataset_item_shapes(self, dataset_config: dict) -> None:
        dataset = RestorationDataset(
            root_dir=dataset_config["paths"]["train_dir"],
            config=dataset_config,
            split="train",
        )
        sample = dataset[0]
        assert sample["lr"].shape == (3, 16, 16)
        assert sample["hr"].shape == (3, 32, 32)
        assert "metadata" in sample

    def test_dataloader_batch_shapes(self, dataset_config: dict) -> None:
        train_loader, val_loader = build_dataloaders(dataset_config)
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert train_batch["lr"].shape == (2, 3, 16, 16)
        assert train_batch["hr"].shape == (2, 3, 32, 32)
        assert len(train_batch["metadata"]) == 2
        assert len(val_loader.dataset) == 3

    def test_paired_mode(self, tmp_path: Path, dataset_config: dict) -> None:
        clean_dir = tmp_path / "clean"
        degraded_dir = tmp_path / "degraded"
        image = np.full((64, 64, 3), 0.6, dtype=np.float32)
        _write_rgb_image(clean_dir / "pair.png", image)
        _write_rgb_image(degraded_dir / "pair.png", image * 0.5)

        dataset_config["paths"]["train_dir"] = str(tmp_path)
        dataset_config["data"]["mode"] = "paired"
        dataset_config["data"]["image_size"] = 32
        dataset = RestorationDataset(
            root_dir=tmp_path,
            config=dataset_config,
            split="train",
        )
        sample = dataset[0]
        assert sample["hr"].shape == (3, 32, 32)
        assert sample["lr"].shape == (3, 16, 16)
        assert sample["metadata"]["mode"] == "paired"

    def test_empty_directory_raises(self, tmp_path: Path, dataset_config: dict) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        dataset_config["paths"]["train_dir"] = str(empty)
        with pytest.raises(ValueError, match="No images found"):
            RestorationDataset(
                root_dir=empty,
                config=dataset_config,
                split="train",
            )


def test_train_config_has_dataset_sections() -> None:
    """Ensure train.yaml documents degradation and augmentation settings."""
    config_path = PROJECT_ROOT / "configs" / "train.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert "degradation" in config
    assert "augmentation" in config
    assert config["degradation"]["downscale_factor"] == 2

