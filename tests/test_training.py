"""Smoke tests for the training loop."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.train import train


def _write_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def test_training_one_epoch_writes_checkpoints(tmp_path: Path) -> None:
    """A tiny generated dataset should train for one epoch and save checkpoints."""
    rng = np.random.default_rng(123)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    for split_dir in (train_dir, val_dir):
        for index in range(3):
            gradient = np.linspace(0.1, 0.9, 32, dtype=np.float32)
            image = np.stack(
                [
                    np.tile(gradient, (32, 1)),
                    np.full((32, 32), 0.2 + 0.1 * index, dtype=np.float32),
                    rng.random((32, 32), dtype=np.float32) * 0.5 + 0.25,
                ],
                axis=-1,
            )
            _write_rgb_image(split_dir / f"sample_{index}.png", image)

    config = {
        "seed": 7,
        "device": "cpu",
        "paths": {
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "model": {
            "name": "NAFNet",
            "in_channels": 3,
            "out_channels": 3,
            "scale_factor": 2,
            "width": 4,
            "num_blocks": 1,
        },
        "data": {
            "image_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "mode": "on_the_fly",
        },
        "degradation": {
            "enabled": True,
            "deterministic_val": True,
            "downscale_factor": 2,
            "speckle_var": 0.01,
            "blur_sigma": 0.5,
            "gaussian_noise_std": 0.0,
            "apply_prob": {"speckle": 1.0, "blur": 1.0, "noise": 0.0, "downscale": 1.0},
        },
        "augmentation": {
            "train": {"crop_size": 16, "horizontal_flip": False, "vertical_flip": False, "rotate90": False},
            "val": {"crop_size": 16},
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-5},
        "loss": {"weights": {"l1": 1.0, "ms_ssim": 0.0, "lpips": 0.0, "edge": 0.0, "fft": 0.0}},
        "checkpoint": {"save_every_n_epochs": 1, "save_best_only": False, "monitor_metric": "val_total", "mode": "min"},
    }

    result = train(config)

    assert len(result["history"]) == 1
    assert Path(result["checkpoint_dir"], "best_model.pth").is_file()
    assert Path(result["checkpoint_dir"], "last_model.pth").is_file()
    assert Path(result["checkpoint_dir"], "epoch_0001.pth").is_file()
