"""Tests for project structure and configuration loading."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_project_structure_exists() -> None:
    """Verify required directories and key files exist."""
    required_dirs = [
        "configs",
        "data/train",
        "data/validation",
        "data/test",
        "notebooks",
        "src/datasets",
        "src/models",
        "src/losses",
        "src/metrics",
        "src/utils",
        "outputs",
        "checkpoints",
        "tests",
    ]
    required_files = [
        "configs/train.yaml",
        "configs/inference.yaml",
        "requirements.txt",
        "README.md",
        "src/train.py",
        "src/evaluate.py",
        "src/inference.py",
    ]
    for rel_path in required_dirs:
        assert (PROJECT_ROOT / rel_path).is_dir(), f"Missing directory: {rel_path}"
    for rel_path in required_files:
        assert (PROJECT_ROOT / rel_path).is_file(), f"Missing file: {rel_path}"


def test_train_config_loads() -> None:
    """Train config should parse as a non-empty dict with expected keys."""
    config = load_config(PROJECT_ROOT / "configs" / "train.yaml")
    assert isinstance(config, dict)
    assert config["model"]["name"] == "NAFNet"
    assert "training" in config
    assert config["training"]["batch_size"] > 0


def test_inference_config_loads() -> None:
    """Inference config should parse and contain path/model sections."""
    config = load_config(PROJECT_ROOT / "configs" / "inference.yaml")
    assert isinstance(config, dict)
    assert "paths" in config
    assert "model" in config
    assert config["model"]["scale_factor"] == 2


def test_config_loader_raises_on_missing_file() -> None:
    """load_config should raise FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "configs" / "nonexistent.yaml")


def test_yaml_files_are_valid() -> None:
    """All config YAML files should be syntactically valid."""
    for name in ("train.yaml", "inference.yaml"):
        path = PROJECT_ROOT / "configs" / name
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data is not None
