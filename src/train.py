"""Training entry point for the image restoration pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.datasets.dataloader import build_dataloaders
from src.models import build_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train an image restoration model (NAFNet baseline).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args()


def train(config: dict[str, Any], resume_path: str | None = None) -> None:
    """Run the training loop.

    Args:
        config: Parsed training configuration.
        resume_path: Optional checkpoint to resume from.

    Raises:
        NotImplementedError: Core training logic is implemented in Phase 5.
    """
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config)
    logger.info(
        "Model ready: %s with %.2fM parameters",
        model.__class__.__name__,
        sum(parameter.numel() for parameter in model.parameters()) / 1_000_000,
    )
    logger.info(
        "Dataloaders ready: train=%d samples, val=%d samples, batch_size=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        config.get("training", {}).get("batch_size"),
    )
    _ = train_loader, val_loader, model
    # TODO(Phase 4): Wire combined loss (L1, MS-SSIM, LPIPS, edge, FFT).
    # TODO(Phase 5): Mixed precision, checkpoints, schedulers, W&B, TensorBoard.
    _ = resume_path
    logger.info("Training config loaded: model=%s", config.get("model", {}).get("name"))
    raise NotImplementedError(
        "Training loop not yet implemented. See Phase 5 in the roadmap."
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()

