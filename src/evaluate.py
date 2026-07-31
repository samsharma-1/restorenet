"""Evaluation entry point for the image restoration pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate restoration quality on a validation/test set.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the config file (train or custom eval config).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model checkpoint. Overrides config if provided.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory with ground-truth/degraded image pairs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval",
        help="Directory to save evaluation reports and predictions.",
    )
    return parser.parse_args()


def evaluate(
    config: dict[str, Any],
    weights_path: str | None,
    input_dir: str | None,
    output_dir: str,
) -> None:
    """Run validation metrics (SSIM, PSNR, LPIPS).

    Args:
        config: Parsed configuration.
        weights_path: Path to model weights.
        input_dir: Directory containing evaluation images.
        output_dir: Directory for metrics and comparison reports.

    Raises:
        NotImplementedError: Core evaluation logic is implemented in Phase 6.
    """
    # TODO(Phase 2): Load evaluation dataset from input_dir or config paths.
    # TODO(Phase 3): Load NAFNet weights.
    # TODO(Phase 6): Compute SSIM, PSNR, LPIPS and write comparison reports.
    _ = (weights_path, input_dir, output_dir)
    logger.info(
        "Evaluation config loaded: model=%s",
        config.get("model", {}).get("name"),
    )
    raise NotImplementedError(
        "Evaluation pipeline not yet implemented. See Phase 6 in the roadmap."
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(Path(args.config))
    evaluate(
        config=config,
        weights_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
