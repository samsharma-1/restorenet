"""Inference entry point for restoring degraded images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference to restore degraded scientific images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to the inference YAML config.",
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
        help="Input directory with degraded images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for restored images.",
    )
    return parser.parse_args()


def run_inference(
    config: dict[str, Any],
    weights_path: str | None,
    input_dir: str | None,
    output_dir: str | None,
) -> None:
    """Restore degraded images and save predictions.

    Args:
        config: Parsed inference configuration.
        weights_path: Path to model weights.
        input_dir: Directory of input images.
        output_dir: Directory to write restored images.

    Raises:
        NotImplementedError: Core inference logic is implemented in Phase 7.
    """
    # TODO(Phase 3): Load NAFNet and weights from config/paths.
    # TODO(Phase 7): Batch inference with optional FP16 and output saving.
    # TODO(Phase 8): torch.compile, ONNX/TensorRT export and benchmarking.
    paths = config.get("paths", {})
    resolved_weights = weights_path or paths.get("weights")
    resolved_input = input_dir or paths.get("input_dir")
    resolved_output = output_dir or paths.get("output_dir")
    logger.info(
        "Inference config: weights=%s, input=%s, output=%s",
        resolved_weights,
        resolved_input,
        resolved_output,
    )
    raise NotImplementedError(
        "Inference pipeline not yet implemented. See Phase 7 in the roadmap."
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(Path(args.config))
    run_inference(
        config=config,
        weights_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
