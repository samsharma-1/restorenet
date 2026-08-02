"""Evaluation entry point for the image restoration pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.dataloader import _collate_restoration_batch
from src.datasets.restoration_dataset import RestorationDataset
from src.metrics.restoration_metrics import RestorationMetrics
from src.models import build_model
from src.utils.config import load_config
import torchvision.utils as tv_utils

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
    """
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    # Resolve paths
    resolved_input = input_dir or config.get("paths", {}).get("val_dir", "data/validation")
    resolved_weights = weights_path or config.get("paths", {}).get("weights", "checkpoints/best_model.pth")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset from: {resolved_input}")
    dataset = RestorationDataset(root_dir=resolved_input, config=config, split="val")
    loader = DataLoader(
        dataset,
        batch_size=config.get("training", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("data", {}).get("num_workers", 4),
        collate_fn=_collate_restoration_batch
    )
    
    logger.info("Building model and loading weights...")
    model = build_model(config).to(device)
    checkpoint = torch.load(resolved_weights, map_location=device)
    # Check if checkpoint has 'model_state', otherwise load directly
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Initializing metrics...")
    metrics_calc = RestorationMetrics(device=device, compute_lpips=True)
    
    total_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    batches = 0
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            
            # Use AMP if enabled in training
            use_amp = config.get("training", {}).get("mixed_precision", True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(lr)
            
            # Convert to float32 for metric calculation just in case
            pred = pred.float()
            
            metrics = metrics_calc(pred, hr)
            total_metrics["psnr"] += metrics["psnr"].item()
            total_metrics["ssim"] += metrics["ssim"].item()
            if "lpips" in metrics:
                total_metrics["lpips"] += metrics["lpips"].item()
            
            batches += 1
            
            # Save first batch images
            if batch_idx == 0:
                for i in range(min(4, lr.shape[0])):
                    comparison = torch.cat([lr[i], pred[i], hr[i]], dim=2)  # [C, H, W*3]
                    tv_utils.save_image(comparison, images_dir / f"compare_{i}.png")
    
    if batches > 0:
        for k in total_metrics:
            total_metrics[k] /= batches
            
    logger.info(f"Evaluation complete. Metrics: {total_metrics}")
    
    with open(out_path / "metrics.json", "w") as f:
        json.dump(total_metrics, f, indent=4)
        
    logger.info(f"Saved evaluation results to {output_dir}")


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
