"""Inference entry point for restoring degraded images."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as tv_utils
from torchvision.transforms.functional import to_tensor

from src.models import build_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """Simple dataset for loading images for inference."""
    
    def __init__(self, input_dir: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        self.input_dir = Path(input_dir)
        self.image_paths = [
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_paths[idx]
        # Read image using OpenCV (BGR to RGB)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float and [0, 1] range, then to tensor [C, H, W]
        img_tensor = to_tensor(img)
        
        return {
            "image": img_tensor,
            "path": str(path),
            "filename": path.name
        }


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
    """
    paths = config.get("paths", {})
    resolved_weights = weights_path or paths.get("weights", "checkpoints/best_model.pth")
    resolved_input = input_dir or paths.get("input_dir", "data/test")
    resolved_output = output_dir or paths.get("output_dir", "outputs/predictions")
    
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    fp16 = bool(config.get("fp16", True))
    
    logger.info(f"Inference config: weights={resolved_weights}, input={resolved_input}, output={resolved_output}")
    
    out_path = Path(resolved_output)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    exts_list = config.get("data", {}).get("extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    dataset = InferenceDataset(resolved_input, extensions=tuple(exts_list))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Load model
    logger.info("Building model and loading weights...")
    model = build_model(config).to(device)
    checkpoint = torch.load(resolved_weights, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Running inference on {len(dataset)} images...")
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            filenames = batch["filename"]
            
            with torch.amp.autocast(device_type=device.type, enabled=fp16):
                preds = model(imgs)
                
            # Clamp to valid range
            preds = torch.clamp(preds, 0.0, 1.0)
            
            for i, filename in enumerate(filenames):
                save_path = out_path / filename
                # Save the image
                tv_utils.save_image(preds[i], save_path)
                logger.info(f"Saved {save_path}")

    logger.info("Inference complete.")


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
