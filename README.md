# KLA AI Hackathon 2026 â€” AI-Based Restoration of Degraded Images

Restore degraded scientific images by jointly removing speckle noise, recovering sharpness, and performing 2Ã— super-resolution while maximizing restoration quality (SSIM, PSNR, low LPIPS) and minimizing inference latency on NVIDIA H100.

## Goals

- Remove multiplicative speckle noise
- Recover sharpness degraded by Gaussian blur/noise
- Perform 2Ã— super-resolution
- Generalize to unseen datasets
- Optimize end-to-end inference time

## Setup

Requires **Python 3.11+**.

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Place training, validation, and test images under `data/train/`, `data/validation/`, and `data/test/` respectively.

## Dataset Layout

```
data/train/        # Clean high-resolution images for training
data/validation/   # Clean HR images for validation
data/test/         # Held-out images for final evaluation
```

Supported formats: PNG, JPEG, TIFF, BMP, WebP.

### Synthetic degradation (training)

By default (`data.mode: on_the_fly`), the loader reads **clean HR images** and applies degradations on the fly:

1. **Gaussian blur** (random Ïƒ)
2. **Additive Gaussian noise** (optional, random std)
3. **Multiplicative speckle noise**
4. **2Ã— bicubic downscale** â†’ low-resolution input

The model receives an LR/degraded tensor (`lr`) and learns to predict the clean HR target (`hr`). Pixel values stay in **[0, 1]** (no ImageNet normalization).

For pre-computed pairs, use paired mode:

```
data/train/
  clean/img001.png
  degraded/img001.png
```

Configure paths, batch size, workers, degradation ranges, and augmentations in `configs/train.yaml`.

### Quick dataset API

```python
from src.utils.config import load_config
from src.datasets import build_dataloaders, RestorationDataset

config = load_config("configs/train.yaml")
train_loader, val_loader = build_dataloaders(config)

batch = next(iter(train_loader))
lr, hr = batch["lr"], batch["hr"]  # shapes: (B, 3, H/2, W/2), (B, 3, H, W)
```

Mixed-resolution images are handled by resizing the shorter side to at least `data.image_size` before cropping, preserving aspect ratio.

## Project Structure

```
project-root/
â”œâ”€â”€ configs/           # YAML configs for training and inference
â”œâ”€â”€ data/              # Dataset directories (train / val / test)
â”œâ”€â”€ notebooks/         # EDA and experimentation
â”œâ”€â”€ src/               # Source package
â”‚   â”œâ”€â”€ datasets/      # Loaders, augmentations, synthetic degradation
â”‚   â”œâ”€â”€ models/        # NAFNet and variants (Phase 3)
â”‚   â”œâ”€â”€ losses/        # Combined loss functions (Phase 4)
â”‚   â”œâ”€â”€ metrics/       # SSIM, PSNR, LPIPS (Phase 6)
â”‚   â”œâ”€â”€ utils/         # Config, logging, I/O helpers
â”‚   â”œâ”€â”€ train.py       # Training CLI
â”‚   â”œâ”€â”€ evaluate.py    # Evaluation CLI
â”‚   â””â”€â”€ inference.py   # Inference CLI
â”œâ”€â”€ outputs/           # Logs, predictions, reports
â”œâ”€â”€ checkpoints/       # Saved model weights
â””â”€â”€ tests/             # Unit tests
```

## Roadmap

| Phase | Focus |
|-------|-------|
| 1 | Dataset exploration (EDA) |
| 2 | Dataset loader, augmentations, synthetic degradation âœ“ |
| 3 | NAFNet baseline implementation |
| 4 | Combined losses (L1, MS-SSIM, optional LPIPS, edge, FFT) ✓ |
| 5 | Basic training pipeline (AMP, checkpoints, resume) ✓ |
| 6 | Validation metrics and comparison reports |
| 7 | Inference CLI |
| 8 | Optimization (torch.compile, FP16, ONNX, TensorRT) |

## Usage (future)

Training is implemented for the baseline model. Evaluation and inference entry points are still scaffolded for later phases.

```bash
# Training (Phase 5)
python -m src.train --config configs/train.yaml

# Evaluation (Phase 6)
python -m src.evaluate --weights checkpoints/best_model.pth --input data/test --output outputs/eval

# Inference (Phase 7)
python -m src.inference --config configs/inference.yaml --weights checkpoints/best_model.pth
```

## Testing

```bash
pytest tests/ -v
```


