# Left Things To Complete RestoreNet

This project currently has the repository structure, dataset loader, synthetic degradation pipeline, tests, and a compact NAFNet-style baseline. The remaining work is to turn the scaffold into a fully trainable, evaluable, and deployable restoration pipeline.

## Current Working Base

Done:

- Dataset discovery and image loading: `src/datasets/base.py`
- Train/validation transforms: `src/datasets/transforms.py`
- Synthetic degradation: `src/datasets/degradation.py`
- Restoration dataset and dataloaders: `src/datasets/restoration_dataset.py`, `src/datasets/dataloader.py`
- Compact NAFNet baseline and factory: `src/models/nafnet.py`, `src/models/__init__.py`
- Config loading: `src/utils/config.py`
- Project configs: `configs/train.yaml`, `configs/inference.yaml`
- Tests: `tests/`

Verified:

```bash
pytest tests/ -v
```

Expected result at last check: `23 passed`.

## 1. Loss Functions

Where to work:

- `src/losses/__init__.py`
- Suggested new file: `src/losses/restoration_loss.py`
- Suggested tests: `tests/test_losses.py`

What to implement:

- `L1Loss` using `torch.nn.functional.l1_loss`
- MS-SSIM loss using `torchmetrics` or a small dependency-compatible wrapper
- LPIPS loss using the `lpips` package
- Edge loss using Sobel filters or Laplacian filters
- FFT loss using magnitude difference in frequency space
- `RestorationLoss` class that combines the configured weights from `configs/train.yaml`

Minimum usable API:

```python
from src.losses import RestorationLoss

criterion = RestorationLoss(config["loss"])
loss, loss_parts = criterion(pred, target)
```

Done when:

- A forward pass returns one scalar loss tensor
- Individual loss parts are returned for logging
- Unit tests verify shape, finite values, and zero/small loss for identical tensors

## 2. Training Loop

Where to work:

- `src/train.py`
- Suggested helper file: `src/utils/checkpoint.py`
- Suggested tests: `tests/test_training_smoke.py`

What to implement:

- Build model with `build_model(config)`
- Move model and batches to configured device
- Optimizer: `torch.optim.AdamW`
- Scheduler: cosine decay with optional warmup
- Mixed precision using `torch.cuda.amp` when CUDA is available
- Gradient clipping from `training.grad_clip_norm`
- Train/validation epoch loops
- Checkpoint saving to `checkpoints/`
- Resume from `--resume`
- Best checkpoint based on `checkpoint.monitor_metric`
- TensorBoard logging
- Optional W&B logging, guarded by `wandb.enabled`

Minimum CLI target:

```bash
python -m src.train --config configs/train.yaml
```

Done when:

- Training runs on a small folder of images without crashing
- `checkpoints/best_model.pth` is created
- Loss decreases on a tiny overfit test set

## 3. Metrics And Evaluation

Where to work:

- `src/metrics/__init__.py`
- Suggested new file: `src/metrics/restoration_metrics.py`
- `src/evaluate.py`
- Suggested tests: `tests/test_metrics.py`, `tests/test_evaluate_smoke.py`

What to implement:

- PSNR
- SSIM
- LPIPS
- Batch metric aggregation
- Load model checkpoint
- Run validation/test data through model
- Save metrics JSON/CSV under `outputs/eval/`
- Optionally save side-by-side comparison images

Minimum CLI target:

```bash
python -m src.evaluate --config configs/train.yaml --weights checkpoints/best_model.pth --output outputs/eval
```

Done when:

- Evaluation prints and saves PSNR, SSIM, and LPIPS
- It works from a fresh terminal command

## 4. Inference CLI

Where to work:

- `src/inference.py`
- Suggested helper file: `src/utils/image_io.py`
- Suggested tests: `tests/test_inference_smoke.py`

What to implement:

- Load config from `configs/inference.yaml`
- Load model weights
- Read input images from `paths.input_dir` or `--input`
- Convert images to tensors in `[0, 1]`
- Run model in `torch.no_grad()` / inference mode
- Save restored images to `paths.output_dir` or `--output`
- Support PNG output first
- Preserve filenames or add `_restored` suffix

Minimum CLI target:

```bash
python -m src.inference --config configs/inference.yaml --weights checkpoints/best_model.pth --input data/test --output outputs/predictions
```

Done when:

- It restores every supported image in `data/test/`
- It writes visible images to `outputs/predictions/`
- It handles missing weights/input paths with clear errors

## 5. Benchmarking And Optimization

Where to work:

- `src/inference.py`
- Suggested new file: `src/utils/benchmark.py`
- Optional new files: `src/export_onnx.py`, `src/export_tensorrt.py`

What to implement after training/evaluation work:

- Warmup and timed inference runs from `configs/inference.yaml`
- End-to-end timing: load model, load image, inference, save output
- Optional `torch.compile()` path
- FP16 inference on CUDA
- ONNX export
- TensorRT export path for H100 deployment

Done when:

- Benchmark results are reproducible and saved to `outputs/benchmark.json`
- Quality metrics are not broken by optimization

## Recommended Completion Order

1. Implement `RestorationLoss`
2. Implement basic training loop without W&B first
3. Add checkpoint save/resume
4. Add PSNR and SSIM metrics
5. Implement evaluation CLI
6. Implement inference CLI
7. Add LPIPS to loss and metrics
8. Add TensorBoard/W&B logging
9. Add benchmarking
10. Add ONNX/TensorRT optimization only after the PyTorch path is stable

## Data Needed To Make It Work

Place clean high-resolution training images here:

```text
data/train/
data/validation/
```

For quick smoke testing, even 5-10 clean images are enough. For real training, add a larger, representative scientific image dataset.

For final inference, place degraded test images here:

```text
data/test/
```

## Practical First Smoke Test

After implementing loss and training, use a tiny dataset first:

```bash
python -m src.train --config configs/train.yaml
python -m src.evaluate --config configs/train.yaml --weights checkpoints/best_model.pth --output outputs/eval
python -m src.inference --config configs/inference.yaml --weights checkpoints/best_model.pth --input data/test --output outputs/predictions
```

If the tiny dataset cannot overfit, fix the training loop or loss before scaling up.

## Known Notes

- `configs/train.yaml` currently defaults to `device: cuda`; switch to `cpu` if CUDA is unavailable.
- `wandb.enabled` is currently `true`; set it to `false` until W&B logging is wired or authenticated.
- `data/` keeps only `.gitkeep` files in Git; real datasets are intentionally ignored by `.gitignore`.
- `outputs/` and `checkpoints/` are ignored by Git except `.gitkeep` placeholders.
