# Left Things To Complete RestoreNet

## Completed Now

- Dataset loading and synthetic degradation
- Compact NAFNet baseline
- Combined restoration loss in `src/losses/restoration_loss.py`
- Basic train/validation loop in `src/train.py`
- AdamW optimizer
- Cosine scheduler with optional warmup
- Mixed precision on CUDA
- Gradient clipping
- Checkpoint saving: `best_model.pth`, `last_model.pth`, optional epoch checkpoints
- Resume support through `python -m src.train --resume checkpoints/last_model.pth`
- Training smoke test and loss tests

Verified:

```bash
pytest tests/ -v
```

Latest result: `27 passed`.

## What Is Still Left

### 1. Metrics

Where:

- `src/metrics/__init__.py`
- Suggested file: `src/metrics/restoration_metrics.py`
- Suggested tests: `tests/test_metrics.py`

Build:

- PSNR
- SSIM
- LPIPS metric, optional until the environment is ready
- Batch metric aggregation

### 2. Evaluation CLI

Where:

- `src/evaluate.py`
- Suggested tests: `tests/test_evaluate_smoke.py`

Build:

- Load `checkpoints/best_model.pth`
- Run validation/test images through the model
- Compute PSNR, SSIM, and LPIPS if enabled
- Save `outputs/eval/metrics.json`
- Optionally save comparison images

Command target:

```bash
python -m src.evaluate --config configs/train.yaml --weights checkpoints/best_model.pth --output outputs/eval
```

### 3. Inference CLI

Where:

- `src/inference.py`
- Suggested helper: `src/utils/image_io.py`
- Suggested tests: `tests/test_inference_smoke.py`

Build:

- Load model from `configs/inference.yaml`
- Load checkpoint weights
- Read degraded images from `data/test/` or `--input`
- Restore images with `torch.no_grad()`
- Save PNG outputs to `outputs/predictions/`

Command target:

```bash
python -m src.inference --config configs/inference.yaml --weights checkpoints/best_model.pth --input data/test --output outputs/predictions
```

### 4. Better Training Features

Where:

- `src/train.py`
- Optional helper: `src/utils/checkpoint.py`

Build:

- TensorBoard logging
- W&B logging after authentication
- Early stopping
- Best metric monitoring after SSIM is implemented
- Cleaner checkpoint helper module

### 5. Better Model Quality

Where:

- `src/models/nafnet.py`
- Optional new model files under `src/models/`

Build:

- Encoder-decoder NAFNet structure
- More skip connections
- Configurable block counts by stage
- Larger model presets for real training

### 6. More Realistic Degradation

Where:

- `src/datasets/degradation.py`

Build:

- JPEG compression artifacts
- Random blur kernels
- Sensor-like noise
- Variable downscale factors
- Degradation presets for train/validation

### 7. Optimization

Where:

- `src/inference.py`
- Suggested file: `src/utils/benchmark.py`
- Optional files: `src/export_onnx.py`, `src/export_tensorrt.py`

Build after PyTorch inference works:

- Benchmark JSON report
- `torch.compile()` path
- FP16 inference
- ONNX export
- TensorRT export for H100

## Best Next Step

Implement metrics first, then evaluation, then inference.

Recommended order:

1. Add `src/metrics/restoration_metrics.py`
2. Make `src/evaluate.py` work with `best_model.pth`
3. Make `src/inference.py` save restored images
4. Add TensorBoard and early stopping
5. Improve model architecture
6. Add optimization/export only after the PyTorch path is stable

## Data Needed

Training needs clean high-resolution images in:

```text
data/train/
data/validation/
```

Inference needs degraded images in:

```text
data/test/
```

For a first smoke test, 5-10 clean images are enough. For real results, use a larger scientific image dataset.
