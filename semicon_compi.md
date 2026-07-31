# KLA AI Hackathon 2026

# AI-Based Restoration of Degraded Images

> Restore degraded scientific images by jointly removing speckle noise,
> recovering sharpness, and performing super-resolution while maximizing
> restoration quality and minimizing inference latency.

------------------------------------------------------------------------

# Objective

Develop a deep learning pipeline capable of restoring degraded
scientific images.

The model must simultaneously:

-   Remove multiplicative speckle noise
-   Recover image sharpness degraded by Gaussian blur/noise
-   Perform 2× super-resolution
-   Generalize to unseen datasets
-   Achieve high SSIM, PSNR and low LPIPS
-   Execute efficiently on NVIDIA H100 GPU

------------------------------------------------------------------------

# Competition Goals

## Primary Metrics

-   SSIM ↑
-   PSNR ↑
-   LPIPS ↓

## Secondary Metric

-   End-to-end inference time
    -   Model loading
    -   Image loading
    -   Inference
    -   Output saving

The objective is to maximize image quality while minimizing runtime.

------------------------------------------------------------------------

# Technology Stack

-   Python 3.11+
-   PyTorch
-   TorchVision
-   OpenCV
-   NumPy
-   Albumentations
-   TorchMetrics
-   LPIPS
-   TensorBoard
-   Weights & Biases
-   ONNX
-   TensorRT
-   CUDA
-   Git

------------------------------------------------------------------------

# Project Structure

``` text
project-root/
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── outputs/
├── checkpoints/
├── tests/
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# Development Roadmap

## Phase 1

-   Dataset exploration
-   Image statistics
-   Histograms
-   FFT analysis
-   Noise estimation
-   Resolution verification

## Phase 2

-   Dataset loader
-   Augmentations
-   Synthetic degradation
-   Mixed-resolution training

## Phase 3

-   Implement NAFNet baseline
-   Train on a small subset

## Phase 4

Implement combined losses: - L1 - MS-SSIM - LPIPS - Edge Loss - FFT Loss

## Phase 5

Training pipeline: - Mixed precision - Checkpoints - Early stopping -
Cosine scheduler - TensorBoard - Weights & Biases

## Phase 6

Validation: - SSIM - PSNR - LPIPS - Comparison reports

## Phase 7

Inference:

``` bash
python evaluate.py \
  --input test_images \
  --output predictions \
  --weights best_model.pth
```

## Phase 8

Optimization: - torch.compile() - FP16 - ONNX - TensorRT - Benchmarking

------------------------------------------------------------------------

# Coding Guidelines

-   Follow PEP8
-   Use type hints
-   Modular code
-   Reusable functions
-   Add docstrings
-   Include unit tests where appropriate

------------------------------------------------------------------------

# Git Branches

-   main
-   development
-   feature/dataset
-   feature/model
-   feature/loss
-   feature/train
-   feature/inference
-   feature/optimization

------------------------------------------------------------------------

# Instructions for Codex

1.  Analyze the existing repository before making changes.
2.  Keep implementations modular.
3.  Avoid duplicate code.
4.  Preserve compatibility with PyTorch 2.x.
5.  Add error handling.
6.  Maintain consistent coding style.
7.  Update documentation if architecture changes.

------------------------------------------------------------------------

# Initial Priority

1.  Repository setup
2.  Dataset loader
3.  Augmentation pipeline
4.  NAFNet implementation
5.  Loss functions
6.  Training loop
7.  Validation
8.  Inference
9.  Optimization
10. Documentation

------------------------------------------------------------------------

# Definition of Done

-   Training completes successfully.
-   Validation metrics are generated.
-   Evaluation script works from the command line.
-   Test images are restored correctly.
-   Benchmark results are reproducible.
-   Documentation is complete.
