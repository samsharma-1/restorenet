"""Synthetic degradation functions for restoration training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class DegradationConfig:
    """Parameters controlling synthetic image degradation."""

    downscale_factor: int = 2
    speckle_var: float = 0.05
    blur_sigma: float = 1.0
    gaussian_noise_std: float = 0.02
    jpeg_quality: int = 90
    poisson_scale: float = 0.0
    apply_speckle: bool = True
    apply_blur: bool = True
    apply_noise: bool = True
    apply_downscale: bool = True
    apply_jpeg: bool = False
    apply_poisson: bool = False


def apply_speckle_noise(
    image: np.ndarray,
    variance: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply multiplicative speckle noise.

    The degradation follows ``output = image + image * n`` where ``n`` is
    zero-mean Gaussian noise with the given variance.

    Args:
        image: Float array (H, W, C) in [0, 1].
        variance: Speckle variance (must be >= 0).
        rng: Optional NumPy random generator.

    Returns:
        Degraded image clipped to [0, 1].
    """
    if variance <= 0:
        return image.copy()

    generator = rng or np.random.default_rng()
    noise = generator.normal(0.0, np.sqrt(variance), size=image.shape).astype(
        np.float32
    )
    degraded = image * (1.0 + noise)
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply isotropic Gaussian blur.

    Args:
        image: Float array (H, W, C) in [0, 1].
        sigma: Gaussian kernel standard deviation.

    Returns:
        Blurred image in [0, 1].
    """
    if sigma <= 0:
        return image.copy()

    kernel_size = int(2 * np.ceil(3.0 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def apply_gaussian_noise(
    image: np.ndarray,
    std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply additive Gaussian noise.

    Args:
        image: Float array (H, W, C) in [0, 1].
        std: Standard deviation of additive noise.
        rng: Optional NumPy random generator.

    Returns:
        Noisy image clipped to [0, 1].
    """
    if std <= 0:
        return image.copy()

    generator = rng or np.random.default_rng()
    noise = generator.normal(0.0, std, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    if quality >= 100 or quality <= 0:
        return image.copy()
    
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img_u8, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    return (decimg / 255.0).astype(np.float32)


def apply_poisson_noise(image: np.ndarray, scale: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply Poisson (sensor-like shot) noise."""
    if scale <= 0:
        return image.copy()
        
    generator = rng or np.random.default_rng()
    scaled = np.clip(image * scale, 0, None)
    noisy = generator.poisson(scaled) / scale
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def apply_downscale(
    image: np.ndarray,
    factor: int = 2,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Downscale an image by an integer factor.

    Args:
        image: Float array (H, W, C) in [0, 1].
        factor: Integer downscale factor (>= 1).
        interpolation: OpenCV interpolation flag.

    Returns:
        Downscaled array with spatial dimensions divided by ``factor``.
    """
    if factor <= 1:
        return image.copy()

    height, width = image.shape[:2]
    new_height = max(1, height // factor)
    new_width = max(1, width // factor)
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    ).astype(np.float32)


def sample_degradation_params(
    degradation_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> DegradationConfig:
    """Sample concrete degradation parameters from a config dict.

    Supports fixed values or ``{min, max}`` ranges for random sampling.
    """

    def _sample(key: str, default: float) -> float:
        value = degradation_cfg.get(key, default)
        if isinstance(value, dict):
            return float(rng.uniform(value["min"], value["max"]))
        return float(value)

    def _sample_bool(key: str, default: float) -> bool:
        probability = float(degradation_cfg.get("apply_prob", {}).get(key, default))
        return bool(rng.random() < probability)

    return DegradationConfig(
        downscale_factor=int(degradation_cfg.get("downscale_factor", 2)),
        speckle_var=_sample("speckle_var", 0.05),
        blur_sigma=_sample("blur_sigma", 1.0),
        gaussian_noise_std=_sample("gaussian_noise_std", 0.02),
        jpeg_quality=int(_sample("jpeg_quality", 90)),
        poisson_scale=_sample("poisson_scale", 100.0),
        apply_speckle=_sample_bool("speckle", 1.0),
        apply_blur=_sample_bool("blur", 1.0),
        apply_noise=_sample_bool("noise", 0.5),
        apply_downscale=_sample_bool("downscale", 1.0),
        apply_jpeg=_sample_bool("jpeg", 0.0),
        apply_poisson=_sample_bool("poisson", 0.0),
    )


def apply_synthetic_degradation(
    image: np.ndarray,
    params: DegradationConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full synthetic degradation pipeline.

    Degradations are applied on the high-resolution image in the order:
    blur -> additive Gaussian noise -> speckle -> downscale.

    Args:
        image: Clean HR array (H, W, C) in [0, 1].
        params: Degradation parameters.
        rng: Optional random generator for stochastic steps.

    Returns:
        Tuple of ``(lr_degraded, hr_clean)`` where ``hr_clean`` is the input
        image unchanged and ``lr_degraded`` is the degraded low-resolution view.
    """
    generator = rng or np.random.default_rng()
    degraded = image.copy()

    if params.apply_poisson:
        degraded = apply_poisson_noise(degraded, params.poisson_scale, rng=generator)
    if params.apply_blur:
        degraded = apply_gaussian_blur(degraded, params.blur_sigma)
    if params.apply_noise:
        degraded = apply_gaussian_noise(
            degraded, params.gaussian_noise_std, rng=generator
        )
    if params.apply_speckle:
        degraded = apply_speckle_noise(degraded, params.speckle_var, rng=generator)
    if params.apply_jpeg:
        degraded = apply_jpeg_compression(degraded, params.jpeg_quality)

    if params.apply_downscale:
        lr = apply_downscale(degraded, factor=params.downscale_factor)
    else:
        lr = degraded

    return lr.astype(np.float32), image.astype(np.float32)
