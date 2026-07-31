"""Base utilities for image dataset discovery and loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)


def discover_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Discover image files under a directory.

    Args:
        directory: Root directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no images are found.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory, got: {root}")

    pattern = "**/*" if recursive else "*"
    paths = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()

    if not paths:
        raise ValueError(
            f"No images found in {root}. Supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    return paths


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a float32 RGB array in [0, 1].

    Args:
        path: Path to the image file.

    Returns:
        Array of shape (H, W, C) with C=3.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be decoded.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    image_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to decode image: {file_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def resize_to_min_side(image: np.ndarray, min_side: int) -> np.ndarray:
    """Resize an image so its shorter side equals ``min_side``.

    Preserves aspect ratio. Used for mixed-resolution training so every
    sample is large enough for cropping without distorting content.

    Args:
        image: Input array (H, W, C) in [0, 1].
        min_side: Target length of the shorter spatial dimension.

    Returns:
        Resized array (H', W', C).
    """
    height, width = image.shape[:2]
    if min(height, width) == min_side:
        return image

    scale = min_side / float(min(height, width))
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
