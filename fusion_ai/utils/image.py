"""Image utility functions."""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


def load_image(
    image_path: Union[str, Path],
    mode: str = "RGB"
) -> Image.Image:
    """Load an image from file.

    Args:
        image_path: Path to image file
        mode: Image mode (RGB, L, etc.)

    Returns:
        PIL Image
    """
    image = Image.open(image_path)
    if mode:
        image = image.convert(mode)
    return image


def save_image(
    image: Union[Image.Image, np.ndarray],
    output_path: Union[str, Path],
    format: Optional[str] = None
) -> None:
    """Save an image to file.

    Args:
        image: Image to save (PIL Image or numpy array)
        output_path: Output file path
        format: Image format (optional, inferred from path if not provided)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if format:
        image.save(output_path, format=format)
    else:
        image.save(output_path)


def resize_image(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int],
    resample: int = Image.LANCZOS,
    return_numpy: bool = False
) -> Union[Image.Image, np.ndarray]:
    """Resize an image.

    Args:
        image: Input image
        size: Target size (width, height)
        resample: Resampling filter
        return_numpy: Whether to return numpy array

    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    resized = image.resize(size, resample)

    if return_numpy:
        return np.array(resized)
    return resized


def normalize_array(
    array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 255.0,
    dtype: type = np.uint8
) -> np.ndarray:
    """Normalize array values to specified range.

    Args:
        array: Input array
        min_val: Minimum output value
        max_val: Maximum output value
        dtype: Output data type

    Returns:
        Normalized array
    """
    array_min = array.min()
    array_max = array.max()

    if array_max - array_min == 0:
        return np.zeros_like(array, dtype=dtype)

    normalized = (array - array_min) / (array_max - array_min)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized.astype(dtype)


def apply_colormap(
    depth_map: np.ndarray,
    colormap: str = "turbo"
) -> np.ndarray:
    """Apply colormap to depth map.

    Args:
        depth_map: Input depth map
        colormap: Matplotlib colormap name

    Returns:
        Colored depth map (RGB)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # Normalize depth values
        norm = Normalize(vmin=depth_map.min(), vmax=depth_map.max())

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(norm(depth_map))

        # Convert to uint8 RGB
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        return colored_rgb

    except ImportError:
        print("matplotlib not available, returning grayscale depth map")
        return depth_map
