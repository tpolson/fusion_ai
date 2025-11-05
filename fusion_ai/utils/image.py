"""Image utility functions."""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


def load_image(
    image_path: Union[str, Path],
    mode: Optional[str] = "RGB",
    preserve_alpha: bool = True
) -> Image.Image:
    """Load an image from file.

    Args:
        image_path: Path to image file
        mode: Image mode (RGB, RGBA, L, etc.). If None, keeps original mode
        preserve_alpha: If True and image has alpha, convert to RGBA instead of RGB

    Returns:
        PIL Image
    """
    image = Image.open(image_path)

    # Preserve alpha channel if requested
    if preserve_alpha and image.mode in ('RGBA', 'LA', 'PA') and mode == "RGB":
        mode = "RGBA"

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


def load_exr(
    exr_path: Union[str, Path],
    channels: Optional[list] = None
) -> np.ndarray:
    """Load an OpenEXR file.

    Args:
        exr_path: Path to EXR file
        channels: List of channel names to load (default: ['R', 'G', 'B'])

    Returns:
        Numpy array in fp32 format
    """
    try:
        import OpenEXR
        import Imath
        import array

        exr_path = str(exr_path)
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Default to RGB channels
        if channels is None:
            available_channels = header['channels'].keys()
            if 'A' in available_channels:
                channels = ['R', 'G', 'B', 'A']
            else:
                channels = ['R', 'G', 'B']

        # Read channels
        channel_data = []
        for channel in channels:
            if channel in header['channels']:
                channel_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                channel_array = array.array('f', channel_str)
                channel_data.append(np.array(channel_array).reshape(height, width))

        # Stack channels
        if len(channel_data) == 1:
            return channel_data[0].astype(np.float32)
        else:
            return np.stack(channel_data, axis=-1).astype(np.float32)

    except ImportError:
        raise ImportError("OpenEXR package required. Install with: pip install OpenEXR")


def save_exr(
    image: np.ndarray,
    output_path: Union[str, Path],
    channels: Optional[list] = None
) -> None:
    """Save image as OpenEXR file.

    Args:
        image: Image array (fp16 or fp32)
        output_path: Output file path
        channels: Channel names (default: ['R', 'G', 'B'] or ['R', 'G', 'B', 'A'])
    """
    try:
        import OpenEXR
        import Imath

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to float32 if needed
        if image.dtype == np.float16:
            image = image.astype(np.float32)

        # Determine channels
        if image.ndim == 2:
            # Single channel (e.g., depth map)
            height, width = image.shape
            if channels is None:
                channels = ['Y']
            channel_data = {'Y': image.tobytes()}
        else:
            # Multi-channel
            height, width, num_channels = image.shape
            if channels is None:
                if num_channels == 3:
                    channels = ['R', 'G', 'B']
                elif num_channels == 4:
                    channels = ['R', 'G', 'B', 'A']
                else:
                    channels = [f'C{i}' for i in range(num_channels)]

            channel_data = {}
            for i, channel_name in enumerate(channels[:num_channels]):
                channel_data[channel_name] = image[:, :, i].tobytes()

        # Create header
        header = OpenEXR.Header(width, height)
        float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = {name: float_chan for name in channels}

        # Write file
        out = OpenEXR.OutputFile(str(output_path), header)
        out.writePixels(channel_data)
        out.close()

    except ImportError:
        raise ImportError("OpenEXR package required. Install with: pip install OpenEXR")


def image_to_float(
    image: Union[Image.Image, np.ndarray],
    dtype: type = np.float32
) -> np.ndarray:
    """Convert image to floating point format (fp16 or fp32).

    Args:
        image: Input image (PIL Image or numpy array)
        dtype: Target dtype (np.float16 or np.float32)

    Returns:
        Float array normalized to [0, 1]
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to float and normalize based on input type
    if image.dtype == np.uint8:
        return (image / 255.0).astype(dtype)
    elif image.dtype == np.uint16:
        return (image / 65535.0).astype(dtype)
    elif image.dtype in (np.float16, np.float32, np.float64):
        return image.astype(dtype)
    else:
        # For other types, just convert
        return image.astype(dtype)


def float_to_image(
    array: np.ndarray,
    output_dtype: type = np.uint8,
    clip: bool = True
) -> np.ndarray:
    """Convert floating point array to image format.

    Args:
        array: Float array (assumed to be in [0, 1] range)
        output_dtype: Target dtype (np.uint8 or np.uint16)
        clip: Whether to clip values to valid range

    Returns:
        Image array in target dtype
    """
    if clip:
        array = np.clip(array, 0, 1)

    if output_dtype == np.uint8:
        return (array * 255.0).astype(np.uint8)
    elif output_dtype == np.uint16:
        return (array * 65535.0).astype(np.uint16)
    else:
        return array.astype(output_dtype)
