"""Base model class for all AI models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import numpy as np
from PIL import Image

from fusion_ai.config import DEVICE, CACHE_DIR


class BaseModel(ABC):
    """Abstract base class for all AI models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize base model.

        Args:
            model_name: Name of the model
            device: Device to run inference on (cuda/cpu)
            cache_dir: Directory to cache models
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = device or DEVICE
        self.cache_dir = cache_dir or CACHE_DIR
        self.model = None
        self.model_kwargs = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def infer(self, input_data: Any, **kwargs) -> Any:
        """Run inference. Must be implemented by subclasses.

        Args:
            input_data: Input data for inference
            **kwargs: Additional inference parameters

        Returns:
            Inference results
        """
        pass

    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_size: Optional[tuple] = None,
        preserve_alpha: bool = True,
        mode: Optional[str] = None
    ) -> Union[Image.Image, tuple]:
        """Preprocess image for model inference.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            target_size: Optional target size (width, height)
            preserve_alpha: If True, preserve alpha channel separately
            mode: Target mode (RGB/RGBA/None for auto)

        Returns:
            Preprocessed PIL Image, or tuple of (RGB Image, alpha channel) if alpha preserved
        """
        alpha_channel = None

        if isinstance(image, (str, Path)):
            # Use load_image utility which handles EXR and alpha channels
            from fusion_ai.utils.image import load_image as util_load_image
            image = util_load_image(image, mode=None, preserve_alpha=preserve_alpha)
        elif isinstance(image, np.ndarray):
            # Handle numpy arrays - detect number of channels
            if image.ndim == 2:
                image = Image.fromarray(image).convert("L")
            elif image.ndim == 3:
                if image.shape[2] == 3:
                    image = Image.fromarray(image).convert("RGB")
                elif image.shape[2] == 4:
                    image = Image.fromarray(image).convert("RGBA")
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Extract and preserve alpha channel if requested
        if preserve_alpha and image.mode in ('RGBA', 'LA', 'PA'):
            # Extract alpha channel before conversion
            if image.mode == 'RGBA':
                alpha_channel = image.split()[3]  # Get alpha channel
                if target_size:
                    alpha_channel = alpha_channel.resize(target_size, Image.LANCZOS)

            # Convert to RGB if mode not specified
            if mode is None:
                mode = "RGB"

        # Convert to target mode
        if mode is None:
            mode = "RGB"
        if image.mode != mode:
            image = image.convert(mode)

        if target_size:
            image = image.resize(target_size, Image.LANCZOS)

        if alpha_channel is not None:
            return image, alpha_channel
        return image

    def postprocess_output(
        self,
        output: torch.Tensor,
        original_size: Optional[tuple] = None
    ) -> np.ndarray:
        """Postprocess model output.

        Args:
            output: Model output tensor
            original_size: Optional original image size for resizing

        Returns:
            Postprocessed numpy array
        """
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()

        if original_size and output.shape[-2:] != original_size:
            # Resize output to original size
            output_img = Image.fromarray(output.squeeze())
            output_img = output_img.resize(original_size[::-1], Image.LANCZOS)
            output = np.array(output_img)

        return output

    def to(self, device: str) -> "BaseModel":
        """Move model to specified device.

        Args:
            device: Target device (cuda/cpu)

        Returns:
            Self for chaining
        """
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
