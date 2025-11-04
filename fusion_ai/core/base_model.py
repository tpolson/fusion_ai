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
        target_size: Optional[tuple] = None
    ) -> Image.Image:
        """Preprocess image for model inference.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            target_size: Optional target size (width, height)

        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if target_size:
            image = image.resize(target_size, Image.LANCZOS)

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
