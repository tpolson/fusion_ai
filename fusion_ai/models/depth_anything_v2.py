"""DepthAnythingV2 model implementation."""

from pathlib import Path
from typing import Optional, Union, Literal
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import MODEL_CONFIGS, CACHE_DIR
from fusion_ai.utils.download import download_model


class DepthAnythingV2(BaseModel):
    """DepthAnythingV2 model for monocular depth estimation."""

    MODEL_SIZES = Literal["small", "base", "large"]

    def __init__(
        self,
        model_size: MODEL_SIZES = "base",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize DepthAnythingV2 model.

        Args:
            model_size: Model size (small/base/large)
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        self.model_size = model_size
        model_name = f"depth_anything_v2_{model_size}"
        super().__init__(model_name, device, cache_dir, **kwargs)

        self.config = MODEL_CONFIGS["depth_anything_v2"][model_size]
        self.load_model()

    def load_model(self) -> None:
        """Load the DepthAnythingV2 model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            print(f"Loading DepthAnythingV2 {self.model_size} model...")

            # Load processor and model from HuggingFace
            self.processor = AutoImageProcessor.from_pretrained(
                self.config["repo_id"],
                cache_dir=str(self.cache_dir)
            )

            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.config["repo_id"],
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"DepthAnythingV2 {self.model_size} loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers is required for DepthAnythingV2. "
                "Install it with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DepthAnythingV2 model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        normalize: bool = True,
        colormap: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate depth map from input image.

        Args:
            image: Input image
            normalize: Whether to normalize depth values to 0-255
            colormap: Optional OpenCV colormap name (e.g., 'COLORMAP_INFERNO')
            **kwargs: Additional inference parameters

        Returns:
            Depth map as numpy array
        """
        # Preprocess image
        pil_image = self.preprocess_image(image)
        original_size = pil_image.size

        # Prepare inputs
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()

        # Normalize if requested
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

        # Apply colormap if requested
        if colormap and hasattr(cv2, colormap):
            colormap_id = getattr(cv2, colormap)
            depth = cv2.applyColorMap(depth.astype(np.uint8), colormap_id)

        return depth

    def infer_batch(
        self,
        images: list,
        normalize: bool = True,
        **kwargs
    ) -> list:
        """Generate depth maps for multiple images.

        Args:
            images: List of input images
            normalize: Whether to normalize depth values
            **kwargs: Additional inference parameters

        Returns:
            List of depth maps
        """
        results = []
        for image in images:
            depth = self.infer(image, normalize=normalize, **kwargs)
            results.append(depth)
        return results

    def save_depth(
        self,
        depth: np.ndarray,
        output_path: Union[str, Path],
        format: str = "png"
    ) -> None:
        """Save depth map to file.

        Args:
            depth: Depth map array
            output_path: Output file path
            format: Image format (png, exr, etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "exr":
            # Save as OpenEXR for full precision
            try:
                import OpenEXR
                import Imath

                height, width = depth.shape
                header = OpenEXR.Header(width, height)
                half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                header['channels'] = {'Z': half_chan}

                out = OpenEXR.OutputFile(str(output_path), header)
                out.writePixels({'Z': depth.astype(np.float32).tobytes()})
                out.close()
            except ImportError:
                print("OpenEXR not available, saving as PNG instead")
                Image.fromarray(depth.astype(np.uint8)).save(output_path)
        else:
            Image.fromarray(depth.astype(np.uint8)).save(output_path)
