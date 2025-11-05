"""Inpainting model for object removal and image completion."""

from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class LamaInpainting(BaseModel):
    """LaMa inpainting model for fast, high-quality object removal."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize LaMa inpainting model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        model_name = "lama_inpainting"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the LaMa inpainting model."""
        try:
            from simple_lama_inpainting import SimpleLama

            print(f"Loading LaMa inpainting model...")

            # Initialize SimpleLama
            self.model = SimpleLama()

            print(f"LaMa inpainting loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "simple-lama-inpainting is required for LaMa. "
                "Install it with: pip install simple-lama-inpainting"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LaMa model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Perform inpainting to remove objects.

        Args:
            image: Input image
            mask: Binary mask (white = remove, black = keep)
            **kwargs: Additional inference parameters

        Returns:
            Inpainted image as numpy array
        """
        # Preprocess image and mask
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        pil_mask = self.preprocess_image(mask, preserve_alpha=False)

        # Convert mask to binary (ensure it's single channel)
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')

        # Perform inpainting
        result = self.model(pil_image, pil_mask)

        # Convert to numpy array
        if isinstance(result, Image.Image):
            result = np.array(result)

        return result


class StableDiffusionInpainting(BaseModel):
    """Stable Diffusion inpainting for prompt-based object removal."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize Stable Diffusion inpainting model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "stable_diffusion_inpainting"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the Stable Diffusion inpainting model."""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print(f"Loading Stable Diffusion inpainting model...")

            # Load the inpainting pipeline
            model_id = "stabilityai/stable-diffusion-2-inpainting"

            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"Stable Diffusion inpainting loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for Stable Diffusion. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Diffusion model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "",
        negative_prompt: str = "blurry, bad quality, distorted",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> np.ndarray:
        """Perform inpainting with prompt guidance.

        Args:
            image: Input image
            mask: Binary mask (white = remove, black = keep)
            prompt: Text prompt for inpainting (empty = automatic background)
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            **kwargs: Additional inference parameters

        Returns:
            Inpainted image as numpy array
        """
        # Preprocess image and mask
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        pil_mask = self.preprocess_image(mask, preserve_alpha=False)

        # Convert mask to binary
        if pil_mask.mode != 'L':
            pil_mask = pil_mask.convert('L')

        # Default prompt for seamless removal
        if not prompt:
            prompt = "natural background, seamless fill, realistic lighting"

        # Perform inpainting
        result = self.model(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        ).images[0]

        # Convert to numpy array
        if isinstance(result, Image.Image):
            result = np.array(result)

        return result
