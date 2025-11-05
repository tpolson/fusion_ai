"""Outpainting model for extending images beyond their borders."""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR
from fusion_ai.utils.lora import LoRAManager


class StableDiffusionOutpainting(BaseModel):
    """Stable Diffusion outpainting for extending images beyond borders."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        lora_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize Stable Diffusion outpainting model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            lora_dir: Directory containing LoRA files
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        self.lora_dir = lora_dir
        self.lora_manager = LoRAManager(lora_dir)
        model_name = "stable_diffusion_outpainting"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the Stable Diffusion inpainting model for outpainting."""
        try:
            from diffusers import StableDiffusionInpaintPipeline

            print(f"Loading Stable Diffusion for outpainting...")

            model_id = "stabilityai/stable-diffusion-2-inpainting"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"SD outpainting loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for outpainting. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SD outpainting model: {e}")

    def create_outpaint_mask(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        position: str = "center"
    ) -> Image.Image:
        """Create a mask for outpainting.

        Args:
            original_size: (width, height) of original image
            target_size: (width, height) of target output
            position: Position of original image (center/top/bottom/left/right)

        Returns:
            Binary mask (white = generate, black = keep)
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size

        # Create full white mask
        mask = Image.new('L', target_size, 255)

        # Calculate position for original image
        if position == "center":
            x = (target_w - orig_w) // 2
            y = (target_h - orig_h) // 2
        elif position == "top":
            x = (target_w - orig_w) // 2
            y = 0
        elif position == "bottom":
            x = (target_w - orig_w) // 2
            y = target_h - orig_h
        elif position == "left":
            x = 0
            y = (target_h - orig_h) // 2
        elif position == "right":
            x = target_w - orig_w
            y = (target_h - orig_h) // 2
        else:
            x = (target_w - orig_w) // 2
            y = (target_h - orig_h) // 2

        # Black out the area where original image will be
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + orig_w, y + orig_h], fill=0)

        return mask

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        extend_pixels: int = 256,
        direction: str = "all",
        prompt: str = "",
        negative_prompt: str = "blurry, bad quality, distorted, watermark",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        lora_paths: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> np.ndarray:
        """Perform outpainting to extend image.

        Args:
            image: Input image
            extend_pixels: Number of pixels to extend
            direction: Direction to extend (all/horizontal/vertical/top/bottom/left/right)
            prompt: Text prompt for outpainting content
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            lora_paths: Optional LoRA file path(s) to apply
            lora_weights: Optional LoRA weight(s) (0-1)
            **kwargs: Additional inference parameters

        Returns:
            Outpainted image as numpy array
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, preserve_alpha=False)
        orig_w, orig_h = pil_image.size

        # Calculate target size based on direction
        if direction == "all":
            target_w = orig_w + 2 * extend_pixels
            target_h = orig_h + 2 * extend_pixels
            position = "center"
        elif direction == "horizontal":
            target_w = orig_w + 2 * extend_pixels
            target_h = orig_h
            position = "center"
        elif direction == "vertical":
            target_w = orig_w
            target_h = orig_h + 2 * extend_pixels
            position = "center"
        elif direction == "top":
            target_w = orig_w
            target_h = orig_h + extend_pixels
            position = "bottom"
        elif direction == "bottom":
            target_w = orig_w
            target_h = orig_h + extend_pixels
            position = "top"
        elif direction == "left":
            target_w = orig_w + extend_pixels
            target_h = orig_h
            position = "right"
        elif direction == "right":
            target_w = orig_w + extend_pixels
            target_h = orig_h
            position = "left"
        else:
            target_w = orig_w + 2 * extend_pixels
            target_h = orig_h + 2 * extend_pixels
            position = "center"

        # Create canvas and place original image
        canvas = Image.new('RGB', (target_w, target_h), (127, 127, 127))

        if position == "center":
            x = (target_w - orig_w) // 2
            y = (target_h - orig_h) // 2
        elif position == "top":
            x = (target_w - orig_w) // 2
            y = 0
        elif position == "bottom":
            x = (target_w - orig_w) // 2
            y = target_h - orig_h
        elif position == "left":
            x = 0
            y = (target_h - orig_h) // 2
        elif position == "right":
            x = target_w - orig_w
            y = (target_h - orig_h) // 2
        else:
            x = (target_w - orig_w) // 2
            y = (target_h - orig_h) // 2

        canvas.paste(pil_image, (x, y))

        # Create mask
        mask = self.create_outpaint_mask((orig_w, orig_h), (target_w, target_h), position)

        # Default prompt for seamless extension
        if not prompt:
            prompt = "seamless extension, natural continuation, same style and lighting"

        # Apply LoRAs if provided
        if lora_paths:
            if isinstance(lora_paths, str):
                lora_paths = [lora_paths]
            if lora_weights is None:
                lora_weights = [0.8] * len(lora_paths)
            elif isinstance(lora_weights, (int, float)):
                lora_weights = [lora_weights] * len(lora_paths)

            for lora_path, lora_weight in zip(lora_paths, lora_weights):
                self.model = self.lora_manager.load_lora(
                    self.model,
                    lora_path,
                    weight=lora_weight
                )

        # Perform outpainting using inpainting
        result = self.model(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        ).images[0]

        # Unload LoRAs for next inference
        if lora_paths:
            self.lora_manager.unload_loras(self.model)

        # Convert to numpy array
        return np.array(result)
