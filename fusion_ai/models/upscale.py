"""AI upscaling models for image super-resolution."""

from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


class RealESRGAN(BaseModel):
    """Real-ESRGAN for high-quality AI upscaling."""

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize Real-ESRGAN upscaling model.

        Args:
            model_name: Model variant (RealESRGAN_x4plus, RealESRNet_x4plus, RealESRGAN_x4plus_anime_6B)
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        self.model_variant = model_name
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            print(f"Loading Real-ESRGAN ({self.model_variant})...")

            # Model configurations
            if self.model_variant == "RealESRGAN_x4plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            elif self.model_variant == "RealESRNet_x4plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            elif self.model_variant == "RealESRGAN_x4plus_anime_6B":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            elif self.model_variant == "RealESRGAN_x2plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            else:
                raise ValueError(f"Unknown model variant: {self.model_variant}")

            # Determine GPU ID
            gpu_id = None
            if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
                gpu_id = 0

            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=512,  # Use tiling to handle large images
                tile_pad=10,
                pre_pad=0,
                half=False,  # fp32 for best quality
                gpu_id=gpu_id
            )

            self.scale = netscale
            print(f"Real-ESRGAN loaded successfully on {self.device} (scale: {netscale}x)")

        except ImportError:
            raise ImportError(
                "realesrgan is required for upscaling. "
                "Install it with: pip install realesrgan basicsr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Real-ESRGAN model: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        outscale: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """Upscale image using Real-ESRGAN.

        Args:
            image: Input image
            outscale: Custom output scale (if None, uses model's default scale)
            **kwargs: Additional inference parameters

        Returns:
            Upscaled image as numpy array
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, preserve_alpha=False)

        # Convert to numpy array (BGR for cv2 compatibility)
        img_np = np.array(pil_image)

        # Convert RGB to BGR if needed
        if img_np.shape[2] == 3:
            img_np = img_np[:, :, ::-1]

        # Upscale
        try:
            output, _ = self.upsampler.enhance(img_np, outscale=outscale)

            # Convert BGR back to RGB
            if output.shape[2] == 3:
                output = output[:, :, ::-1]

            return output

        except Exception as e:
            raise RuntimeError(f"Upscaling failed: {e}")


class StableDiffusionUpscale(BaseModel):
    """Stable Diffusion x4 Upscaler for creative upscaling."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize Stable Diffusion upscaler.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            use_fp16: Use fp16 precision
            **kwargs: Additional arguments
        """
        self.use_fp16 = use_fp16
        model_name = "stable_diffusion_x4_upscaler"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load the Stable Diffusion x4 upscaler."""
        try:
            from diffusers import StableDiffusionUpscalePipeline

            print(f"Loading Stable Diffusion x4 Upscaler...")

            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionUpscalePipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir)
            )

            self.model = self.model.to(self.device)

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"SD Upscaler loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for SD upscaling. "
                "Install it with: pip install diffusers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SD upscaler: {e}")

    @torch.no_grad()
    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "high quality, detailed",
        negative_prompt: str = "blurry, low quality",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> np.ndarray:
        """Upscale image 4x using Stable Diffusion.

        Args:
            image: Input image (will be upscaled 4x)
            prompt: Prompt to guide upscaling
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            **kwargs: Additional inference parameters

        Returns:
            Upscaled image as numpy array (4x original size)
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, preserve_alpha=False)

        # Upscale
        result = self.model(
            prompt=prompt,
            image=pil_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        ).images[0]

        return np.array(result)
