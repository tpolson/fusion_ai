"""Style transfer models for applying artistic styles to images."""

from pathlib import Path
from typing import Optional, Union, List, Dict
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR
from fusion_ai.utils.lora import LoRAManager


class StyleTransfer(BaseModel):
    """Neural style transfer for applying artistic styles."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize style transfer model.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        model_name = "style_transfer"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load style transfer model."""
        try:
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            print(f"Loading style transfer model...")

            # Use VGG19 for feature extraction
            self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()

            # Freeze parameters
            for param in self.vgg.parameters():
                param.requires_grad_(False)

            # Normalization for VGG
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            # Feature layers for style and content
            self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
            self.content_layers = ['conv4_2']

            print(f"Style transfer model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load style transfer model: {e}")

    def get_features(self, image: torch.Tensor, layers: list) -> dict:
        """Extract features from specific layers.

        Args:
            image: Input tensor
            layers: List of layer names

        Returns:
            Dictionary of layer features
        """
        features = {}
        x = image

        # VGG19 layer names mapping
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }

        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layer_names and layer_names[name] in layers:
                features[layer_names[name]] = x

        return features

    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix for style representation.

        Args:
            tensor: Input feature tensor

        Returns:
            Gram matrix
        """
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    @torch.no_grad()
    def infer(
        self,
        content_image: Union[str, Path, Image.Image, np.ndarray],
        style_image: Union[str, Path, Image.Image, np.ndarray],
        style_weight: float = 1e6,
        content_weight: float = 1,
        num_steps: int = 300,
        **kwargs
    ) -> np.ndarray:
        """Apply style transfer from style image to content image.

        Args:
            content_image: Content image to stylize
            style_image: Style reference image
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            num_steps: Number of optimization steps
            **kwargs: Additional parameters

        Returns:
            Stylized image as numpy array
        """
        import torchvision.transforms as transforms

        # Preprocess images
        content_pil = self.preprocess_image(content_image, preserve_alpha=False)
        style_pil = self.preprocess_image(style_image, preserve_alpha=False)

        # Resize style to match content
        style_pil = style_pil.resize(content_pil.size, Image.LANCZOS)

        # Convert to tensors
        to_tensor = transforms.ToTensor()
        content_tensor = to_tensor(content_pil).unsqueeze(0).to(self.device)
        style_tensor = to_tensor(style_pil).unsqueeze(0).to(self.device)

        # Normalize
        content_tensor = self.normalize(content_tensor)
        style_tensor = self.normalize(style_tensor)

        # Extract features
        content_features = self.get_features(content_tensor, self.content_layers)
        style_features = self.get_features(style_tensor, self.style_layers)

        # Calculate style gram matrices
        style_grams = {layer: self.gram_matrix(style_features[layer])
                      for layer in style_features}

        # Initialize target as content image
        target = content_tensor.clone().requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([target], lr=0.003)

        # Style transfer optimization
        for step in range(num_steps):
            target_features = self.get_features(target, self.style_layers + self.content_layers)

            # Content loss
            content_loss = torch.mean(
                (target_features['conv4_2'] - content_features['conv4_2']) ** 2
            )

            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]

                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                b, c, h, w = target_feature.shape
                style_loss += layer_style_loss / (c * h * w)

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step}/{num_steps}, Loss: {total_loss.item():.4f}")

        # Convert back to image
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)

        result = target.squeeze(0).detach()
        result = result * std + mean
        result = torch.clamp(result, 0, 1)

        # Convert to numpy
        result_np = result.cpu().numpy().transpose(1, 2, 0)
        result_np = (result_np * 255).astype(np.uint8)

        return result_np


class InstantStyleTransfer(BaseModel):
    """Fast arbitrary style transfer using pre-trained model."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize instant style transfer.

        Args:
            device: Device to run inference on
            cache_dir: Directory to cache models
            **kwargs: Additional arguments
        """
        model_name = "instant_style_transfer"
        super().__init__(model_name, device, cache_dir, **kwargs)
        # Model will be loaded on first use
        self.model_loaded = False

    def load_model(self) -> None:
        """Load pre-trained style transfer model."""
        # Placeholder - can integrate models like:
        # - Arbitrary Style Transfer (magenta)
        # - Fast Neural Style
        # - AdaIN
        print("Instant style transfer initialized")
        self.model_loaded = True

    @torch.no_grad()
    def infer(
        self,
        content_image: Union[str, Path, Image.Image, np.ndarray],
        style_image: Union[str, Path, Image.Image, np.ndarray],
        alpha: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Fast style transfer.

        Args:
            content_image: Content image
            style_image: Style image
            alpha: Style strength (0-1)
            **kwargs: Additional parameters

        Returns:
            Stylized image
        """
        # For now, use basic style transfer
        # Can be replaced with faster models
        st = StyleTransfer(device=self.device, cache_dir=self.cache_dir)
        return st.infer(content_image, style_image, num_steps=100, **kwargs)


class StableDiffusionStyleTransfer(BaseModel):
    """Style transfer using Stable Diffusion with LoRA support."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        lora_dir: Optional[Path] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """Initialize SD style transfer.

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
        model_name = "sd_style_transfer"
        super().__init__(model_name, device, cache_dir, **kwargs)
        self.load_model()

    def load_model(self) -> None:
        """Load Stable Diffusion model for style transfer."""
        try:
            from diffusers import (
                StableDiffusionImg2ImgPipeline,
                DPMSolverMultistepScheduler
            )

            print(f"Loading Stable Diffusion for style transfer...")

            model_id = "stabilityai/stable-diffusion-2-1"
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir),
                safety_checker=None
            )

            # Use faster scheduler
            self.model.scheduler = DPMSolverMultistepScheduler.from_config(
                self.model.scheduler.config
            )

            self.model = self.model.to(self.device)

            # Enable memory efficient attention
            try:
                self.model.enable_attention_slicing()
                self.model.enable_vae_slicing()
            except:
                pass

            precision = "fp16" if self.use_fp16 else "fp32"
            print(f"SD style transfer loaded successfully on {self.device} ({precision})")

        except ImportError:
            raise ImportError(
                "diffusers is required for SD style transfer. "
                "Install it with: pip install diffusers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SD style transfer: {e}")

    @torch.no_grad()
    def infer(
        self,
        content_image: Union[str, Path, Image.Image, np.ndarray],
        style_prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        lora_paths: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Apply style using Stable Diffusion with optional LoRA.

        Args:
            content_image: Content image to stylize
            style_prompt: Text prompt describing the desired style
            strength: Strength of stylization (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            lora_paths: Optional LoRA file path(s) to apply
            lora_weights: Optional LoRA weight(s) (0-1)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters

        Returns:
            Stylized image as numpy array
        """
        # Preprocess content image
        content_pil = self.preprocess_image(content_image, preserve_alpha=False)

        # Ensure proper size (SD works best with multiples of 8)
        w, h = content_pil.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        content_pil = content_pil.resize((w, h), Image.LANCZOS)

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

        # Set seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate stylized image
        result = self.model(
            prompt=style_prompt,
            image=content_pil,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )

        # Unload LoRAs for next inference
        if lora_paths:
            self.lora_manager.unload_loras(self.model)

        # Convert to numpy
        result_image = result.images[0]
        result_np = np.array(result_image)

        return result_np

    def scan_available_loras(self) -> List[Dict[str, str]]:
        """Get list of available LoRA files.

        Returns:
            List of dicts with 'name' and 'path' keys
        """
        return self.lora_manager.scan_loras()
