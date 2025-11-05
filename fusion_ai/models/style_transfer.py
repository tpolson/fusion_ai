"""Style transfer models for applying artistic styles to images."""

from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from fusion_ai.core.base_model import BaseModel
from fusion_ai.config import CACHE_DIR


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
