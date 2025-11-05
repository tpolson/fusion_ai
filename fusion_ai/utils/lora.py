"""LoRA (Low-Rank Adaptation) utilities for Fusion AI models."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import json


class LoRAManager:
    """Manages LoRA loading and application for Stable Diffusion pipelines."""

    def __init__(self, lora_dir: Optional[Union[str, Path]] = None):
        """Initialize LoRA manager.

        Args:
            lora_dir: Directory containing LoRA files
        """
        self.lora_dir = Path(lora_dir) if lora_dir else None
        self.loaded_loras: Dict[str, Dict] = {}

    def scan_loras(self) -> List[Dict[str, str]]:
        """Scan for available LoRA files.

        Returns:
            List of dicts with 'name' and 'path' keys
        """
        if not self.lora_dir or not self.lora_dir.exists():
            return []

        loras = []
        # Look for .safetensors and .pt files
        for ext in ['*.safetensors', '*.pt', '*.ckpt']:
            for lora_path in self.lora_dir.glob(ext):
                loras.append({
                    'name': lora_path.stem,
                    'path': str(lora_path)
                })

        return loras

    def load_lora(
        self,
        pipeline,
        lora_path: Union[str, Path],
        adapter_name: Optional[str] = None,
        weight: float = 1.0
    ):
        """Load LoRA weights into a diffusers pipeline.

        Args:
            pipeline: Diffusers pipeline to load LoRA into
            lora_path: Path to LoRA file
            adapter_name: Name for the adapter (defaults to filename)
            weight: Weight/strength of the LoRA (0-1)

        Returns:
            Pipeline with LoRA loaded
        """
        try:
            from diffusers import StableDiffusionPipeline
            from safetensors.torch import load_file

            lora_path = Path(lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")

            if adapter_name is None:
                adapter_name = lora_path.stem

            # Load LoRA weights
            if lora_path.suffix == '.safetensors':
                state_dict = load_file(str(lora_path))
            else:
                state_dict = torch.load(str(lora_path), map_location='cpu')

            # Use diffusers' built-in LoRA loading if available
            if hasattr(pipeline, 'load_lora_weights'):
                # Modern diffusers API
                pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
                if hasattr(pipeline, 'fuse_lora'):
                    pipeline.fuse_lora(lora_scale=weight)
            else:
                # Manual LoRA application (fallback)
                self._apply_lora_weights(pipeline, state_dict, weight)

            self.loaded_loras[adapter_name] = {
                'path': str(lora_path),
                'weight': weight
            }

            print(f"Loaded LoRA: {adapter_name} (weight: {weight})")
            return pipeline

        except ImportError as e:
            print(f"Missing dependency for LoRA loading: {e}")
            print("Install with: pip install safetensors")
            return pipeline
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
            return pipeline

    def _apply_lora_weights(
        self,
        pipeline,
        state_dict: Dict[str, torch.Tensor],
        weight: float
    ):
        """Manually apply LoRA weights to pipeline (fallback method).

        Args:
            pipeline: Diffusers pipeline
            state_dict: LoRA state dictionary
            weight: LoRA weight/strength
        """
        # This is a simplified implementation
        # Full implementation would need to handle different LoRA formats
        # and properly merge weights into UNet and text encoder

        if not hasattr(pipeline, 'unet'):
            return

        # Apply to UNet
        for name, param in pipeline.unet.named_parameters():
            # Look for matching LoRA weights
            lora_up_key = f"{name}.lora_up.weight"
            lora_down_key = f"{name}.lora_down.weight"

            if lora_up_key in state_dict and lora_down_key in state_dict:
                lora_up = state_dict[lora_up_key]
                lora_down = state_dict[lora_down_key]

                # Calculate LoRA delta
                delta = (lora_up @ lora_down) * weight

                # Add to original weights
                with torch.no_grad():
                    param.add_(delta.to(param.device, dtype=param.dtype))

    def load_multiple_loras(
        self,
        pipeline,
        loras: List[Dict[str, Union[str, float]]]
    ):
        """Load multiple LoRAs with different weights.

        Args:
            pipeline: Diffusers pipeline
            loras: List of dicts with 'path' and 'weight' keys

        Returns:
            Pipeline with all LoRAs loaded
        """
        for lora_config in loras:
            lora_path = lora_config.get('path')
            weight = lora_config.get('weight', 1.0)
            adapter_name = lora_config.get('name')

            if lora_path:
                pipeline = self.load_lora(
                    pipeline,
                    lora_path,
                    adapter_name=adapter_name,
                    weight=weight
                )

        return pipeline

    def unload_loras(self, pipeline):
        """Unload all LoRAs from pipeline.

        Args:
            pipeline: Diffusers pipeline

        Returns:
            Pipeline with LoRAs unloaded
        """
        try:
            if hasattr(pipeline, 'unfuse_lora'):
                pipeline.unfuse_lora()
            if hasattr(pipeline, 'unload_lora_weights'):
                pipeline.unload_lora_weights()

            self.loaded_loras.clear()
            print("Unloaded all LoRAs")

        except Exception as e:
            print(f"Failed to unload LoRAs: {e}")

        return pipeline

    def get_loaded_loras(self) -> Dict[str, Dict]:
        """Get currently loaded LoRAs.

        Returns:
            Dict of loaded LoRAs with their configs
        """
        return self.loaded_loras.copy()


def create_lora_config(
    lora_paths: Union[str, List[str]],
    weights: Union[float, List[float]] = 1.0
) -> List[Dict[str, Union[str, float]]]:
    """Create LoRA configuration list.

    Args:
        lora_paths: Single path or list of paths to LoRA files
        weights: Single weight or list of weights (0-1)

    Returns:
        List of LoRA configurations
    """
    if isinstance(lora_paths, str):
        lora_paths = [lora_paths]

    if isinstance(weights, (int, float)):
        weights = [weights] * len(lora_paths)

    if len(weights) != len(lora_paths):
        raise ValueError("Number of weights must match number of LoRA paths")

    configs = []
    for path, weight in zip(lora_paths, weights):
        configs.append({
            'path': path,
            'weight': weight,
            'name': Path(path).stem
        })

    return configs


def apply_style_lora(
    pipeline,
    style_name: str,
    lora_dir: Optional[Union[str, Path]] = None,
    weight: float = 0.8
):
    """Apply a style LoRA to a pipeline.

    Args:
        pipeline: Diffusers pipeline
        style_name: Name of the style (will look for matching LoRA file)
        lora_dir: Directory containing LoRA files
        weight: LoRA weight (0-1)

    Returns:
        Pipeline with style LoRA applied
    """
    manager = LoRAManager(lora_dir)
    available_loras = manager.scan_loras()

    # Find matching LoRA
    matching_lora = None
    for lora in available_loras:
        if style_name.lower() in lora['name'].lower():
            matching_lora = lora
            break

    if matching_lora:
        return manager.load_lora(
            pipeline,
            matching_lora['path'],
            weight=weight
        )
    else:
        print(f"Style LoRA not found: {style_name}")
        return pipeline
