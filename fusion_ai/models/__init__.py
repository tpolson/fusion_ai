"""AI model implementations."""

from fusion_ai.models.qwen_edit import QwenEdit
from fusion_ai.models.depth_anything_v2 import DepthAnythingV2
from fusion_ai.models.inpainting import LamaInpainting, StableDiffusionInpainting
from fusion_ai.models.outpainting import StableDiffusionOutpainting
from fusion_ai.models.upscale import RealESRGAN, StableDiffusionUpscale
from fusion_ai.models.style_transfer import StyleTransfer, InstantStyleTransfer
from fusion_ai.models.frame_extension import FrameExtension, FrameInterpolation

__all__ = [
    "QwenEdit",
    "DepthAnythingV2",
    "LamaInpainting",
    "StableDiffusionInpainting",
    "StableDiffusionOutpainting",
    "RealESRGAN",
    "StableDiffusionUpscale",
    "StyleTransfer",
    "InstantStyleTransfer",
    "FrameExtension",
    "FrameInterpolation",
]
