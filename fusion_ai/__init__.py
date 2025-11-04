"""Fusion AI - AI models integration for Blackmagic Design Fusion."""

__version__ = "0.1.0"
__author__ = "Fusion AI Contributors"

from fusion_ai.models.qwen_edit import QwenEdit
from fusion_ai.models.depth_anything_v2 import DepthAnythingV2
from fusion_ai.core.base_model import BaseModel

__all__ = [
    "QwenEdit",
    "DepthAnythingV2",
    "BaseModel",
]
