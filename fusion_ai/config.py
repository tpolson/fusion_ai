"""Configuration for Fusion AI package."""

import os
from pathlib import Path
from typing import Optional

# Base directories
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Model cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fusion_ai"
CACHE_DIR = Path(os.environ.get("FUSION_AI_CACHE_DIR", DEFAULT_CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
DEVICE = os.environ.get("FUSION_AI_DEVICE", DEFAULT_DEVICE)

# Model configurations
MODEL_CONFIGS = {
    "depth_anything_v2": {
        "small": {
            "repo_id": "depth-anything/Depth-Anything-V2-Small",
            "model_name": "depth_anything_v2_vits.pth",
            "encoder": "vits",
        },
        "base": {
            "repo_id": "depth-anything/Depth-Anything-V2-Base",
            "model_name": "depth_anything_v2_vitb.pth",
            "encoder": "vitb",
        },
        "large": {
            "repo_id": "depth-anything/Depth-Anything-V2-Large",
            "model_name": "depth_anything_v2_vitl.pth",
            "encoder": "vitl",
        },
    },
    "qwen_edit": {
        "default": {
            "repo_id": "Qwen/Qwen-VL-Chat",
            "model_name": "Qwen-VL-Chat",
        },
    },
}

# Fusion paths
FUSION_PATHS = {
    "windows": Path(os.environ.get("APPDATA", "")) / "Blackmagic Design" / "Fusion" / "Fuses",
    "darwin": Path.home() / "Library" / "Application Support" / "Blackmagic Design" / "Fusion" / "Fuses",
    "linux": Path.home() / ".fusion" / "BlackmagicDesign" / "Fusion" / "Fuses",
}

def get_fusion_fuse_path() -> Optional[Path]:
    """Get the Fusion Fuse installation path for the current platform."""
    import platform
    system = platform.system().lower()

    if system == "windows":
        return FUSION_PATHS["windows"]
    elif system == "darwin":
        return FUSION_PATHS["darwin"]
    elif system == "linux":
        return FUSION_PATHS["linux"]
    return None

# Inference settings
MAX_IMAGE_SIZE = 4096
DEFAULT_BATCH_SIZE = 1
USE_FP16 = True  # Use half precision for faster inference

# Logging
LOG_LEVEL = os.environ.get("FUSION_AI_LOG_LEVEL", "INFO")
