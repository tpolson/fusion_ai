"""Model download utilities."""

from pathlib import Path
from typing import Optional
import os


def download_model(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """Download model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: Specific file to download (optional)
        cache_dir: Cache directory
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded model
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        if filename:
            # Download specific file
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir else None,
                force_download=force_download
            )
            return Path(model_path)
        else:
            # Download entire repo
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                force_download=force_download
            )
            return Path(model_path)

    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install it with: pip install huggingface-hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


def check_model_cached(
    repo_id: str,
    cache_dir: Optional[Path] = None
) -> bool:
    """Check if model is already cached.

    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Cache directory

    Returns:
        True if model is cached
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        # This is a simplified check
        # In practice, you'd check for specific model files
        return False  # Implement actual caching check if needed

    except ImportError:
        return False
