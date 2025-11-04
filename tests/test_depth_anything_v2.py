"""Tests for DepthAnythingV2 model."""

import pytest
from fusion_ai.models import DepthAnythingV2


def test_depth_model_initialization():
    """Test model initialization."""
    model = DepthAnythingV2(model_size="small", device="cpu")
    assert model.model_size == "small"
    assert model.device == "cpu"
    assert model.model_name == "depth_anything_v2_small"


def test_model_sizes():
    """Test different model sizes."""
    for size in ["small", "base", "large"]:
        model = DepthAnythingV2(model_size=size, device="cpu")
        assert model.model_size == size


# Note: Actual inference tests would require downloading models
# and having test images available
