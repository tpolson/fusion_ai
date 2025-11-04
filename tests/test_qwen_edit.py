"""Tests for QwenEdit model."""

import pytest
from fusion_ai.models import QwenEdit


def test_qwen_model_initialization():
    """Test model initialization."""
    model = QwenEdit(device="cpu")
    assert model.model_name == "qwen_vl_chat"
    assert model.device == "cpu"


# Note: Actual inference tests would require downloading models
# and having test images available
