"""Tests for utility functions."""

import pytest
import numpy as np
from PIL import Image
from fusion_ai.utils.image import normalize_array, resize_image


def test_normalize_array():
    """Test array normalization."""
    array = np.array([[0, 50, 100], [150, 200, 255]])
    normalized = normalize_array(array, min_val=0, max_val=255)

    assert normalized.min() == 0
    assert normalized.max() == 255
    assert normalized.dtype == np.uint8


def test_resize_image():
    """Test image resizing."""
    # Create test image
    img = Image.new("RGB", (100, 100), color="red")

    # Resize
    resized = resize_image(img, (50, 50))

    assert resized.size == (50, 50)
    assert isinstance(resized, Image.Image)


def test_resize_image_numpy():
    """Test image resizing with numpy output."""
    img = Image.new("RGB", (100, 100), color="blue")

    # Resize with numpy output
    resized = resize_image(img, (50, 50), return_numpy=True)

    assert isinstance(resized, np.ndarray)
    assert resized.shape[:2] == (50, 50)
