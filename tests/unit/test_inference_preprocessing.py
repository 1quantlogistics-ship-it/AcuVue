"""
Unit tests for inference preprocessing.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

from src.inference.preprocessing import (
    get_inference_transforms,
    preprocess_image,
    load_and_preprocess,
    unnormalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:128, :, 0] = 255
    arr[128:, :, 2] = 255
    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def temp_image_path(sample_rgb_image) -> str:
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_rgb_image.save(f.name)
        yield f.name
    os.unlink(f.name)


class TestGetInferenceTransforms:
    """Tests for get_inference_transforms function."""

    def test_default_size(self, sample_rgb_image):
        """Test transforms with default input size (224)."""
        transforms = get_inference_transforms()
        tensor = transforms(sample_rgb_image)
        assert tensor.shape == (3, 224, 224)

    def test_custom_size(self, sample_rgb_image):
        """Test transforms with custom input size."""
        transforms = get_inference_transforms(input_size=512)
        tensor = transforms(sample_rgb_image)
        assert tensor.shape == (3, 512, 512)


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_basic_preprocessing(self, sample_rgb_image):
        """Test basic image preprocessing."""
        tensor = preprocess_image(sample_rgb_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_grayscale_conversion(self):
        """Test that grayscale images are converted to RGB."""
        gray_img = Image.new('L', (256, 256), color=128)
        tensor = preprocess_image(gray_img)
        assert tensor.shape == (1, 3, 224, 224)


class TestLoadAndPreprocess:
    """Tests for load_and_preprocess function."""

    def test_load_from_path(self, temp_image_path):
        """Test loading and preprocessing from file path."""
        tensor = load_and_preprocess(temp_image_path)
        assert tensor.shape == (1, 3, 224, 224)


class TestUnnormalize:
    """Tests for unnormalize function."""

    def test_3d_tensor(self):
        """Test unnormalization of 3D tensor (C, H, W)."""
        tensor = torch.zeros(3, 224, 224)
        unnormed = unnormalize(tensor)
        expected_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        assert torch.allclose(unnormed, expected_mean, atol=1e-5)


class TestConstants:
    """Tests for preprocessing constants."""

    def test_imagenet_mean_values(self):
        """Test ImageNet mean values are correct."""
        assert len(IMAGENET_MEAN) == 3
        assert IMAGENET_MEAN == [0.485, 0.456, 0.406]

    def test_imagenet_std_values(self):
        """Test ImageNet std values are correct."""
        assert len(IMAGENET_STD) == 3
        assert IMAGENET_STD == [0.229, 0.224, 0.225]
