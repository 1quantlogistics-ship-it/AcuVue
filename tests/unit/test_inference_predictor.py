"""
Unit tests for inference predictor.

Tests the GlaucomaClassifier and GlaucomaPredictor classes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os
import json

from src.inference.predictor import (
    GlaucomaClassifier,
    GlaucomaPredictor,
    PredictionResult,
)
from src.inference.config import InferenceConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB fundus-like image."""
    # Create a 512x512 RGB image with fundus-like colors
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    # Dark red/orange background (like fundus)
    arr[:, :, 0] = 180  # Red channel
    arr[:, :, 1] = 80   # Green channel
    arr[:, :, 2] = 50   # Blue channel

    # Add a lighter circular region (like optic disc)
    y, x = np.ogrid[:512, :512]
    center_x, center_y = 256, 256
    r = 50
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2
    arr[mask, 0] = 255
    arr[mask, 1] = 200
    arr[mask, 2] = 150

    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def temp_image_path(sample_rgb_image) -> str:
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_rgb_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_checkpoint_path() -> str:
    """Create a temporary checkpoint file with random weights."""
    model = GlaucomaClassifier(num_classes=2, dropout=0.3)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_checkpoint_with_dict() -> str:
    """Create a checkpoint file with nested dict format."""
    model = GlaucomaClassifier(num_classes=2, dropout=0.3)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 30,
        'optimizer_state_dict': {},
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def classifier() -> GlaucomaClassifier:
    """Create a GlaucomaClassifier instance."""
    return GlaucomaClassifier(num_classes=2, dropout=0.3)


@pytest.fixture
def predictor(temp_checkpoint_path) -> GlaucomaPredictor:
    """Create a GlaucomaPredictor instance."""
    return GlaucomaPredictor.from_checkpoint(
        temp_checkpoint_path,
        device='cpu'
    )


# ============================================================================
# PredictionResult Tests
# ============================================================================

class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_creation(self):
        """Test creating a PredictionResult."""
        result = PredictionResult(
            prediction='glaucoma',
            confidence=0.95,
            probabilities={'normal': 0.05, 'glaucoma': 0.95},
            image_path='/path/to/image.png'
        )

        assert result.prediction == 'glaucoma'
        assert result.confidence == 0.95
        assert result.probabilities['glaucoma'] == 0.95

    def test_repr(self):
        """Test string representation."""
        result = PredictionResult(
            prediction='normal',
            confidence=0.8,
            probabilities={'normal': 0.8, 'glaucoma': 0.2}
        )

        repr_str = repr(result)
        assert 'normal' in repr_str
        assert '0.800' in repr_str

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = PredictionResult(
            prediction='glaucoma',
            confidence=0.9,
            probabilities={'normal': 0.1, 'glaucoma': 0.9},
            image_path='/test.png'
        )

        d = result.to_dict()
        assert d['prediction'] == 'glaucoma'
        assert d['confidence'] == 0.9
        assert d['image_path'] == '/test.png'
        assert 'probabilities' in d


# ============================================================================
# GlaucomaClassifier Tests
# ============================================================================

class TestGlaucomaClassifier:
    """Tests for GlaucomaClassifier model."""

    def test_initialization(self, classifier):
        """Test model initialization."""
        assert isinstance(classifier, nn.Module)
        assert classifier.num_features == 1280  # EfficientNet-B0

    def test_forward_pass(self, classifier):
        """Test forward pass with random input."""
        classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = classifier(x)

        assert output.shape == (1, 2)

    def test_batch_forward(self, classifier):
        """Test forward pass with batch input."""
        classifier.eval()
        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = classifier(x)

        assert output.shape == (4, 2)

    def test_output_logits(self, classifier):
        """Test that output is logits (not probabilities)."""
        classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = classifier(x)

        # Logits can be any real number
        # Probabilities would sum to 1 after softmax
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)

    def test_different_input_sizes(self, classifier):
        """Test that model handles different input sizes."""
        classifier.eval()

        for size in [224, 256, 299, 384]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = classifier(x)
            assert output.shape == (1, 2)

    def test_dropout_in_training(self, classifier):
        """Test that dropout is applied in training mode."""
        classifier.train()
        x = torch.randn(1, 3, 224, 224)

        # Multiple forward passes should give different results
        outputs = []
        for _ in range(5):
            output = classifier(x)
            outputs.append(output.clone())

        # Not all outputs should be identical
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same

    def test_no_dropout_in_eval(self, classifier):
        """Test that dropout is disabled in eval mode."""
        classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = [classifier(x).clone() for _ in range(3)]

        # All outputs should be identical in eval mode
        for out in outputs[1:]:
            assert torch.allclose(outputs[0], out)


# ============================================================================
# GlaucomaPredictor Tests
# ============================================================================

class TestGlaucomaPredictorLoading:
    """Tests for GlaucomaPredictor loading functionality."""

    def test_from_checkpoint(self, temp_checkpoint_path):
        """Test loading from checkpoint file."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_checkpoint_path,
            device='cpu'
        )

        assert isinstance(predictor, GlaucomaPredictor)
        assert predictor.device == 'cpu'

    def test_from_checkpoint_dict_format(self, temp_checkpoint_with_dict):
        """Test loading from checkpoint with nested dict format."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_checkpoint_with_dict,
            device='cpu'
        )

        assert isinstance(predictor, GlaucomaPredictor)

    def test_from_checkpoint_auto_device(self, temp_checkpoint_path):
        """Test auto device selection."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_checkpoint_path,
            device=None  # Auto-select
        )

        # Should be on CPU if CUDA not available
        assert predictor.device in ['cpu', 'cuda']

    def test_from_checkpoint_not_found(self):
        """Test error when checkpoint not found."""
        with pytest.raises(FileNotFoundError):
            GlaucomaPredictor.from_checkpoint('/nonexistent/model.pt')

    def test_model_is_in_eval_mode(self, predictor):
        """Test that loaded model is in eval mode."""
        assert not predictor.model.training

    def test_class_names_default(self, predictor):
        """Test default class names."""
        assert predictor.class_names == ['normal', 'glaucoma']


class TestGlaucomaPredictorInference:
    """Tests for GlaucomaPredictor inference functionality."""

    def test_predict_from_pil_image(self, predictor, sample_rgb_image):
        """Test prediction from PIL Image."""
        result = predictor.predict(sample_rgb_image)

        assert isinstance(result, PredictionResult)
        assert result.prediction in ['normal', 'glaucoma']
        assert 0 <= result.confidence <= 1
        assert result.image_path is None

    def test_predict_from_path(self, predictor, temp_image_path):
        """Test prediction from file path."""
        result = predictor.predict(temp_image_path)

        assert isinstance(result, PredictionResult)
        assert result.image_path == temp_image_path

    def test_predict_probabilities_sum_to_one(self, predictor, sample_rgb_image):
        """Test that probabilities sum to 1."""
        result = predictor.predict(sample_rgb_image)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-5

    def test_predict_confidence_matches_prediction(self, predictor, sample_rgb_image):
        """Test that confidence matches predicted class probability."""
        result = predictor.predict(sample_rgb_image)

        assert result.confidence == result.probabilities[result.prediction]

    def test_predict_batch(self, predictor, sample_rgb_image):
        """Test batch prediction."""
        images = [sample_rgb_image] * 3
        results = predictor.predict_batch(images)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)

    def test_predict_batch_from_paths(self, predictor, temp_image_path):
        """Test batch prediction from paths."""
        paths = [temp_image_path] * 3
        results = predictor.predict_batch(paths)

        assert len(results) == 3
        for result in results:
            assert result.image_path == temp_image_path

    def test_predict_batch_mixed(self, predictor, sample_rgb_image, temp_image_path):
        """Test batch prediction with mixed inputs."""
        images = [sample_rgb_image, temp_image_path, sample_rgb_image]
        results = predictor.predict_batch(images)

        assert len(results) == 3
        assert results[0].image_path is None
        assert results[1].image_path == temp_image_path
        assert results[2].image_path is None

    def test_predict_batch_large(self, predictor, sample_rgb_image):
        """Test batch prediction with many images."""
        images = [sample_rgb_image] * 10
        results = predictor.predict_batch(images, batch_size=4)

        assert len(results) == 10

    def test_repr(self, predictor):
        """Test string representation of predictor."""
        repr_str = repr(predictor)
        assert 'GlaucomaPredictor' in repr_str
        assert 'cpu' in repr_str


class TestGlaucomaPredictorConfig:
    """Tests for predictor configuration."""

    def test_custom_config(self, temp_checkpoint_path):
        """Test using custom configuration."""
        config = InferenceConfig(
            model_path=temp_checkpoint_path,
            device='cpu',
            input_size=(512, 512),
            confidence_threshold=0.7
        )

        predictor = GlaucomaPredictor.from_checkpoint(
            temp_checkpoint_path,
            device='cpu',
            config=config
        )

        assert predictor.config.input_size == (512, 512)
        assert predictor.config.confidence_threshold == 0.7


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_very_small_image(self, predictor):
        """Test prediction on very small image."""
        small_img = Image.new('RGB', (16, 16), color='red')
        result = predictor.predict(small_img)

        assert isinstance(result, PredictionResult)

    def test_predict_grayscale_image(self, predictor):
        """Test prediction on grayscale image."""
        gray_img = Image.new('L', (224, 224), color=128)
        result = predictor.predict(gray_img)

        assert isinstance(result, PredictionResult)

    def test_predict_rgba_image(self, predictor):
        """Test prediction on RGBA image."""
        rgba_img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
        result = predictor.predict(rgba_img)

        assert isinstance(result, PredictionResult)

    def test_predict_empty_batch(self, predictor):
        """Test prediction on empty batch."""
        results = predictor.predict_batch([])

        assert results == []
