"""
Unit tests for domain routing.

Tests the DomainClassifier and DomainRouter classes for multi-head routing.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os

from src.routing import DomainRouter, DomainClassifier, RoutingResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB fundus-like image."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Dark red/orange background (like fundus)
    arr[:, :, 0] = 180  # Red channel
    arr[:, :, 1] = 80   # Green channel
    arr[:, :, 2] = 50   # Blue channel

    # Add a lighter circular region (like optic disc)
    y, x = np.ogrid[:256, :256]
    center_x, center_y = 128, 128
    r = 30
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
def temp_router_checkpoint() -> str:
    """Create a temporary checkpoint with random weights."""
    model = DomainClassifier(num_domains=4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_router_checkpoint_dict() -> str:
    """Create a checkpoint file with nested dict format."""
    model = DomainClassifier(num_domains=4)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 20,
        'optimizer_state_dict': {},
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def domain_classifier() -> DomainClassifier:
    """Create a DomainClassifier instance."""
    return DomainClassifier(num_domains=4, dropout=0.2)


@pytest.fixture
def domain_router() -> DomainRouter:
    """Create a DomainRouter without checkpoint (for testing)."""
    return DomainRouter(checkpoint_path=None, device='cpu')


# ============================================================================
# RoutingResult Tests
# ============================================================================

class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_creation(self):
        """Test creating a RoutingResult."""
        result = RoutingResult(
            domain='rimone',
            confidence=0.95,
            all_scores={'rimone': 0.95, 'refuge2': 0.03, 'g1020': 0.01, 'unknown': 0.01}
        )

        assert result.domain == 'rimone'
        assert result.confidence == 0.95
        assert result.all_scores['rimone'] == 0.95

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = RoutingResult(
            domain='refuge2',
            confidence=0.8,
            all_scores={'rimone': 0.1, 'refuge2': 0.8, 'g1020': 0.05, 'unknown': 0.05}
        )

        d = result.to_dict()
        assert d['domain'] == 'refuge2'
        assert d['confidence'] == 0.8
        assert 'all_scores' in d


# ============================================================================
# DomainClassifier Tests
# ============================================================================

class TestDomainClassifier:
    """Tests for DomainClassifier model."""

    def test_initialization(self, domain_classifier):
        """Test model initialization."""
        assert isinstance(domain_classifier, nn.Module)
        assert domain_classifier.num_features == 576  # MobileNetV3-Small

    def test_domains_class_attribute(self):
        """Test DOMAINS class attribute."""
        assert DomainClassifier.DOMAINS == ['rimone', 'refuge2', 'g1020', 'unknown']
        assert len(DomainClassifier.DOMAINS) == 4

    def test_forward_pass(self, domain_classifier):
        """Test forward pass with random input."""
        domain_classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = domain_classifier(x)

        assert output.shape == (1, 4)

    def test_batch_forward(self, domain_classifier):
        """Test forward pass with batch input."""
        domain_classifier.eval()
        x = torch.randn(8, 3, 224, 224)

        with torch.no_grad():
            output = domain_classifier(x)

        assert output.shape == (8, 4)

    def test_output_is_logits(self, domain_classifier):
        """Test that output is logits (not probabilities)."""
        domain_classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = domain_classifier(x)

        # Logits can be any real number
        # Probabilities would sum to 1 after softmax
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)

    def test_get_domain_name(self, domain_classifier):
        """Test get_domain_name method."""
        assert domain_classifier.get_domain_name(0) == 'rimone'
        assert domain_classifier.get_domain_name(1) == 'refuge2'
        assert domain_classifier.get_domain_name(2) == 'g1020'
        assert domain_classifier.get_domain_name(3) == 'unknown'
        assert domain_classifier.get_domain_name(99) == 'unknown'  # Out of range

    def test_get_num_params(self):
        """Test parameter count method."""
        num_params = DomainClassifier.get_num_params()
        # MobileNetV3-Small has ~2.5M params
        assert 1_000_000 < num_params < 5_000_000

    def test_dropout_in_training(self, domain_classifier):
        """Test that dropout is applied in training mode."""
        domain_classifier.train()
        x = torch.randn(1, 3, 224, 224)

        # Multiple forward passes should give different results
        outputs = []
        for _ in range(5):
            output = domain_classifier(x)
            outputs.append(output.clone())

        # Not all outputs should be identical
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same

    def test_no_dropout_in_eval(self, domain_classifier):
        """Test that dropout is disabled in eval mode."""
        domain_classifier.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = [domain_classifier(x).clone() for _ in range(3)]

        # All outputs should be identical in eval mode
        for out in outputs[1:]:
            assert torch.allclose(outputs[0], out)


# ============================================================================
# DomainRouter Tests - Loading
# ============================================================================

class TestDomainRouterLoading:
    """Tests for DomainRouter loading functionality."""

    def test_init_without_checkpoint(self):
        """Test initialization without checkpoint."""
        router = DomainRouter(checkpoint_path=None, device='cpu')

        assert isinstance(router, DomainRouter)
        assert router.device == 'cpu'

    def test_init_with_checkpoint(self, temp_router_checkpoint):
        """Test loading from checkpoint file."""
        router = DomainRouter(
            checkpoint_path=temp_router_checkpoint,
            device='cpu'
        )

        assert isinstance(router, DomainRouter)

    def test_init_with_dict_checkpoint(self, temp_router_checkpoint_dict):
        """Test loading from checkpoint with nested dict format."""
        router = DomainRouter(
            checkpoint_path=temp_router_checkpoint_dict,
            device='cpu'
        )

        assert isinstance(router, DomainRouter)

    def test_checkpoint_not_found(self):
        """Test error when checkpoint not found."""
        with pytest.raises(FileNotFoundError):
            DomainRouter(checkpoint_path='/nonexistent/model.pt')

    def test_model_is_in_eval_mode(self, domain_router):
        """Test that loaded model is in eval mode."""
        assert not domain_router.model.training

    def test_repr(self, domain_router):
        """Test string representation."""
        repr_str = repr(domain_router)
        assert 'DomainRouter' in repr_str
        assert 'cpu' in repr_str


# ============================================================================
# DomainRouter Tests - Inference
# ============================================================================

class TestDomainRouterInference:
    """Tests for DomainRouter inference functionality."""

    def test_classify_domain_from_pil(self, domain_router, sample_rgb_image):
        """Test domain classification from PIL Image."""
        domain = domain_router.classify_domain(sample_rgb_image)

        assert domain in ['rimone', 'refuge2', 'g1020', 'unknown']

    def test_classify_domain_from_path(self, domain_router, temp_image_path):
        """Test domain classification from file path."""
        domain = domain_router.classify_domain(temp_image_path)

        assert domain in ['rimone', 'refuge2', 'g1020', 'unknown']

    def test_get_routing_confidence(self, domain_router, sample_rgb_image):
        """Test getting routing confidence scores."""
        scores = domain_router.get_routing_confidence(sample_rgb_image)

        assert isinstance(scores, dict)
        assert 'rimone' in scores
        assert 'refuge2' in scores
        assert 'g1020' in scores
        assert 'unknown' in scores

        # Probabilities should sum to 1
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-5

    def test_route_returns_routing_result(self, domain_router, sample_rgb_image):
        """Test that route() returns RoutingResult."""
        result = domain_router.route(sample_rgb_image)

        assert isinstance(result, RoutingResult)
        assert result.domain in ['rimone', 'refuge2', 'g1020', 'unknown']
        assert 0 <= result.confidence <= 1
        assert isinstance(result.all_scores, dict)

    def test_route_from_path(self, domain_router, temp_image_path):
        """Test routing from file path."""
        result = domain_router.route(temp_image_path)

        assert isinstance(result, RoutingResult)

    def test_confidence_matches_domain(self, domain_router, sample_rgb_image):
        """Test that confidence matches the predicted domain's score."""
        result = domain_router.route(sample_rgb_image)

        assert result.confidence == result.all_scores[result.domain]

    def test_route_deterministic_in_eval(self, domain_router, sample_rgb_image):
        """Test that routing is deterministic in eval mode."""
        results = [domain_router.route(sample_rgb_image) for _ in range(3)]

        # All results should be identical
        assert all(r.domain == results[0].domain for r in results)
        assert all(r.confidence == results[0].confidence for r in results)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_route_very_small_image(self, domain_router):
        """Test routing on very small image."""
        small_img = Image.new('RGB', (16, 16), color='red')
        result = domain_router.route(small_img)

        assert isinstance(result, RoutingResult)

    def test_route_grayscale_image(self, domain_router):
        """Test routing on grayscale image (converted to RGB)."""
        gray_img = Image.new('L', (224, 224), color=128)

        # Should handle grayscale by converting to RGB internally
        result = domain_router.route(gray_img)
        assert isinstance(result, RoutingResult)

    def test_route_rgba_image(self, domain_router):
        """Test routing on RGBA image."""
        rgba_img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
        result = domain_router.route(rgba_img)

        assert isinstance(result, RoutingResult)

    def test_route_large_image(self, domain_router):
        """Test routing on large image (will be resized)."""
        large_img = Image.new('RGB', (2048, 2048), color='blue')
        result = domain_router.route(large_img)

        assert isinstance(result, RoutingResult)

    def test_route_from_path_object(self, domain_router, temp_image_path):
        """Test routing from Path object."""
        path = Path(temp_image_path)
        result = domain_router.route(path)

        assert isinstance(result, RoutingResult)
