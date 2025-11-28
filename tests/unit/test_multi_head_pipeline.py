"""
Unit tests for multi-head inference pipeline.

Tests the MultiHeadPipeline class and PipelineResult for domain-routed inference.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch

from src.inference.pipeline import MultiHeadPipeline, PipelineResult
from src.inference.predictor import GlaucomaPredictor, PredictionResult
from src.routing import DomainRouter, RoutingResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB fundus-like image."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 0] = 180  # Red channel
    arr[:, :, 1] = 80   # Green channel
    arr[:, :, 2] = 50   # Blue channel
    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def temp_image_path(sample_rgb_image) -> str:
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_rgb_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_router():
    """Create a mock DomainRouter."""
    router = Mock(spec=DomainRouter)
    router.route.return_value = RoutingResult(
        domain='rimone',
        confidence=0.95,
        all_scores={'rimone': 0.95, 'refuge2': 0.03, 'g1020': 0.01, 'unknown': 0.01}
    )
    return router


@pytest.fixture
def mock_predictor():
    """Create a mock GlaucomaPredictor."""
    predictor = Mock(spec=GlaucomaPredictor)
    predictor.predict.return_value = PredictionResult(
        prediction='normal',
        confidence=0.85,
        probabilities={'normal': 0.85, 'glaucoma': 0.15}
    )
    return predictor


@pytest.fixture
def mock_predictor_glaucoma():
    """Create a mock GlaucomaPredictor that returns glaucoma."""
    predictor = Mock(spec=GlaucomaPredictor)
    predictor.predict.return_value = PredictionResult(
        prediction='glaucoma',
        confidence=0.92,
        probabilities={'normal': 0.08, 'glaucoma': 0.92}
    )
    return predictor


@pytest.fixture
def pipeline_with_mocks(mock_router, mock_predictor):
    """Create a MultiHeadPipeline with mocked components."""
    heads = {'glaucoma_rimone_v1': mock_predictor}
    domain_mapping = {
        'rimone': 'glaucoma_rimone_v1',
        'refuge2': 'glaucoma_rimone_v1',
        'g1020': 'glaucoma_rimone_v1',
        'unknown': 'glaucoma_rimone_v1',
    }
    return MultiHeadPipeline(
        router=mock_router,
        heads=heads,
        domain_head_mapping=domain_mapping
    )


@pytest.fixture
def pipeline_with_multiple_heads(mock_router, mock_predictor, mock_predictor_glaucoma):
    """Create a MultiHeadPipeline with multiple expert heads."""
    heads = {
        'head_normal': mock_predictor,
        'head_glaucoma': mock_predictor_glaucoma,
    }
    domain_mapping = {
        'rimone': 'head_normal',
        'refuge2': 'head_glaucoma',
        'g1020': 'head_normal',
        'unknown': 'head_normal',
    }
    return MultiHeadPipeline(
        router=mock_router,
        heads=heads,
        domain_head_mapping=domain_mapping
    )


# ============================================================================
# PipelineResult Tests
# ============================================================================

class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_creation(self):
        """Test creating a PipelineResult."""
        result = PipelineResult(
            prediction='glaucoma',
            confidence=0.92,
            probabilities={'normal': 0.08, 'glaucoma': 0.92},
            routed_domain='rimone',
            routing_confidence=0.95,
            head_used='glaucoma_rimone_v1',
            image_path='/path/to/image.png'
        )

        assert result.prediction == 'glaucoma'
        assert result.confidence == 0.92
        assert result.routed_domain == 'rimone'
        assert result.routing_confidence == 0.95
        assert result.head_used == 'glaucoma_rimone_v1'

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = PipelineResult(
            prediction='normal',
            confidence=0.85,
            probabilities={'normal': 0.85, 'glaucoma': 0.15},
            routed_domain='refuge2',
            routing_confidence=0.88,
            head_used='glaucoma_refuge_v1'
        )

        d = result.to_dict()
        assert d['prediction'] == 'normal'
        assert d['confidence'] == 0.85
        assert d['routed_domain'] == 'refuge2'
        assert d['routing_confidence'] == 0.88
        assert d['head_used'] == 'glaucoma_refuge_v1'
        assert 'probabilities' in d

    def test_image_path_optional(self):
        """Test that image_path is optional."""
        result = PipelineResult(
            prediction='normal',
            confidence=0.9,
            probabilities={'normal': 0.9, 'glaucoma': 0.1},
            routed_domain='g1020',
            routing_confidence=0.75,
            head_used='head_v1'
        )

        assert result.image_path is None


# ============================================================================
# MultiHeadPipeline Tests - Initialization
# ============================================================================

class TestMultiHeadPipelineInit:
    """Tests for MultiHeadPipeline initialization."""

    def test_init_with_components(self, mock_router, mock_predictor):
        """Test initialization with components."""
        heads = {'head1': mock_predictor}
        mapping = {'rimone': 'head1'}

        pipeline = MultiHeadPipeline(
            router=mock_router,
            heads=heads,
            domain_head_mapping=mapping
        )

        assert pipeline.router == mock_router
        assert pipeline.heads == heads
        assert pipeline.domain_head_mapping == mapping

    def test_get_loaded_heads(self, pipeline_with_mocks):
        """Test getting list of loaded heads."""
        heads = pipeline_with_mocks.get_loaded_heads()

        assert isinstance(heads, list)
        assert 'glaucoma_rimone_v1' in heads

    def test_repr(self, pipeline_with_mocks):
        """Test string representation."""
        repr_str = repr(pipeline_with_mocks)

        assert 'MultiHeadPipeline' in repr_str
        assert 'heads=' in repr_str
        assert 'domains=' in repr_str


# ============================================================================
# MultiHeadPipeline Tests - Prediction
# ============================================================================

class TestMultiHeadPipelinePredict:
    """Tests for MultiHeadPipeline prediction functionality."""

    def test_predict_returns_pipeline_result(self, pipeline_with_mocks, sample_rgb_image):
        """Test that predict returns PipelineResult."""
        result = pipeline_with_mocks.predict(sample_rgb_image)

        assert isinstance(result, PipelineResult)

    def test_predict_calls_router(self, pipeline_with_mocks, sample_rgb_image):
        """Test that predict calls the router."""
        pipeline_with_mocks.predict(sample_rgb_image)

        pipeline_with_mocks.router.route.assert_called_once()

    def test_predict_calls_head(self, pipeline_with_mocks, sample_rgb_image):
        """Test that predict calls the appropriate head."""
        pipeline_with_mocks.predict(sample_rgb_image)

        head = pipeline_with_mocks.heads['glaucoma_rimone_v1']
        head.predict.assert_called_once()

    def test_predict_returns_correct_domain(self, pipeline_with_mocks, sample_rgb_image):
        """Test that predict returns routing domain."""
        result = pipeline_with_mocks.predict(sample_rgb_image)

        assert result.routed_domain == 'rimone'
        assert result.routing_confidence == 0.95

    def test_predict_returns_head_prediction(self, pipeline_with_mocks, sample_rgb_image):
        """Test that predict returns head's prediction."""
        result = pipeline_with_mocks.predict(sample_rgb_image)

        assert result.prediction == 'normal'
        assert result.confidence == 0.85

    def test_predict_from_path(self, pipeline_with_mocks, temp_image_path):
        """Test prediction from file path."""
        result = pipeline_with_mocks.predict(temp_image_path)

        assert isinstance(result, PipelineResult)
        assert result.image_path == temp_image_path

    def test_predict_image_path_pil(self, pipeline_with_mocks, sample_rgb_image):
        """Test that image_path is None for PIL input."""
        result = pipeline_with_mocks.predict(sample_rgb_image)

        assert result.image_path is None

    def test_predict_routes_to_correct_head(self, pipeline_with_multiple_heads, sample_rgb_image):
        """Test that different domains route to different heads."""
        # Default routing to 'rimone' -> 'head_normal'
        result = pipeline_with_multiple_heads.predict(sample_rgb_image)

        assert result.prediction == 'normal'
        assert result.head_used == 'head_normal'

    def test_predict_unknown_domain_fallback(self, mock_predictor, sample_rgb_image):
        """Test fallback when domain mapping not found."""
        router = Mock(spec=DomainRouter)
        router.route.return_value = RoutingResult(
            domain='new_domain',  # Not in mapping
            confidence=0.5,
            all_scores={'new_domain': 0.5}
        )

        pipeline = MultiHeadPipeline(
            router=router,
            heads={'only_head': mock_predictor},
            domain_head_mapping={'rimone': 'only_head'}
        )

        result = pipeline.predict(sample_rgb_image)

        # Should fallback to first available head
        assert result.head_used == 'only_head'


# ============================================================================
# MultiHeadPipeline Tests - Ensemble
# ============================================================================

class TestMultiHeadPipelineEnsemble:
    """Tests for ensemble prediction functionality."""

    def test_predict_with_ensemble(self, pipeline_with_multiple_heads, sample_rgb_image):
        """Test ensemble prediction with multiple heads."""
        result = pipeline_with_multiple_heads.predict_with_ensemble(sample_rgb_image)

        assert isinstance(result, PipelineResult)
        assert 'ensemble(' in result.head_used

    def test_ensemble_averages_probabilities(self, mock_router, sample_rgb_image):
        """Test that ensemble averages probabilities correctly."""
        # Create predictors with known probabilities
        pred1 = Mock(spec=GlaucomaPredictor)
        pred1.predict.return_value = PredictionResult(
            prediction='normal',
            confidence=0.8,
            probabilities={'normal': 0.8, 'glaucoma': 0.2}
        )

        pred2 = Mock(spec=GlaucomaPredictor)
        pred2.predict.return_value = PredictionResult(
            prediction='glaucoma',
            confidence=0.6,
            probabilities={'normal': 0.4, 'glaucoma': 0.6}
        )

        pipeline = MultiHeadPipeline(
            router=mock_router,
            heads={'head1': pred1, 'head2': pred2},
            domain_head_mapping={'rimone': 'head1'}
        )

        result = pipeline.predict_with_ensemble(sample_rgb_image)

        # Average: normal = (0.8 + 0.4) / 2 = 0.6, glaucoma = (0.2 + 0.6) / 2 = 0.4
        assert abs(result.probabilities['normal'] - 0.6) < 1e-5
        assert abs(result.probabilities['glaucoma'] - 0.4) < 1e-5
        assert result.prediction == 'normal'

    def test_ensemble_specific_heads(self, pipeline_with_multiple_heads, sample_rgb_image):
        """Test ensemble with specific head subset."""
        result = pipeline_with_multiple_heads.predict_with_ensemble(
            sample_rgb_image,
            head_names=['head_normal']
        )

        assert 'head_normal' in result.head_used

    def test_ensemble_no_valid_heads_error(self, mock_router, sample_rgb_image):
        """Test error when no valid heads specified."""
        pipeline = MultiHeadPipeline(
            router=mock_router,
            heads={'head1': Mock()},
            domain_head_mapping={'rimone': 'head1'}
        )

        with pytest.raises(ValueError, match="No valid heads found"):
            pipeline.predict_with_ensemble(sample_rgb_image, head_names=['nonexistent'])


# ============================================================================
# MultiHeadPipeline Tests - Routing Info
# ============================================================================

class TestMultiHeadPipelineRoutingInfo:
    """Tests for getting routing information without prediction."""

    def test_get_routing_info(self, pipeline_with_mocks, sample_rgb_image):
        """Test getting routing info without prediction."""
        result = pipeline_with_mocks.get_routing_info(sample_rgb_image)

        assert isinstance(result, RoutingResult)
        assert result.domain == 'rimone'
        assert result.confidence == 0.95

    def test_get_routing_info_does_not_call_heads(self, pipeline_with_mocks, sample_rgb_image):
        """Test that get_routing_info doesn't run head inference."""
        pipeline_with_mocks.get_routing_info(sample_rgb_image)

        # Router should be called
        pipeline_with_mocks.router.route.assert_called_once()

        # But heads should NOT be called
        head = pipeline_with_mocks.heads['glaucoma_rimone_v1']
        head.predict.assert_not_called()


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_grayscale_image(self, pipeline_with_mocks):
        """Test prediction on grayscale image."""
        gray_img = Image.new('L', (224, 224), color=128)
        result = pipeline_with_mocks.predict(gray_img)

        assert isinstance(result, PipelineResult)

    def test_predict_rgba_image(self, pipeline_with_mocks):
        """Test prediction on RGBA image."""
        rgba_img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
        result = pipeline_with_mocks.predict(rgba_img)

        assert isinstance(result, PipelineResult)

    def test_predict_very_small_image(self, pipeline_with_mocks):
        """Test prediction on very small image."""
        small_img = Image.new('RGB', (16, 16), color='red')
        result = pipeline_with_mocks.predict(small_img)

        assert isinstance(result, PipelineResult)

    def test_predict_path_object(self, pipeline_with_mocks, temp_image_path):
        """Test prediction from Path object."""
        path = Path(temp_image_path)
        result = pipeline_with_mocks.predict(path)

        assert isinstance(result, PipelineResult)
        assert result.image_path == str(path)
