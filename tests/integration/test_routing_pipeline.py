"""
Integration tests for the multi-head routing pipeline.

Tests end-to-end flow from image input through domain routing to prediction.
These tests verify that all components work together correctly.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os
import json

from src.routing import DomainRouter, DomainClassifier, RoutingResult
from src.inference.pipeline import MultiHeadPipeline, PipelineResult
from src.inference.predictor import GlaucomaPredictor, GlaucomaClassifier
from src.inference.head_registry import (
    HeadConfig,
    EXPERT_HEADS,
    DOMAIN_HEAD_MAPPING,
    get_head_for_domain,
    get_available_heads,
    get_available_domains,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_fundus_image() -> Image.Image:
    """Create a realistic sample fundus image."""
    # Create a fundus-like image with typical characteristics
    arr = np.zeros((512, 512, 3), dtype=np.uint8)

    # Dark red/orange background (like fundus)
    arr[:, :, 0] = 160  # Red
    arr[:, :, 1] = 60   # Green
    arr[:, :, 2] = 40   # Blue

    # Add a lighter circular region (like optic disc)
    y, x = np.ogrid[:512, :512]
    center_x, center_y = 280, 256
    r = 60
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2
    arr[mask, 0] = 255
    arr[mask, 1] = 220
    arr[mask, 2] = 180

    # Add blood vessels (rough simulation)
    for i in range(10):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(100, 200)
        x1 = int(280 + 60 * np.cos(angle))
        y1 = int(256 + 60 * np.sin(angle))
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        # Draw line
        rr = np.linspace(y1, y2, 50).astype(int)
        cc = np.linspace(x1, x2, 50).astype(int)
        valid = (rr >= 0) & (rr < 512) & (cc >= 0) & (cc < 512)
        arr[rr[valid], cc[valid], 0] = 100
        arr[rr[valid], cc[valid], 1] = 30
        arr[rr[valid], cc[valid], 2] = 30

    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def temp_fundus_path(sample_fundus_image) -> str:
    """Create a temporary fundus image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_fundus_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_router_checkpoint() -> str:
    """Create a temporary router checkpoint."""
    model = DomainClassifier(num_domains=4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_predictor_checkpoint() -> str:
    """Create a temporary predictor checkpoint."""
    model = GlaucomaClassifier(num_classes=2)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 30,
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_pipeline_config(temp_router_checkpoint, temp_predictor_checkpoint) -> str:
    """Create a temporary pipeline configuration file."""
    config = {
        'router': {
            'checkpoint': temp_router_checkpoint
        },
        'heads': {
            'test_head_v1': {
                'checkpoint': temp_predictor_checkpoint
            }
        },
        'domain_mapping': {
            'rimone': 'test_head_v1',
            'refuge2': 'test_head_v1',
            'g1020': 'test_head_v1',
            'unknown': 'test_head_v1',
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        yield f.name
    os.unlink(f.name)


# ============================================================================
# End-to-End Router Tests
# ============================================================================

class TestRouterEndToEnd:
    """End-to-end tests for domain router."""

    def test_router_without_checkpoint(self, sample_fundus_image):
        """Test router initialization and inference without trained weights."""
        router = DomainRouter(checkpoint_path=None, device='cpu')
        result = router.route(sample_fundus_image)

        assert isinstance(result, RoutingResult)
        assert result.domain in DomainClassifier.DOMAINS
        assert 0 <= result.confidence <= 1
        assert len(result.all_scores) == 4

    def test_router_with_checkpoint(self, temp_router_checkpoint, sample_fundus_image):
        """Test router with loaded checkpoint."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')
        result = router.route(sample_fundus_image)

        assert isinstance(result, RoutingResult)
        assert result.domain in DomainClassifier.DOMAINS

    def test_router_from_path(self, temp_router_checkpoint, temp_fundus_path):
        """Test router with file path input."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')
        result = router.route(temp_fundus_path)

        assert isinstance(result, RoutingResult)

    def test_router_consistency(self, temp_router_checkpoint, sample_fundus_image):
        """Test that router gives consistent results for same image."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')

        results = [router.route(sample_fundus_image) for _ in range(5)]

        # All results should be identical (deterministic in eval mode)
        domains = [r.domain for r in results]
        confidences = [r.confidence for r in results]

        assert len(set(domains)) == 1
        assert len(set(confidences)) == 1

    def test_router_all_confidence_scores(self, sample_fundus_image):
        """Test that all confidence scores are returned."""
        router = DomainRouter(checkpoint_path=None, device='cpu')
        result = router.route(sample_fundus_image)

        expected_domains = ['rimone', 'refuge2', 'g1020', 'unknown']
        for domain in expected_domains:
            assert domain in result.all_scores
            assert 0 <= result.all_scores[domain] <= 1

        # Probabilities should sum to 1
        total = sum(result.all_scores.values())
        assert abs(total - 1.0) < 1e-5


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestPipelineEndToEnd:
    """End-to-end tests for multi-head pipeline."""

    def test_pipeline_from_components(
        self,
        temp_router_checkpoint,
        temp_predictor_checkpoint,
        sample_fundus_image
    ):
        """Test pipeline built from individual components."""
        # Build router
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')

        # Build predictor
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        # Build pipeline
        pipeline = MultiHeadPipeline(
            router=router,
            heads={'glaucoma_v1': predictor},
            domain_head_mapping={
                'rimone': 'glaucoma_v1',
                'refuge2': 'glaucoma_v1',
                'g1020': 'glaucoma_v1',
                'unknown': 'glaucoma_v1',
            }
        )

        # Run prediction
        result = pipeline.predict(sample_fundus_image)

        assert isinstance(result, PipelineResult)
        assert result.prediction in ['normal', 'glaucoma']
        assert 0 <= result.confidence <= 1
        assert result.routed_domain in DomainClassifier.DOMAINS
        assert result.head_used == 'glaucoma_v1'

    @pytest.mark.skipif(
        not Path('configs/pipeline_v1.yaml').exists(),
        reason="Production config not available"
    )
    def test_pipeline_from_config(self, sample_fundus_image):
        """Test pipeline loaded from config file."""
        pipeline = MultiHeadPipeline.from_config(
            'configs/pipeline_v1.yaml',
            device='cpu'
        )

        result = pipeline.predict(sample_fundus_image)
        assert isinstance(result, PipelineResult)

    def test_pipeline_full_flow(
        self,
        temp_router_checkpoint,
        temp_predictor_checkpoint,
        sample_fundus_image
    ):
        """Test complete pipeline flow with all outputs."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        pipeline = MultiHeadPipeline(
            router=router,
            heads={'glaucoma_v1': predictor},
            domain_head_mapping={'rimone': 'glaucoma_v1', 'unknown': 'glaucoma_v1'}
        )

        # Get routing info first
        routing_info = pipeline.get_routing_info(sample_fundus_image)
        assert isinstance(routing_info, RoutingResult)

        # Then get full prediction
        result = pipeline.predict(sample_fundus_image)

        # Routing should be consistent
        assert result.routed_domain == routing_info.domain
        assert abs(result.routing_confidence - routing_info.confidence) < 1e-5

    def test_pipeline_ensemble_mode(
        self,
        temp_router_checkpoint,
        temp_predictor_checkpoint,
        sample_fundus_image
    ):
        """Test pipeline ensemble prediction mode."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        pipeline = MultiHeadPipeline(
            router=router,
            heads={'head1': predictor, 'head2': predictor},
            domain_head_mapping={'rimone': 'head1'}
        )

        result = pipeline.predict_with_ensemble(sample_fundus_image)

        assert isinstance(result, PipelineResult)
        assert 'ensemble' in result.head_used


# ============================================================================
# Head Registry Tests
# ============================================================================

class TestHeadRegistry:
    """Tests for head registry functionality."""

    def test_expert_heads_structure(self):
        """Test that EXPERT_HEADS has correct structure."""
        for name, config in EXPERT_HEADS.items():
            assert isinstance(config, HeadConfig)
            assert config.name == name
            assert isinstance(config.checkpoint_path, str)
            assert isinstance(config.domains, list)
            assert isinstance(config.architecture, str)
            assert isinstance(config.metrics, dict)

    def test_domain_head_mapping(self):
        """Test that all domains have mappings."""
        expected_domains = ['rimone', 'refuge2', 'g1020', 'unknown']
        for domain in expected_domains:
            assert domain in DOMAIN_HEAD_MAPPING

    def test_get_head_for_domain(self):
        """Test getting head config for domain."""
        config = get_head_for_domain('rimone')

        assert isinstance(config, HeadConfig)
        assert 'rimone' in config.domains or config.name in DOMAIN_HEAD_MAPPING.values()

    def test_get_head_for_unknown_domain(self):
        """Test fallback for unknown domain."""
        config = get_head_for_domain('completely_new_domain')

        # Should return fallback head
        assert isinstance(config, HeadConfig)

    def test_get_available_heads(self):
        """Test listing available heads."""
        heads = get_available_heads()

        assert isinstance(heads, list)
        assert len(heads) > 0
        assert all(isinstance(h, str) for h in heads)

    def test_get_available_domains(self):
        """Test listing available domains."""
        domains = get_available_domains()

        assert isinstance(domains, list)
        assert 'rimone' in domains
        assert 'refuge2' in domains
        assert 'g1020' in domains
        assert 'unknown' in domains


# ============================================================================
# Batch Processing Tests
# ============================================================================

class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_predictor_batch(self, temp_predictor_checkpoint, sample_fundus_image):
        """Test batch prediction with predictor."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        images = [sample_fundus_image] * 5
        results = predictor.predict_batch(images, batch_size=2)

        assert len(results) == 5
        for result in results:
            assert result.prediction in ['normal', 'glaucoma']


# ============================================================================
# Device Handling Tests
# ============================================================================

class TestDeviceHandling:
    """Tests for CPU/GPU device handling."""

    def test_router_cpu(self, temp_router_checkpoint, sample_fundus_image):
        """Test router on CPU."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')

        assert router.device == 'cpu'
        result = router.route(sample_fundus_image)
        assert isinstance(result, RoutingResult)

    def test_predictor_cpu(self, temp_predictor_checkpoint, sample_fundus_image):
        """Test predictor on CPU."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        assert predictor.device == 'cpu'
        result = predictor.predict(sample_fundus_image)
        assert result.prediction in ['normal', 'glaucoma']

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_cuda(self, temp_router_checkpoint, sample_fundus_image):
        """Test router on CUDA."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cuda')

        assert router.device == 'cuda'
        result = router.route(sample_fundus_image)
        assert isinstance(result, RoutingResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_predictor_cuda(self, temp_predictor_checkpoint, sample_fundus_image):
        """Test predictor on CUDA."""
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cuda'
        )

        assert predictor.device == 'cuda'
        result = predictor.predict(sample_fundus_image)
        assert result.prediction in ['normal', 'glaucoma']


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for result serialization."""

    def test_routing_result_to_dict(self, sample_fundus_image):
        """Test RoutingResult JSON serialization."""
        router = DomainRouter(checkpoint_path=None, device='cpu')
        result = router.route(sample_fundus_image)

        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        assert loaded['domain'] == result.domain
        assert loaded['confidence'] == result.confidence

    def test_pipeline_result_to_dict(
        self,
        temp_router_checkpoint,
        temp_predictor_checkpoint,
        sample_fundus_image
    ):
        """Test PipelineResult JSON serialization."""
        router = DomainRouter(checkpoint_path=temp_router_checkpoint, device='cpu')
        predictor = GlaucomaPredictor.from_checkpoint(
            temp_predictor_checkpoint,
            device='cpu'
        )

        pipeline = MultiHeadPipeline(
            router=router,
            heads={'head1': predictor},
            domain_head_mapping={'rimone': 'head1', 'unknown': 'head1'}
        )

        result = pipeline.predict(sample_fundus_image)
        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        assert loaded['prediction'] == result.prediction
        assert loaded['routed_domain'] == result.routed_domain
        assert loaded['head_used'] == result.head_used
