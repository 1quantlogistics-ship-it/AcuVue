"""
Unit Tests for Architecture Grammar System
==========================================

Comprehensive test suite for fusion modules, backbones, and model factory.
All tests are CPU-compatible (no GPU required).

Part of ARC Phase E: Architecture Grammar System
Dev 2 implementation - Week 1

Test Coverage:
1. Fusion modules: Shape correctness, gradient flow
2. Backbones: Feature extraction, pretrained weights
3. Model factory: Build all combinations, validation
4. Integration: End-to-end forward/backward pass

Run with: pytest test_architectures.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion_modules import (
    FiLMLayer, CrossAttentionFusion, GatedFusion, LateFusion,
    create_fusion_module
)
from backbones import (
    EfficientNetBackbone, ConvNeXtBackbone,
    create_backbone, get_backbone_feature_dim
)
from model_factory import (
    MultiModalClassifier, build_model_from_spec,
    validate_architecture_spec, get_model_summary
)


# Test fixtures
@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def image_size():
    return 224


@pytest.fixture
def clinical_dim():
    return 4


@pytest.fixture
def dummy_cnn_features(batch_size):
    """Dummy CNN features [B, 512, 7, 7]"""
    return torch.randn(batch_size, 512, 7, 7)


@pytest.fixture
def dummy_clinical(batch_size, clinical_dim):
    """Dummy clinical indicators [B, K]"""
    return torch.randn(batch_size, clinical_dim)


@pytest.fixture
def dummy_images(batch_size, image_size):
    """Dummy input images [B, 3, H, W]"""
    return torch.randn(batch_size, 3, image_size, image_size)


# ============================================================================
# Fusion Module Tests
# ============================================================================

class TestFiLMLayer:
    """Tests for FiLM (Feature-wise Linear Modulation) fusion."""

    def test_output_shape(self, batch_size, dummy_cnn_features, dummy_clinical):
        """Test that output shape is correct."""
        fusion = FiLMLayer(
            cnn_channels=512,
            clinical_dim=4,
            output_dim=256
        )

        output = fusion(dummy_cnn_features, dummy_clinical)

        assert output.shape == (batch_size, 256), \
            f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_gradient_flow(self, dummy_cnn_features, dummy_clinical):
        """Test that gradients flow through FiLM layer."""
        fusion = FiLMLayer(cnn_channels=512, clinical_dim=4, output_dim=256)

        # Enable gradient tracking
        dummy_cnn_features.requires_grad = True
        dummy_clinical.requires_grad = True

        output = fusion(dummy_cnn_features, dummy_clinical)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert dummy_cnn_features.grad is not None, "No gradient for CNN features"
        assert dummy_clinical.grad is not None, "No gradient for clinical indicators"
        assert torch.any(dummy_cnn_features.grad != 0), "CNN gradients are all zero"
        assert torch.any(dummy_clinical.grad != 0), "Clinical gradients are all zero"

    def test_modulation_effect(self, dummy_cnn_features, dummy_clinical):
        """Test that clinical indicators actually modulate features."""
        fusion = FiLMLayer(cnn_channels=512, clinical_dim=4, output_dim=256)

        # Two different clinical vectors should produce different outputs
        clinical_1 = torch.randn(2, 4)
        clinical_2 = torch.randn(2, 4)

        output_1 = fusion(dummy_cnn_features, clinical_1)
        output_2 = fusion(dummy_cnn_features, clinical_2)

        # Outputs should be different
        assert not torch.allclose(output_1, output_2, atol=1e-5), \
            "Different clinical indicators produced identical outputs"


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion."""

    def test_output_shape(self, batch_size, dummy_cnn_features, dummy_clinical):
        """Test that output shape is correct."""
        fusion = CrossAttentionFusion(
            cnn_channels=512,
            clinical_dim=4,
            num_heads=4,
            output_dim=256
        )

        output = fusion(dummy_cnn_features, dummy_clinical)

        assert output.shape == (batch_size, 256), \
            f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_gradient_flow(self, dummy_cnn_features, dummy_clinical):
        """Test that gradients flow through attention mechanism."""
        fusion = CrossAttentionFusion(cnn_channels=512, clinical_dim=4, output_dim=256)

        dummy_cnn_features.requires_grad = True
        dummy_clinical.requires_grad = True

        output = fusion(dummy_cnn_features, dummy_clinical)
        loss = output.sum()
        loss.backward()

        assert dummy_cnn_features.grad is not None
        assert dummy_clinical.grad is not None
        assert torch.any(dummy_cnn_features.grad != 0)
        assert torch.any(dummy_clinical.grad != 0)

    def test_num_heads_divisibility(self, dummy_cnn_features, dummy_clinical):
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            fusion = CrossAttentionFusion(
                cnn_channels=512,
                clinical_dim=4,
                num_heads=5,  # 64 is not divisible by 5
                hidden_dim=64,
                output_dim=256
            )


class TestGatedFusion:
    """Tests for gated fusion."""

    def test_output_shape_sigmoid(self, batch_size, dummy_cnn_features, dummy_clinical):
        """Test output shape with sigmoid gating."""
        fusion = GatedFusion(
            cnn_channels=512,
            clinical_dim=4,
            output_dim=256,
            gate_activation='sigmoid'
        )

        output = fusion(dummy_cnn_features, dummy_clinical)

        assert output.shape == (batch_size, 256), \
            f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_output_shape_softmax(self, batch_size, dummy_cnn_features, dummy_clinical):
        """Test output shape with softmax gating."""
        fusion = GatedFusion(
            cnn_channels=512,
            clinical_dim=4,
            output_dim=256,
            gate_activation='softmax'
        )

        output = fusion(dummy_cnn_features, dummy_clinical)

        assert output.shape == (batch_size, 256), \
            f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_gradient_flow(self, dummy_cnn_features, dummy_clinical):
        """Test gradient flow through gating mechanism."""
        fusion = GatedFusion(cnn_channels=512, clinical_dim=4, output_dim=256)

        dummy_cnn_features.requires_grad = True
        dummy_clinical.requires_grad = True

        output = fusion(dummy_cnn_features, dummy_clinical)
        loss = output.sum()
        loss.backward()

        assert dummy_cnn_features.grad is not None
        assert dummy_clinical.grad is not None

    def test_invalid_gate_activation(self, dummy_cnn_features, dummy_clinical):
        """Test that invalid gate_activation raises error."""
        with pytest.raises(ValueError):
            fusion = GatedFusion(
                cnn_channels=512,
                clinical_dim=4,
                gate_activation='invalid'
            )


class TestLateFusion:
    """Tests for late fusion (baseline)."""

    def test_output_shape(self, batch_size, dummy_cnn_features, dummy_clinical):
        """Test that output shape is correct."""
        fusion = LateFusion(
            cnn_channels=512,
            clinical_dim=4,
            output_dim=256
        )

        output = fusion(dummy_cnn_features, dummy_clinical)

        assert output.shape == (batch_size, 256), \
            f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_gradient_flow(self, dummy_cnn_features, dummy_clinical):
        """Test gradient flow through MLP."""
        fusion = LateFusion(cnn_channels=512, clinical_dim=4, output_dim=256)

        dummy_cnn_features.requires_grad = True
        dummy_clinical.requires_grad = True

        output = fusion(dummy_cnn_features, dummy_clinical)
        loss = output.sum()
        loss.backward()

        assert dummy_cnn_features.grad is not None
        assert dummy_clinical.grad is not None


class TestFusionFactory:
    """Tests for fusion module factory function."""

    @pytest.mark.parametrize("fusion_type", ['film', 'cross_attention', 'gated', 'late'])
    def test_create_all_fusion_types(self, fusion_type, dummy_cnn_features, dummy_clinical):
        """Test that factory can create all fusion types."""
        fusion = create_fusion_module(
            fusion_type=fusion_type,
            cnn_channels=512,
            clinical_dim=4,
            output_dim=256
        )

        output = fusion(dummy_cnn_features, dummy_clinical)
        assert output.shape == (2, 256)

    def test_invalid_fusion_type(self):
        """Test that invalid fusion type raises error."""
        with pytest.raises(ValueError):
            create_fusion_module(
                fusion_type='invalid',
                cnn_channels=512,
                clinical_dim=4
            )


# ============================================================================
# Backbone Tests
# ============================================================================

class TestEfficientNetBackbone:
    """Tests for EfficientNet backbones."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ('b0', 1280),
        ('b3', 1536)
    ])
    def test_feature_dimensions(self, variant, expected_dim, dummy_images):
        """Test that backbones produce correct feature dimensions."""
        backbone = EfficientNetBackbone(
            variant=variant,
            pretrained=False,  # Don't download weights in tests
            feature_extraction=True
        )

        features = backbone(dummy_images)

        # Check feature dimension (channel dimension)
        assert features.shape[1] == expected_dim, \
            f"Expected {expected_dim} channels, got {features.shape[1]}"

    def test_freeze_backbone(self, dummy_images):
        """Test that freeze_backbone actually freezes parameters."""
        backbone = EfficientNetBackbone(
            variant='b0',
            pretrained=False,
            freeze_backbone=True
        )

        # Check that all parameters are frozen
        for param in backbone.backbone.parameters():
            assert not param.requires_grad, "Found unfrozen parameter in frozen backbone"

    def test_feature_extraction_mode(self, dummy_images):
        """Test feature extraction returns spatial features."""
        backbone = EfficientNetBackbone(
            variant='b0',
            pretrained=False,
            feature_extraction=True
        )

        features = backbone(dummy_images)

        # Should return 4D tensor [B, C, H, W]
        assert len(features.shape) == 4, \
            f"Expected 4D tensor, got shape {features.shape}"


class TestConvNeXtBackbone:
    """Tests for ConvNeXt backbones."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ('tiny', 768),
        ('small', 768)
    ])
    def test_feature_dimensions(self, variant, expected_dim, dummy_images):
        """Test that ConvNeXt backbones produce correct feature dimensions."""
        backbone = ConvNeXtBackbone(
            variant=variant,
            pretrained=False,
            feature_extraction=True
        )

        features = backbone(dummy_images)

        assert features.shape[1] == expected_dim, \
            f"Expected {expected_dim} channels, got {features.shape[1]}"

    def test_freeze_backbone(self, dummy_images):
        """Test parameter freezing."""
        backbone = ConvNeXtBackbone(
            variant='tiny',
            pretrained=False,
            freeze_backbone=True
        )

        for param in backbone.backbone.parameters():
            assert not param.requires_grad


class TestBackboneFactory:
    """Tests for backbone factory function."""

    @pytest.mark.parametrize("backbone_name,expected_dim", [
        ('efficientnet-b0', 1280),
        ('efficientnet-b3', 1536),
        ('convnext-tiny', 768),
        ('convnext-small', 768)
    ])
    def test_create_all_backbones(self, backbone_name, expected_dim, dummy_images):
        """Test that factory can create all backbone types."""
        backbone = create_backbone(
            backbone_name=backbone_name,
            pretrained=False,
            feature_extraction=True
        )

        features = backbone(dummy_images)
        assert features.shape[1] == expected_dim

    def test_get_feature_dim(self):
        """Test feature dimension lookup without instantiation."""
        assert get_backbone_feature_dim('efficientnet-b3') == 1536
        assert get_backbone_feature_dim('convnext-tiny') == 768

    def test_invalid_backbone_name(self):
        """Test that invalid backbone name raises error."""
        with pytest.raises(ValueError):
            create_backbone('invalid-backbone')


# ============================================================================
# Model Factory Tests
# ============================================================================

class TestModelFactory:
    """Tests for complete model building from specs."""

    def test_build_basic_model(self, dummy_images, dummy_clinical):
        """Test building a basic model from spec."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'head_config': {'dropout': 0.3},
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)

        # Test forward pass
        logits = model(dummy_images, dummy_clinical)
        assert logits.shape == (2, 2), f"Expected shape (2, 2), got {logits.shape}"

    @pytest.mark.parametrize("backbone", ['efficientnet-b3', 'convnext-tiny'])
    @pytest.mark.parametrize("fusion_type", ['film', 'cross_attention', 'gated', 'late'])
    def test_all_architecture_combinations(self, backbone, fusion_type, dummy_images, dummy_clinical):
        """Test all valid architecture combinations."""
        spec = {
            'backbone': backbone,
            'fusion_type': fusion_type,
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        # Validate spec
        assert validate_architecture_spec(spec)

        # Build model
        model = build_model_from_spec(spec, num_classes=2)

        # Test forward pass
        logits = model(dummy_images, dummy_clinical)
        assert logits.shape == (2, 2)

    def test_spec_validation_missing_fields(self):
        """Test that validation catches missing required fields."""
        incomplete_spec = {
            'backbone': 'efficientnet-b3',
            # Missing fusion_type and clinical_dim
        }

        with pytest.raises(ValueError):
            validate_architecture_spec(incomplete_spec)

    def test_spec_validation_invalid_backbone(self):
        """Test that validation catches invalid backbone."""
        invalid_spec = {
            'backbone': 'invalid-backbone',
            'fusion_type': 'film',
            'clinical_dim': 4
        }

        with pytest.raises(ValueError):
            validate_architecture_spec(invalid_spec)

    def test_spec_validation_invalid_fusion(self):
        """Test that validation catches invalid fusion type."""
        invalid_spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'invalid-fusion',
            'clinical_dim': 4
        }

        with pytest.raises(ValueError):
            validate_architecture_spec(invalid_spec)

    def test_get_model_summary(self, dummy_images, dummy_clinical):
        """Test model summary generation."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)
        summary = get_model_summary(model)

        # Check summary contains expected keys
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'backbone_parameters' in summary
        assert 'fusion_parameters' in summary
        assert 'classifier_parameters' in summary

        # Check that parameter counts make sense
        assert summary['total_parameters'] > 0
        assert summary['trainable_parameters'] <= summary['total_parameters']


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_forward_backward_pass(self, dummy_images, dummy_clinical):
        """Test complete forward and backward pass through model."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)

        # Forward pass
        logits = model(dummy_images, dummy_clinical)
        assert logits.shape == (2, 2)

        # Compute loss and backward pass
        target = torch.tensor([0, 1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)

        loss.backward()

        # Check that gradients exist for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_get_feature_embeddings(self, dummy_images, dummy_clinical):
        """Test extracting feature embeddings before classification."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'head_config': {'hidden_dim': 256},
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)

        embeddings = model.get_feature_embeddings(dummy_images, dummy_clinical)

        assert embeddings.shape == (2, 256), \
            f"Expected embeddings shape (2, 256), got {embeddings.shape}"

    def test_batch_processing(self):
        """Test that model handles different batch sizes correctly."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 224, 224)
            clinical = torch.randn(batch_size, 4)

            logits = model(images, clinical)
            assert logits.shape == (batch_size, 2)


# ============================================================================
# Performance Tests (CPU-only)
# ============================================================================

class TestPerformance:
    """Performance and memory tests (CPU-only)."""

    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)
        summary = get_model_summary(model)

        # EfficientNet-B3 should have ~10-15M parameters
        total_params = summary['total_parameters']
        assert 5_000_000 < total_params < 20_000_000, \
            f"Unexpected parameter count: {total_params:,}"

    def test_forward_pass_determinism(self, dummy_images, dummy_clinical):
        """Test that forward pass is deterministic with same inputs."""
        spec = {
            'backbone': 'efficientnet-b3',
            'fusion_type': 'film',
            'clinical_dim': 4,
            'backbone_config': {'pretrained': False}
        }

        model = build_model_from_spec(spec, num_classes=2)
        model.eval()  # Set to eval mode (disable dropout)

        with torch.no_grad():
            output1 = model(dummy_images, dummy_clinical)
            output2 = model(dummy_images, dummy_clinical)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Forward pass is not deterministic"


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v', '--tb=short'])
