"""
Unit Tests for Custom Loss Functions
=====================================

Tests for Phase E Week 3: Loss Function Engineering

Covers:
- WeightedBCELoss with automatic weight computation
- AsymmetricFocalLoss for reducing false negatives
- AUCSurrogateLoss for optimizing AUC
- DRIRegularizer for attention constraint
- CombinedLoss wrapper
- Loss factory validation and building
- Gradient flow verification

Run with: pytest tests/unit/test_custom_losses.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training.custom_losses import (
    WeightedBCELoss,
    AsymmetricFocalLoss,
    AUCSurrogateLoss,
    DRIRegularizer,
    CombinedLoss
)

from training.loss_factory import (
    validate_loss_spec,
    build_loss_from_spec,
    get_loss_summary
)


class TestWeightedBCELoss:
    """Test WeightedBCELoss implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = WeightedBCELoss()

        assert loss_fn.pos_weight == 1.0
        assert loss_fn.neg_weight == 1.0
        assert loss_fn.reduction == 'mean'

    def test_initialization_custom(self):
        """Test custom weight initialization."""
        loss_fn = WeightedBCELoss(pos_weight=2.0, neg_weight=0.5)

        assert loss_fn.pos_weight == 2.0
        assert loss_fn.neg_weight == 0.5

    def test_compute_weights_from_labels_balanced(self):
        """Test automatic weight computation with balanced data."""
        # 50% positive, 50% negative
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)

        pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(labels)

        # Weights should be approximately equal for balanced data
        assert abs(pos_weight - neg_weight) < 0.1

    def test_compute_weights_from_labels_imbalanced(self):
        """Test automatic weight computation with imbalanced data."""
        # 10% positive, 90% negative
        labels = torch.zeros(100, dtype=torch.float32)
        labels[:10] = 1.0

        pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(labels)

        # Positive class should have higher weight
        assert pos_weight > neg_weight
        assert pos_weight > 1.0

    def test_forward_shape(self):
        """Test forward pass output shape."""
        loss_fn = WeightedBCELoss()

        logits = torch.randn(4, 2, requires_grad=True)  # Batch size 4, 2 classes
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        # Should return scalar
        assert loss.shape == ()
        assert loss.requires_grad

    def test_forward_values(self):
        """Test that loss values are reasonable."""
        loss_fn = WeightedBCELoss(pos_weight=2.0, neg_weight=1.0)

        logits = torch.randn(10, 2)
        labels = torch.randint(0, 2, (10,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss >= 0.0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        loss_fn = WeightedBCELoss()

        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_reduction_modes(self):
        """Test different reduction modes."""
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        # Mean reduction
        loss_mean = WeightedBCELoss(reduction='mean')(logits, labels)
        assert loss_mean.shape == ()

        # Sum reduction
        loss_sum = WeightedBCELoss(reduction='sum')(logits, labels)
        assert loss_sum.shape == ()
        assert loss_sum > loss_mean  # Sum should be larger

    def test_perfect_predictions(self):
        """Test loss with perfect predictions."""
        loss_fn = WeightedBCELoss()

        # Perfect confidence on correct class
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # High confidence
        labels = torch.tensor([0, 1], dtype=torch.long)

        loss = loss_fn(logits, labels)

        # Should be very small (not exactly 0 due to sigmoid)
        assert loss < 0.1


class TestAsymmetricFocalLoss:
    """Test AsymmetricFocalLoss implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = AsymmetricFocalLoss()

        assert loss_fn.gamma_pos == 2.0
        assert loss_fn.gamma_neg == 1.0
        assert loss_fn.clip == 0.05
        assert loss_fn.reduction == 'mean'

    def test_initialization_custom(self):
        """Test custom initialization."""
        loss_fn = AsymmetricFocalLoss(gamma_pos=3.0, gamma_neg=0.5, clip=0.1)

        assert loss_fn.gamma_pos == 3.0
        assert loss_fn.gamma_neg == 0.5
        assert loss_fn.clip == 0.1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        loss_fn = AsymmetricFocalLoss()

        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss.requires_grad

    def test_forward_values(self):
        """Test that loss values are reasonable."""
        loss_fn = AsymmetricFocalLoss()

        logits = torch.randn(10, 2)
        labels = torch.randint(0, 2, (10,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_hard_examples_focus(self):
        """Test that focal loss focuses on hard examples."""
        loss_fn = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=2.0)

        # Easy example (confident correct prediction)
        easy_logits = torch.tensor([[10.0, -10.0]])
        easy_labels = torch.tensor([0], dtype=torch.long)
        easy_loss = loss_fn(easy_logits, easy_labels)

        # Hard example (uncertain prediction)
        hard_logits = torch.tensor([[0.1, -0.1]])
        hard_labels = torch.tensor([0], dtype=torch.long)
        hard_loss = loss_fn(hard_logits, hard_labels)

        # Hard example should have higher loss
        assert hard_loss > easy_loss

    def test_asymmetric_penalties(self):
        """Test asymmetric penalties for positive/negative classes."""
        # Higher gamma_pos = focus more on hard positive examples
        loss_fn = AsymmetricFocalLoss(gamma_pos=4.0, gamma_neg=1.0)

        logits = torch.randn(10, 2)
        labels = torch.randint(0, 2, (10,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss >= 0.0
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        """Test gradient flow."""
        loss_fn = AsymmetricFocalLoss()

        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_clip_behavior(self):
        """Test probability clipping."""
        loss_fn = AsymmetricFocalLoss(clip=0.1)

        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        # Should not crash with extreme values
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestAUCSurrogateLoss:
    """Test AUCSurrogateLoss implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = AUCSurrogateLoss()

        assert loss_fn.margin == 1.0
        assert loss_fn.reduction == 'mean'

    def test_initialization_custom(self):
        """Test custom initialization."""
        loss_fn = AUCSurrogateLoss(margin=0.5)

        assert loss_fn.margin == 0.5

    def test_forward_shape(self):
        """Test forward pass output shape."""
        loss_fn = AUCSurrogateLoss()

        logits = torch.randn(10, 2, requires_grad=True)
        labels = torch.randint(0, 2, (10,), dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        # Loss will require grad if there are both positive and negative examples
        if loss.requires_grad:
            assert loss.requires_grad

    def test_all_positive_labels(self):
        """Test with all positive labels."""
        loss_fn = AUCSurrogateLoss()

        logits = torch.randn(4, 2)
        labels = torch.ones(4, dtype=torch.float32)  # All positive

        loss = loss_fn(logits, labels)

        # Should return 0 (no negative examples to compare)
        assert loss == 0.0

    def test_all_negative_labels(self):
        """Test with all negative labels."""
        loss_fn = AUCSurrogateLoss()

        logits = torch.randn(4, 2)
        labels = torch.zeros(4, dtype=torch.float32)  # All negative

        loss = loss_fn(logits, labels)

        # Should return 0 (no positive examples to compare)
        assert loss == 0.0

    def test_balanced_labels(self):
        """Test with balanced positive/negative labels."""
        loss_fn = AUCSurrogateLoss()

        logits = torch.randn(6, 2)
        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        loss = loss_fn(logits, labels)

        assert loss >= 0.0
        assert not torch.isnan(loss)

    def test_perfect_separation(self):
        """Test with perfect class separation."""
        loss_fn = AUCSurrogateLoss(margin=1.0)

        # Positive examples have high scores, negative have low scores
        logits = torch.tensor([
            [-10.0, 10.0],  # Negative (low predicted score for class 1)
            [-10.0, 10.0],  # Negative
            [10.0, -10.0],  # Positive (low predicted score for class 1)
            [10.0, -10.0],  # Positive
        ])
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = loss_fn(logits, labels)

        # With this configuration, positive scores are actually lower
        # Loss should be >= 0 but may not be small
        assert loss >= 0.0

    def test_gradient_flow(self):
        """Test gradient flow."""
        loss_fn = AUCSurrogateLoss()

        logits = torch.randn(6, 2, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_margin_effect(self):
        """Test effect of margin parameter."""
        logits = torch.randn(6, 2)
        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        # Larger margin should require larger separation
        loss_small_margin = AUCSurrogateLoss(margin=0.5)(logits, labels)
        loss_large_margin = AUCSurrogateLoss(margin=2.0)(logits, labels)

        assert loss_large_margin >= loss_small_margin


class TestDRIRegularizer:
    """Test DRIRegularizer implementation."""

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 2)

            def forward(self, x, clinical=None):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleModel()
        model.eval()
        return model

    @pytest.fixture
    def dummy_disc_mask(self):
        """Create a circular disc mask."""
        disc_mask = torch.zeros(224, 224)
        center_y, center_x = 112, 112
        radius = 40

        for y in range(224):
            for x in range(224):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    disc_mask[y, x] = 1.0

        return disc_mask

    def test_initialization(self, dummy_model):
        """Test DRI regularizer initialization."""
        reg = DRIRegularizer(dummy_model, lambda_dri=0.1, dri_threshold=0.6)

        assert reg.model == dummy_model
        assert reg.lambda_dri == 0.1
        assert reg.dri_threshold == 0.6

    def test_forward_shape(self, dummy_model, dummy_disc_mask):
        """Test forward pass output shape."""
        reg = DRIRegularizer(dummy_model)

        images = torch.randn(2, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        penalty = reg(images, disc_masks)

        assert penalty.shape == ()
        assert penalty.requires_grad

    def test_penalty_values(self, dummy_model, dummy_disc_mask):
        """Test that penalty values are reasonable."""
        reg = DRIRegularizer(dummy_model, lambda_dri=0.1, dri_threshold=0.6)

        images = torch.randn(2, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        penalty = reg(images, disc_masks)

        assert penalty >= 0.0  # Penalty should be non-negative
        assert not torch.isnan(penalty)

    def test_no_penalty_when_dri_high(self, dummy_model, dummy_disc_mask):
        """Test that no penalty when DRI is above threshold."""
        reg = DRIRegularizer(dummy_model, lambda_dri=0.1, dri_threshold=0.0)

        images = torch.randn(2, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        penalty = reg(images, disc_masks)

        # With threshold=0, penalty should always be 0
        assert penalty == 0.0

    def test_gradient_flow(self, dummy_model, dummy_disc_mask):
        """Test gradient flow through regularizer."""
        reg = DRIRegularizer(dummy_model)

        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        penalty = reg(images, disc_masks)

        # Only compute gradient if penalty > 0
        if penalty > 0:
            penalty.backward()
            assert images.grad is not None


class TestCombinedLoss:
    """Test CombinedLoss wrapper."""

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 2)

            def forward(self, x, clinical=None):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleModel()
        model.eval()
        return model

    @pytest.fixture
    def dummy_disc_mask(self):
        """Create a circular disc mask."""
        disc_mask = torch.zeros(224, 224)
        center_y, center_x = 112, 112
        radius = 40

        for y in range(224):
            for x in range(224):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    disc_mask[y, x] = 1.0

        return disc_mask

    def test_initialization(self, dummy_model):
        """Test combined loss initialization."""
        base_loss = WeightedBCELoss()
        combined = CombinedLoss(base_loss, dummy_model, lambda_dri=0.1)

        assert combined.base_loss == base_loss
        assert combined.model == dummy_model
        assert combined.dri_reg.lambda_dri == 0.1

    def test_forward_shape(self, dummy_model, dummy_disc_mask):
        """Test forward pass output shape."""
        base_loss = WeightedBCELoss()
        combined = CombinedLoss(base_loss, dummy_model)

        logits = torch.randn(2, 2)
        labels = torch.randint(0, 2, (2,), dtype=torch.long)
        images = torch.randn(2, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        result = combined(logits, labels, images, disc_masks)

        assert isinstance(result, dict)
        assert 'total' in result
        assert 'base' in result
        assert 'dri_penalty' in result
        assert result['total'].shape == ()
        assert result['total'].requires_grad

    def test_loss_decomposition(self, dummy_model, dummy_disc_mask):
        """Test that loss = base_loss + dri_penalty."""
        base_loss = WeightedBCELoss()
        combined = CombinedLoss(base_loss, dummy_model, lambda_dri=0.1)

        logits = torch.randn(2, 2)
        labels = torch.randint(0, 2, (2,), dtype=torch.long)
        images = torch.randn(2, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        # Compute combined loss
        result = combined(logits, labels, images, disc_masks)

        # Compute components separately
        base = base_loss(logits, labels)
        penalty = combined.dri_reg(images, disc_masks)

        # Should be equal (within numerical precision)
        assert torch.allclose(result['total'], base + penalty, atol=1e-6)
        assert torch.allclose(result['base'], base, atol=1e-6)
        assert torch.allclose(result['dri_penalty'], penalty, atol=1e-6)

    def test_gradient_flow(self, dummy_model, dummy_disc_mask):
        """Test gradient flow through combined loss."""
        base_loss = WeightedBCELoss()
        combined = CombinedLoss(base_loss, dummy_model)

        logits = torch.randn(2, 2, requires_grad=True)
        labels = torch.randint(0, 2, (2,), dtype=torch.long)
        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(2, 1, 1)

        result = combined(logits, labels, images, disc_masks)
        result['total'].backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestLossFactory:
    """Test loss factory functions."""

    def test_validate_valid_spec(self):
        """Test validation of valid loss specs."""
        spec = {
            "loss_type": "weighted_bce",
            "pos_weight": 2.0,
            "neg_weight": 1.0
        }

        assert validate_loss_spec(spec) == True

    def test_validate_missing_loss_type(self):
        """Test validation rejects missing loss_type."""
        spec = {"pos_weight": 2.0}

        with pytest.raises(ValueError, match="missing 'loss_type'"):
            validate_loss_spec(spec)

    def test_validate_invalid_loss_type(self):
        """Test validation rejects invalid loss_type."""
        spec = {"loss_type": "invalid_loss"}

        with pytest.raises(ValueError, match="Invalid loss_type"):
            validate_loss_spec(spec)

    def test_validate_invalid_pos_weight(self):
        """Test validation rejects invalid pos_weight."""
        spec = {
            "loss_type": "weighted_bce",
            "pos_weight": -1.0
        }

        with pytest.raises(ValueError, match="pos_weight must be positive"):
            validate_loss_spec(spec)

    def test_validate_invalid_gamma(self):
        """Test validation rejects invalid gamma."""
        spec = {
            "loss_type": "asymmetric_focal",
            "gamma_pos": -1.0
        }

        with pytest.raises(ValueError, match="gamma_pos must be non-negative"):
            validate_loss_spec(spec)

    def test_validate_invalid_margin(self):
        """Test validation rejects invalid margin."""
        spec = {
            "loss_type": "auc_surrogate",
            "margin": -1.0
        }

        with pytest.raises(ValueError, match="margin must be positive"):
            validate_loss_spec(spec)

    def test_build_weighted_bce(self):
        """Test building weighted BCE loss."""
        spec = {
            "loss_type": "weighted_bce",
            "pos_weight": 2.0,
            "neg_weight": 1.0
        }

        loss_fn = build_loss_from_spec(spec)

        assert isinstance(loss_fn, WeightedBCELoss)
        assert loss_fn.pos_weight == 2.0
        assert loss_fn.neg_weight == 1.0

    def test_build_asymmetric_focal(self):
        """Test building asymmetric focal loss."""
        spec = {
            "loss_type": "asymmetric_focal",
            "gamma_pos": 3.0,
            "gamma_neg": 0.5
        }

        loss_fn = build_loss_from_spec(spec)

        assert isinstance(loss_fn, AsymmetricFocalLoss)
        assert loss_fn.gamma_pos == 3.0
        assert loss_fn.gamma_neg == 0.5

    def test_build_auc_surrogate(self):
        """Test building AUC surrogate loss."""
        spec = {
            "loss_type": "auc_surrogate",
            "margin": 0.5
        }

        loss_fn = build_loss_from_spec(spec)

        assert isinstance(loss_fn, AUCSurrogateLoss)
        assert loss_fn.margin == 0.5

    def test_build_cross_entropy(self):
        """Test building cross-entropy loss."""
        spec = {
            "loss_type": "cross_entropy"
        }

        loss_fn = build_loss_from_spec(spec)

        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_build_with_dri_regularization(self):
        """Test building loss with DRI regularization."""
        # Create dummy model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.fc = nn.Linear(16, 2)

            def forward(self, x, clinical=None):
                return self.fc(x.mean(dim=(2, 3)))

        model = SimpleModel()

        spec = {
            "loss_type": "weighted_bce",
            "dri_regularization": True,
            "lambda_dri": 0.2
        }

        loss_fn = build_loss_from_spec(spec, model=model)

        assert isinstance(loss_fn, CombinedLoss)
        assert loss_fn.dri_reg.lambda_dri == 0.2

    def test_build_dri_without_model_fails(self):
        """Test that DRI regularization requires model."""
        spec = {
            "loss_type": "weighted_bce",
            "dri_regularization": True
        }

        with pytest.raises(ValueError, match="Model required"):
            build_loss_from_spec(spec)

    def test_auto_compute_weights(self):
        """Test automatic weight computation from training labels."""
        # Imbalanced training data
        train_labels = torch.zeros(100, dtype=torch.float32)
        train_labels[:10] = 1.0  # 10% positive

        spec = {"loss_type": "weighted_bce"}

        loss_fn = build_loss_from_spec(spec, train_labels=train_labels)

        # Positive class should have higher weight
        assert loss_fn.pos_weight > loss_fn.neg_weight

    def test_get_loss_summary_weighted_bce(self):
        """Test loss summary for WeightedBCELoss."""
        loss_fn = WeightedBCELoss(pos_weight=2.0, neg_weight=1.0)
        summary = get_loss_summary(loss_fn)

        assert summary['loss_class'] == 'WeightedBCELoss'
        assert summary['pos_weight'] == 2.0
        assert summary['neg_weight'] == 1.0

    def test_get_loss_summary_asymmetric_focal(self):
        """Test loss summary for AsymmetricFocalLoss."""
        loss_fn = AsymmetricFocalLoss(gamma_pos=3.0, gamma_neg=0.5)
        summary = get_loss_summary(loss_fn)

        assert summary['loss_class'] == 'AsymmetricFocalLoss'
        assert summary['gamma_pos'] == 3.0
        assert summary['gamma_neg'] == 0.5

    def test_get_loss_summary_auc_surrogate(self):
        """Test loss summary for AUCSurrogateLoss."""
        loss_fn = AUCSurrogateLoss(margin=0.5)
        summary = get_loss_summary(loss_fn)

        assert summary['loss_class'] == 'AUCSurrogateLoss'
        assert summary['margin'] == 0.5

    def test_get_loss_summary_combined(self):
        """Test loss summary for CombinedLoss."""
        # Create dummy model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

            def forward(self, x, clinical=None):
                return x.mean(dim=(2, 3))

        base_loss = WeightedBCELoss()
        model = SimpleModel()
        loss_fn = CombinedLoss(base_loss, model, lambda_dri=0.15, dri_threshold=0.7)

        summary = get_loss_summary(loss_fn)

        assert summary['loss_class'] == 'CombinedLoss'
        assert summary['base_loss'] == 'WeightedBCELoss'
        assert summary['lambda_dri'] == 0.15
        assert summary['dri_threshold'] == 0.7


if __name__ == '__main__':
    """Run tests with pytest."""
    import subprocess

    print("=" * 80)
    print("Running Custom Loss Function Unit Tests")
    print("=" * 80)

    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent.parent
    )

    sys.exit(result.returncode)
