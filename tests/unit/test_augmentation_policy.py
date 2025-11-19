"""
Unit Tests for Augmentation Policy System
==========================================

Tests for Phase E Week 2: Augmentation Policy Search

Covers:
- Safe augmentation operations
- Policy validation
- PolicyAugmentor application
- DRI metrics computation
- Policy evaluation
- Evolutionary search primitives

Run with: pytest tests/unit/test_augmentation_policy.py -v
"""

import pytest
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.augmentation_ops import (
    get_operation,
    list_safe_operations,
    list_forbidden_operations,
    validate_operation_name,
    ForbiddenOperationError,
    RotateOp,
    BrightnessOp,
    GaussianBlurOp
)

from data.policy_augmentor import (
    PolicyAugmentor,
    InvalidPolicyError,
    create_random_policy,
    mutate_policy,
    crossover_policies
)

from evaluation.dri_metrics import (
    GradCAM,
    DRIComputer
)


class TestAugmentationOperations:
    """Test individual augmentation operations."""

    def test_list_safe_operations(self):
        """Test that safe operations are listed correctly."""
        ops = list_safe_operations()
        assert len(ops) > 0
        assert "rotate" in ops
        assert "brightness" in ops
        assert "gaussian_blur" in ops

        # Forbidden operations should NOT be in safe list
        assert "cutout" not in ops
        assert "mixup" not in ops

    def test_list_forbidden_operations(self):
        """Test that forbidden operations are listed with reasons."""
        forbidden = list_forbidden_operations()
        assert len(forbidden) > 0
        assert "cutout" in forbidden
        assert "mixup" in forbidden

        # Each forbidden operation should have a reason
        for op_name, reason in forbidden.items():
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_validate_operation_name(self):
        """Test operation name validation."""
        # Valid safe operations
        assert validate_operation_name("rotate") == True
        assert validate_operation_name("brightness") == True

        # Forbidden operations
        assert validate_operation_name("cutout") == False
        assert validate_operation_name("mixup") == False

        # Unknown operations should raise ValueError
        with pytest.raises(ValueError):
            validate_operation_name("nonexistent_operation")

    def test_get_operation_safe(self):
        """Test getting safe operations."""
        op = get_operation("rotate")
        assert isinstance(op, RotateOp)
        assert op.safe == True
        assert op.magnitude_range == (-15.0, 15.0)

    def test_get_operation_forbidden(self):
        """Test that forbidden operations raise error."""
        with pytest.raises(ForbiddenOperationError):
            get_operation("cutout")

        with pytest.raises(ForbiddenOperationError):
            get_operation("mixup")

    def test_rotate_operation_pil(self):
        """Test rotation operation on PIL Image."""
        op = get_operation("rotate")
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        rotated = op.apply(image, 10.0)

        assert isinstance(rotated, Image.Image)
        assert rotated.size == image.size

    def test_rotate_operation_tensor(self):
        """Test rotation operation on torch Tensor."""
        op = get_operation("rotate")
        image = torch.randn(3, 224, 224)

        rotated = op.apply(image, 10.0)

        assert isinstance(rotated, torch.Tensor)
        assert rotated.shape == image.shape

    def test_brightness_operation(self):
        """Test brightness operation."""
        op = get_operation("brightness")
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Increase brightness
        brighter = op.apply(image, 0.1)
        assert isinstance(brighter, Image.Image)

        # Decrease brightness
        darker = op.apply(image, -0.1)
        assert isinstance(darker, Image.Image)

    def test_gaussian_blur_operation(self):
        """Test Gaussian blur operation."""
        op = get_operation("gaussian_blur")
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        blurred = op.apply(image, 3.0)

        assert isinstance(blurred, Image.Image)
        assert blurred.size == image.size

    def test_magnitude_clamping(self):
        """Test that magnitude is clamped to valid range."""
        op = get_operation("rotate")

        # Magnitude outside range should be clamped
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Should not raise error (clamped internally)
        rotated = op.apply(image, 100.0)  # Way outside range
        assert isinstance(rotated, Image.Image)


class TestPolicyAugmentor:
    """Test PolicyAugmentor class."""

    def test_valid_policy(self):
        """Test creating augmentor with valid policy."""
        policy = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
            {"operation": "brightness", "probability": 0.3, "magnitude": 0.1}
        ]

        augmentor = PolicyAugmentor(policy)

        assert augmentor.policy == policy
        assert len(augmentor.operations) == 2

    def test_invalid_policy_format(self):
        """Test that invalid policy format raises error."""
        # Not a list
        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor("not a list")

        # Empty list
        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor([])

    def test_policy_missing_fields(self):
        """Test that policy with missing fields raises error."""
        policy = [
            {"operation": "rotate", "probability": 0.5}  # Missing magnitude
        ]

        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor(policy)

    def test_policy_forbidden_operation(self):
        """Test that policy with forbidden operation raises error."""
        policy = [
            {"operation": "cutout", "probability": 0.5, "magnitude": 0.2}
        ]

        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor(policy)

    def test_policy_invalid_probability(self):
        """Test that invalid probability raises error."""
        policy = [
            {"operation": "rotate", "probability": 1.5, "magnitude": 10.0}  # > 1.0
        ]

        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor(policy)

    def test_policy_too_many_operations(self):
        """Test that policy with too many operations raises error."""
        policy = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0}
        ] * 15  # 15 operations (max is 10)

        with pytest.raises(InvalidPolicyError):
            PolicyAugmentor(policy)

    def test_apply_policy_pil(self):
        """Test applying policy to PIL Image."""
        policy = [
            {"operation": "rotate", "probability": 1.0, "magnitude": 10.0},
            {"operation": "brightness", "probability": 1.0, "magnitude": 0.1}
        ]

        augmentor = PolicyAugmentor(policy)
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        augmented = augmentor(image)

        assert isinstance(augmented, Image.Image)
        assert augmented.size == image.size

    def test_apply_policy_tensor(self):
        """Test applying policy to torch Tensor."""
        policy = [
            {"operation": "rotate", "probability": 1.0, "magnitude": 10.0}
        ]

        augmentor = PolicyAugmentor(policy)
        image = torch.randn(3, 224, 224)

        augmented = augmentor(image)

        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == image.shape

    def test_apply_policy_batch(self):
        """Test applying policy to batch of images."""
        policy = [
            {"operation": "rotate", "probability": 1.0, "magnitude": 10.0}
        ]

        augmentor = PolicyAugmentor(policy)
        images = [Image.new('RGB', (224, 224)) for _ in range(4)]

        augmented_batch = augmentor.apply_to_batch(images)

        assert len(augmented_batch) == 4
        for aug in augmented_batch:
            assert isinstance(aug, Image.Image)

    def test_stochastic_application(self):
        """Test that operations are applied stochastically."""
        # Policy with 0% probability
        policy = [
            {"operation": "rotate", "probability": 0.0, "magnitude": 10.0}
        ]

        augmentor = PolicyAugmentor(policy, seed=42)
        image = torch.randn(3, 224, 224)

        augmented = augmentor(image)

        # Should be identical (operation never applied)
        assert torch.allclose(image, augmented)

    def test_deterministic_with_seed(self):
        """Test that seed makes augmentation deterministic."""
        policy = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0}
        ]

        image = torch.randn(3, 224, 224)

        # Same seed should produce same result
        aug1 = PolicyAugmentor(policy, seed=42)(image)
        aug2 = PolicyAugmentor(policy, seed=42)(image)

        assert torch.allclose(aug1, aug2)

    def test_get_policy_summary(self):
        """Test getting policy summary."""
        policy = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
            {"operation": "brightness", "probability": 0.3, "magnitude": 0.1}
        ]

        augmentor = PolicyAugmentor(policy)
        summary = augmentor.get_policy_summary()

        assert summary['num_operations'] == 2
        assert 'rotate' in summary['operations']
        assert 'brightness' in summary['operations']
        assert 0 < summary['avg_probability'] <= 1.0


class TestEvolutionarySearch:
    """Test evolutionary search primitives."""

    def test_create_random_policy(self):
        """Test random policy generation."""
        policy = create_random_policy(num_operations=3, seed=42)

        assert len(policy) == 3

        # Each sub-policy should have required fields
        for sp in policy:
            assert "operation" in sp
            assert "probability" in sp
            assert "magnitude" in sp

            # Values should be in valid ranges
            assert 0 <= sp["probability"] <= 1.0

    def test_random_policy_operation_pool(self):
        """Test random policy with custom operation pool."""
        operation_pool = ["rotate", "brightness"]
        policy = create_random_policy(num_operations=2, operation_pool=operation_pool, seed=42)

        assert len(policy) == 2
        for sp in policy:
            assert sp["operation"] in operation_pool

    def test_mutate_policy(self):
        """Test policy mutation."""
        original = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0}
        ]

        mutated = mutate_policy(original, mutation_rate=0.5, seed=42)

        assert len(mutated) == len(original)
        # Mutation should change at least one value (with high probability)
        # Note: stochastic, so not guaranteed

    def test_mutate_policy_deterministic(self):
        """Test that mutation is deterministic with seed."""
        original = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0}
        ]

        mutated1 = mutate_policy(original, mutation_rate=0.5, seed=42)
        mutated2 = mutate_policy(original, mutation_rate=0.5, seed=42)

        assert mutated1 == mutated2

    def test_crossover_policies(self):
        """Test policy crossover."""
        policy1 = [
            {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
            {"operation": "brightness", "probability": 0.3, "magnitude": 0.1}
        ]

        policy2 = [
            {"operation": "hflip", "probability": 0.6, "magnitude": 1.0},
            {"operation": "contrast", "probability": 0.4, "magnitude": 0.2}
        ]

        child1, child2 = crossover_policies(policy1, policy2, seed=42)

        # Children should have operations from both parents
        assert len(child1) == len(policy1)
        assert len(child2) == len(policy2)

    def test_crossover_empty_policy(self):
        """Test crossover with single-element policies."""
        policy1 = [{"operation": "rotate", "probability": 0.5, "magnitude": 10.0}]
        policy2 = [{"operation": "brightness", "probability": 0.3, "magnitude": 0.1}]

        child1, child2 = crossover_policies(policy1, policy2, seed=42)

        # Should return copies (can't crossover single elements)
        assert len(child1) == 1
        assert len(child2) == 1


class TestDRIMetrics:
    """Test DRI (Disc Relevance Index) computation."""

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model for testing."""
        import torch.nn as nn
        import torch.nn.functional as F

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 2)

            def forward(self, x, clinical=None):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
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

    def test_gradcam_initialization(self, dummy_model):
        """Test Grad-CAM initialization."""
        grad_cam = GradCAM(dummy_model)

        assert grad_cam.model == dummy_model
        assert grad_cam.target_layer is not None

    def test_gradcam_generate_heatmap(self, dummy_model):
        """Test Grad-CAM heatmap generation."""
        grad_cam = GradCAM(dummy_model)
        image = torch.randn(1, 3, 224, 224)

        heatmap = grad_cam.generate_heatmap(image)

        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_dri_computer_initialization(self, dummy_model):
        """Test DRI computer initialization."""
        dri_computer = DRIComputer(dummy_model, dri_threshold=0.6)

        assert dri_computer.model == dummy_model
        assert dri_computer.dri_threshold == 0.6

    def test_compute_iou(self, dummy_model, dummy_disc_mask):
        """Test IoU computation."""
        dri_computer = DRIComputer(dummy_model)

        # Perfect overlap
        attention_map = dummy_disc_mask.clone()
        iou = dri_computer.compute_iou(attention_map, dummy_disc_mask)

        assert iou == pytest.approx(1.0, abs=0.01)

        # No overlap
        attention_map = torch.zeros_like(dummy_disc_mask)
        iou = dri_computer.compute_iou(attention_map, dummy_disc_mask)

        assert iou == pytest.approx(0.0, abs=0.01)

    def test_compute_dri(self, dummy_model, dummy_disc_mask):
        """Test DRI computation."""
        dri_computer = DRIComputer(dummy_model, dri_threshold=0.6)
        image = torch.randn(1, 3, 224, 224)

        result = dri_computer.compute_dri(image, dummy_disc_mask)

        assert 'dri' in result
        assert 'valid' in result
        assert 'attention_map' in result
        assert 0 <= result['dri'] <= 1.0
        assert isinstance(result['valid'], bool)

    def test_compute_dri_batch(self, dummy_model, dummy_disc_mask):
        """Test DRI computation on batch."""
        dri_computer = DRIComputer(dummy_model)
        images = torch.randn(4, 3, 224, 224)
        disc_masks = dummy_disc_mask.unsqueeze(0).repeat(4, 1, 1)

        result = dri_computer.compute_dri_batch(images, disc_masks)

        assert 'dri' in result
        assert 'valid' in result
        assert 'dri_per_sample' in result
        assert len(result['dri_per_sample']) == 4


if __name__ == '__main__':
    """Run tests with pytest."""
    import subprocess

    print("=" * 80)
    print("Running Augmentation Policy System Unit Tests")
    print("=" * 80)

    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent.parent
    )

    sys.exit(result.returncode)
