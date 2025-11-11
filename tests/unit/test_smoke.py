"""
Unit tests for Phase 01 smoke test.

Tests core functionality of data pipeline, preprocessing, and model.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import normalize_illumination, center_crop
from src.data.segmentation_dataset import SegmentationDataset, create_dummy_dataset
from src.models.unet_disc_cup import UNet, dice_loss


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_normalize_illumination_rgb(self):
        """Test illumination normalization on RGB images."""
        # Create dummy RGB image
        img = np.ones((256, 256, 3), dtype=np.uint8) * 127

        # Apply normalization
        result = normalize_illumination(img)

        # Check shape is preserved
        assert result.shape == img.shape, "Shape changed after normalization"

        # Check dtype is preserved
        assert result.dtype == img.dtype, "Data type changed after normalization"

        # Check it's not the same as input (CLAHE should change values)
        # Note: With constant input, green channel should be equalized
        assert not np.array_equal(result, img), "Normalization had no effect"

    def test_normalize_illumination_grayscale(self):
        """Test illumination normalization on grayscale images."""
        # Create dummy grayscale image
        img = np.ones((256, 256), dtype=np.uint8) * 127

        # Apply normalization
        result = normalize_illumination(img)

        # Check shape is preserved
        assert result.shape == img.shape, "Shape changed after normalization"

        # Check dtype is preserved
        assert result.dtype == img.dtype, "Data type changed after normalization"

    def test_center_crop_default_margin(self):
        """Test center crop with default 10% margin."""
        # Create dummy image
        h, w = 512, 512
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Apply crop
        result = center_crop(img)

        # Expected dimensions (10% margin on each side)
        expected_h = int(h * 0.8)  # Remove 10% from top and bottom
        expected_w = int(w * 0.8)  # Remove 10% from left and right

        assert result.shape == (expected_h, expected_w, 3), \
            f"Crop dimensions incorrect: got {result.shape}, expected ({expected_h}, {expected_w}, 3)"

    def test_center_crop_custom_margin(self):
        """Test center crop with custom margin."""
        # Create dummy image
        h, w = 512, 512
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Apply crop with 20% margin
        margin = 0.2
        result = center_crop(img, margin_ratio=margin)

        # Expected dimensions
        expected_h = int(h * (1 - 2 * margin))
        expected_w = int(w * (1 - 2 * margin))

        assert result.shape == (expected_h, expected_w, 3), \
            f"Crop dimensions incorrect with custom margin"


class TestDataset:
    """Test SegmentationDataset."""

    def test_create_dummy_dataset(self):
        """Test dummy dataset creation."""
        num_samples = 5
        image_size = 512

        images, masks = create_dummy_dataset(num_samples, image_size)

        # Check number of samples
        assert len(images) == num_samples, "Wrong number of images"
        assert len(masks) == num_samples, "Wrong number of masks"

        # Check image shapes
        for img in images:
            assert img.shape == (image_size, image_size, 3), "Image shape incorrect"
            assert img.dtype == np.uint8, "Image dtype incorrect"

        # Check mask shapes
        for mask in masks:
            assert mask.shape == (image_size, image_size), "Mask shape incorrect"
            assert mask.dtype == np.uint8, "Mask dtype incorrect"

    def test_dataset_length(self):
        """Test dataset __len__ method."""
        images, masks = create_dummy_dataset(num_samples=10)
        dataset = SegmentationDataset(images, masks, augment=False)

        assert len(dataset) == 10, "Dataset length incorrect"

    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        images, masks = create_dummy_dataset(num_samples=5, image_size=512)
        dataset = SegmentationDataset(images, masks, augment=False)

        # Get first item
        img_tensor, mask_tensor = dataset[0]

        # Check types
        assert isinstance(img_tensor, torch.Tensor), "Image is not a tensor"
        assert isinstance(mask_tensor, torch.Tensor), "Mask is not a tensor"

        # Check shapes
        assert img_tensor.shape == (3, 512, 512), f"Image tensor shape incorrect: {img_tensor.shape}"
        assert mask_tensor.shape == (1, 512, 512), f"Mask tensor shape incorrect: {mask_tensor.shape}"

        # Check dtypes
        assert img_tensor.dtype == torch.float32, "Image tensor dtype incorrect"
        assert mask_tensor.dtype == torch.float32, "Mask tensor dtype incorrect"

        # Check value ranges (normalized to [0, 1])
        assert img_tensor.min() >= 0 and img_tensor.max() <= 1, \
            f"Image values out of range: [{img_tensor.min()}, {img_tensor.max()}]"
        assert mask_tensor.min() >= 0 and mask_tensor.max() <= 1, \
            f"Mask values out of range: [{mask_tensor.min()}, {mask_tensor.max()}]"

    def test_dataset_augmentation(self):
        """Test that augmentation changes outputs."""
        images, masks = create_dummy_dataset(num_samples=1, image_size=512)

        # Create dataset with augmentation
        dataset = SegmentationDataset(images, masks, augment=True)

        # Get multiple samples of the same item
        # With random augmentation, they should be different
        samples = [dataset[0] for _ in range(5)]

        # Extract image tensors
        img_tensors = [s[0] for s in samples]

        # At least some should be different (not all identical)
        # Note: There's a small chance all are identical, but very unlikely
        all_same = all(torch.equal(img_tensors[0], t) for t in img_tensors[1:])
        assert not all_same, "Augmentation appears to have no effect"


class TestModel:
    """Test UNet model."""

    def test_model_instantiation(self):
        """Test model can be instantiated."""
        model = UNet()
        assert model is not None, "Model instantiation failed"

    def test_model_forward_pass(self):
        """Test forward pass with dummy input."""
        model = UNet()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 512, 512)

        # Forward pass
        output = model(dummy_input)

        # Check output shape
        expected_shape = (batch_size, 1, 512, 512)
        assert output.shape == expected_shape, \
            f"Output shape incorrect: {output.shape}, expected {expected_shape}"

        # Check output range (sigmoid should be 0-1)
        assert output.min() >= 0 and output.max() <= 1, \
            f"Output values out of range: [{output.min()}, {output.max()}]"

    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = UNet()
        param_count = model.count_parameters()

        # UNet should have a reasonable number of parameters
        # For this architecture, expect around 7-8 million parameters
        assert param_count > 1_000_000, "Too few parameters"
        assert param_count < 50_000_000, "Too many parameters"

    def test_model_on_cuda(self):
        """Test model can be moved to CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = UNet()
        model = model.cuda()

        # Check model is on CUDA
        assert next(model.parameters()).is_cuda, "Model not on CUDA"

        # Test forward pass on CUDA
        dummy_input = torch.randn(1, 3, 512, 512).cuda()
        output = model(dummy_input)

        assert output.is_cuda, "Output not on CUDA"

    def test_dice_loss_perfect_overlap(self):
        """Test dice loss with perfect overlap (should be ~0)."""
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)

        loss = dice_loss(pred, target)

        # Perfect overlap should give loss close to 0
        assert loss < 0.01, f"Dice loss for perfect overlap should be ~0, got {loss}"

    def test_dice_loss_no_overlap(self):
        """Test dice loss with no overlap (should be ~1)."""
        pred = torch.ones(2, 1, 64, 64)
        target = torch.zeros(2, 1, 64, 64)

        loss = dice_loss(pred, target)

        # No overlap should give loss close to 1
        assert loss > 0.9, f"Dice loss for no overlap should be ~1, got {loss}"

    def test_dice_loss_partial_overlap(self):
        """Test dice loss with partial overlap."""
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        loss = dice_loss(pred, target)

        # Partial overlap should give loss between 0 and 1
        assert 0 <= loss <= 1, f"Dice loss out of range: {loss}"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: data -> model -> loss."""
        # Create dummy dataset
        images, masks = create_dummy_dataset(num_samples=4, image_size=512)
        dataset = SegmentationDataset(images, masks, augment=False)

        # Create dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create model
        model = UNet()
        model.eval()

        # Get a batch
        img_batch, mask_batch = next(iter(loader))

        # Forward pass
        with torch.no_grad():
            predictions = model(img_batch)

        # Compute loss
        loss = dice_loss(predictions, mask_batch)

        # Verify everything worked
        assert predictions.shape == mask_batch.shape, "Prediction/target shape mismatch"
        assert 0 <= loss <= 1, f"Loss out of range: {loss}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
