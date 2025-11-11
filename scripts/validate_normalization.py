#!/usr/bin/env python3
"""
ImageNet Normalization Validation Script - Phase 03e
======================================================

Validates that ImageNet normalization is correctly applied to fundus images.

Tests:
1. Load batch with use_imagenet_norm=False → verify [0,1] range
2. Load batch with use_imagenet_norm=True → verify ImageNet normalized range
3. Compare batch statistics to expected values
4. Validate normalization formula: (img/255 - mean) / std

Usage:
    python scripts/validate_normalization.py

Output:
    reports/imagenet_norm_validation.txt
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path
from data.fundus_dataset import FundusDataset

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_batch(data_root: str, use_imagenet_norm: bool, batch_size: int = 32):
    """Load a batch of images with specified normalization."""
    dataset = FundusDataset(
        data_root=data_root,
        split='train',
        task='classification',
        image_size=512,
        augment=False,  # No augmentation for validation
        use_imagenet_norm=use_imagenet_norm
    )

    # Load first batch_size samples
    batch_images = []
    batch_labels = []

    for i in range(min(batch_size, len(dataset))):
        image, label = dataset[i]
        batch_images.append(image)
        batch_labels.append(label)

    batch_images = torch.stack(batch_images)
    batch_labels = torch.stack(batch_labels)

    return batch_images, batch_labels


def compute_batch_stats(batch: torch.Tensor):
    """Compute mean and std per channel for a batch."""
    # batch shape: (N, C, H, W)
    mean_per_channel = batch.mean(dim=[0, 2, 3])  # Average over batch and spatial dims
    std_per_channel = batch.std(dim=[0, 2, 3])

    return {
        'mean': mean_per_channel.numpy(),
        'std': std_per_channel.numpy(),
        'min': batch.min().item(),
        'max': batch.max().item(),
        'shape': tuple(batch.shape)
    }


def validate_normalization():
    """Run validation tests."""
    data_root = "data/processed/combined_v2"
    batch_size = 32

    print("=" * 70)
    print("ImageNet Normalization Validation - Phase 03e")
    print("=" * 70)
    print()

    # Test 1: Load batch WITHOUT ImageNet normalization
    print("Test 1: Loading batch WITHOUT ImageNet normalization...")
    batch_no_norm, labels_no_norm = load_batch(data_root, use_imagenet_norm=False, batch_size=batch_size)
    stats_no_norm = compute_batch_stats(batch_no_norm)

    print(f"  Batch shape: {stats_no_norm['shape']}")
    print(f"  Value range: [{stats_no_norm['min']:.4f}, {stats_no_norm['max']:.4f}]")
    print(f"  Mean per channel (R,G,B): [{stats_no_norm['mean'][0]:.4f}, {stats_no_norm['mean'][1]:.4f}, {stats_no_norm['mean'][2]:.4f}]")
    print(f"  Std per channel (R,G,B):  [{stats_no_norm['std'][0]:.4f}, {stats_no_norm['std'][1]:.4f}, {stats_no_norm['std'][2]:.4f}]")

    # Validate: should be in [0, 1] range
    assert stats_no_norm['min'] >= 0.0, f"Minimum value {stats_no_norm['min']} is below 0.0"
    assert stats_no_norm['max'] <= 1.0, f"Maximum value {stats_no_norm['max']} is above 1.0"
    print("  ✓ Values are in [0, 1] range")
    print()

    # Test 2: Load batch WITH ImageNet normalization
    print("Test 2: Loading batch WITH ImageNet normalization...")
    batch_imagenet, labels_imagenet = load_batch(data_root, use_imagenet_norm=True, batch_size=batch_size)
    stats_imagenet = compute_batch_stats(batch_imagenet)

    print(f"  Batch shape: {stats_imagenet['shape']}")
    print(f"  Value range: [{stats_imagenet['min']:.4f}, {stats_imagenet['max']:.4f}]")
    print(f"  Mean per channel (R,G,B): [{stats_imagenet['mean'][0]:.4f}, {stats_imagenet['mean'][1]:.4f}, {stats_imagenet['mean'][2]:.4f}]")
    print(f"  Std per channel (R,G,B):  [{stats_imagenet['std'][0]:.4f}, {stats_imagenet['std'][1]:.4f}, {stats_imagenet['std'][2]:.4f}]")

    # Validate: range should extend beyond [0, 1]
    # After ImageNet normalization: (x - mean) / std can be negative
    assert stats_imagenet['min'] < 0.0, f"Minimum value {stats_imagenet['min']} should be negative after normalization"
    print("  ✓ Values extend beyond [0, 1] range (normalized)")
    print()

    # Test 3: Verify transformation formula
    print("Test 3: Verifying normalization formula...")
    print(f"  ImageNet mean: {IMAGENET_MEAN.numpy()}")
    print(f"  ImageNet std:  {IMAGENET_STD.numpy()}")

    # Manual normalization: (batch_no_norm - mean) / std
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)  # Broadcast to (1, C, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    batch_manual = (batch_no_norm - mean) / std

    # Compare with actual ImageNet-normalized batch
    diff = torch.abs(batch_imagenet - batch_manual).max().item()
    print(f"  Max difference: {diff:.6f}")

    # Should be very close (< 1e-5)
    assert diff < 1e-5, f"Manual normalization differs by {diff} (expected < 1e-5)"
    print("  ✓ Normalization formula is correct")
    print()

    # Test 4: Verify labels are unchanged
    print("Test 4: Verifying labels are unchanged...")
    assert torch.all(labels_no_norm == labels_imagenet), "Labels differ between normalized and non-normalized batches"
    print(f"  Labels match: {labels_no_norm[:5].tolist()}")
    print("  ✓ Labels are unchanged")
    print()

    # Summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print()
    print("✓ All tests passed!")
    print()
    print("Results:")
    print(f"  - WITHOUT normalization: values in [{stats_no_norm['min']:.4f}, {stats_no_norm['max']:.4f}]")
    print(f"  - WITH normalization: values in [{stats_imagenet['min']:.4f}, {stats_imagenet['max']:.4f}]")
    print(f"  - Normalization formula verified (max diff: {diff:.6f})")
    print(f"  - Batch size: {batch_size} samples")
    print()
    print("ImageNet normalization is correctly implemented and applied.")
    print("=" * 70)

    # Save report
    output_path = Path("reports/imagenet_norm_validation.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ImageNet Normalization Validation - Phase 03e\n")
        f.write("=" * 70 + "\n\n")

        f.write("Test Configuration:\n")
        f.write(f"  Dataset: {data_root}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  ImageNet mean: {IMAGENET_MEAN.numpy()}\n")
        f.write(f"  ImageNet std: {IMAGENET_STD.numpy()}\n\n")

        f.write("Test 1: WITHOUT ImageNet normalization\n")
        f.write(f"  Value range: [{stats_no_norm['min']:.4f}, {stats_no_norm['max']:.4f}]\n")
        f.write(f"  Mean (R,G,B): [{stats_no_norm['mean'][0]:.4f}, {stats_no_norm['mean'][1]:.4f}, {stats_no_norm['mean'][2]:.4f}]\n")
        f.write(f"  Std (R,G,B): [{stats_no_norm['std'][0]:.4f}, {stats_no_norm['std'][1]:.4f}, {stats_no_norm['std'][2]:.4f}]\n")
        f.write("  Result: ✓ PASS (values in [0, 1])\n\n")

        f.write("Test 2: WITH ImageNet normalization\n")
        f.write(f"  Value range: [{stats_imagenet['min']:.4f}, {stats_imagenet['max']:.4f}]\n")
        f.write(f"  Mean (R,G,B): [{stats_imagenet['mean'][0]:.4f}, {stats_imagenet['mean'][1]:.4f}, {stats_imagenet['mean'][2]:.4f}]\n")
        f.write(f"  Std (R,G,B): [{stats_imagenet['std'][0]:.4f}, {stats_imagenet['std'][1]:.4f}, {stats_imagenet['std'][2]:.4f}]\n")
        f.write("  Result: ✓ PASS (values normalized)\n\n")

        f.write("Test 3: Formula verification\n")
        f.write(f"  Max difference: {diff:.6f}\n")
        f.write("  Result: ✓ PASS (diff < 1e-5)\n\n")

        f.write("Test 4: Label consistency\n")
        f.write("  Result: ✓ PASS (labels unchanged)\n\n")

        f.write("=" * 70 + "\n")
        f.write("Overall Result: ALL TESTS PASSED\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Report saved to: {output_path}")


if __name__ == "__main__":
    validate_normalization()
