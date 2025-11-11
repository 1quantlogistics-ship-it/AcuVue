#!/usr/bin/env python3
"""
Batch Statistics Logger - Phase 03e
====================================

Computes comprehensive batch statistics for clean datasets (without CLAHE).
Establishes baseline statistics for future preprocessing audits.

Statistics computed:
- Per-channel mean and std (before normalization)
- Global min/max values
- Histogram entropy per channel
- Sample count per split
- Class distribution per split

Usage:
    python scripts/log_batch_stats.py --data-root data/processed/combined_v2

Output:
    reports/batch_statistics.json
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from data.fundus_dataset import FundusDataset


def compute_channel_stats(dataset: FundusDataset, split: str, num_samples: int = None) -> Dict:
    """
    Compute statistics for a single split.

    Args:
        dataset: FundusDataset instance
        split: Split name ('train', 'val', 'test')
        num_samples: Number of samples to analyze (None = all)

    Returns:
        Dictionary with statistics
    """
    print(f"\nComputing statistics for {split} split...")

    # Initialize accumulators
    num_samples = num_samples or len(dataset)
    num_samples = min(num_samples, len(dataset))

    channel_sums = torch.zeros(3)
    channel_sq_sums = torch.zeros(3)
    pixel_count = 0

    global_min = float('inf')
    global_max = float('-inf')

    # Class distribution
    class_counts = {}

    # Collect samples for histogram analysis
    sample_images = []

    for i in tqdm(range(num_samples), desc=f"Processing {split}"):
        # Get sample (image, label for classification dataset)
        sample = dataset[i]

        if len(sample) == 2:  # Classification: (image, label)
            image, label = sample
        elif len(sample) == 3:  # Segmentation: (image, mask, label)
            image, _, label = sample
        else:
            raise ValueError(f"Unexpected sample format: {len(sample)} elements")

        # Image is in [0, 1] range after loading (before normalization)
        # Accumulate per-channel statistics
        channel_sums += image.mean(dim=[1, 2])  # Mean over spatial dims
        channel_sq_sums += (image ** 2).mean(dim=[1, 2])
        pixel_count += 1

        # Track global min/max
        global_min = min(global_min, image.min().item())
        global_max = max(global_max, image.max().item())

        # Track class distribution
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[label_val] = class_counts.get(label_val, 0) + 1

        # Store first 100 samples for histogram analysis
        if len(sample_images) < 100:
            sample_images.append(image)

    # Compute mean and std per channel
    mean_per_channel = (channel_sums / pixel_count).numpy()

    # Var = E[X^2] - E[X]^2
    var_per_channel = (channel_sq_sums / pixel_count) - (channel_sums / pixel_count) ** 2
    std_per_channel = torch.sqrt(var_per_channel).numpy()

    # Compute histogram entropy for sample batch
    sample_batch = torch.stack(sample_images[:100])  # (N, C, H, W)

    entropy_per_channel = []
    for c in range(3):
        channel_data = (sample_batch[:, c, :, :] * 255).numpy().astype(np.uint8)
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        entropy_per_channel.append(entropy)

    return {
        'split': split,
        'num_samples': num_samples,
        'num_samples_total': len(dataset),
        'mean_per_channel': {
            'R': float(mean_per_channel[0]),
            'G': float(mean_per_channel[1]),
            'B': float(mean_per_channel[2])
        },
        'std_per_channel': {
            'R': float(std_per_channel[0]),
            'G': float(std_per_channel[1]),
            'B': float(std_per_channel[2])
        },
        'global_min': float(global_min),
        'global_max': float(global_max),
        'entropy_per_channel': {
            'R': float(entropy_per_channel[0]),
            'G': float(entropy_per_channel[1]),
            'B': float(entropy_per_channel[2])
        },
        'class_distribution': class_counts
    }


def main():
    parser = argparse.ArgumentParser(description='Compute batch statistics for dataset')
    parser.add_argument('--data-root', type=str, default='data/processed/combined_v2',
                        help='Root directory of dataset')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to analyze per split (None = all)')
    parser.add_argument('--output', type=str, default='reports/batch_statistics.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("Batch Statistics Logger - Phase 03e")
    print("=" * 70)
    print(f"Dataset: {args.data_root}")
    print(f"Output: {args.output}")
    print()

    # Compute statistics for each split
    stats = {
        'dataset': str(args.data_root),
        'preprocessing_version': 'v3_no_clahe_imagenet_norm',
        'note': 'Statistics computed BEFORE ImageNet normalization (images in [0, 1] range)',
        'splits': {}
    }

    for split_name in ['train', 'val', 'test']:
        try:
            dataset = FundusDataset(
                data_root=args.data_root,
                split=split_name,
                task='classification',
                image_size=512,
                augment=False,  # No augmentation for statistics
                use_imagenet_norm=False  # Compute stats BEFORE normalization
            )

            split_stats = compute_channel_stats(dataset, split_name, args.num_samples)
            stats['splits'][split_name] = split_stats

        except Exception as e:
            print(f"Error processing {split_name} split: {e}")
            continue

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("Batch Statistics Summary")
    print("=" * 70)
    print()

    for split_name, split_stats in stats['splits'].items():
        print(f"{split_name.upper()} Split:")
        print(f"  Samples: {split_stats['num_samples']}/{split_stats['num_samples_total']}")
        print(f"  Mean (R,G,B): ({split_stats['mean_per_channel']['R']:.4f}, "
              f"{split_stats['mean_per_channel']['G']:.4f}, "
              f"{split_stats['mean_per_channel']['B']:.4f})")
        print(f"  Std (R,G,B):  ({split_stats['std_per_channel']['R']:.4f}, "
              f"{split_stats['std_per_channel']['G']:.4f}, "
              f"{split_stats['std_per_channel']['B']:.4f})")
        print(f"  Value range: [{split_stats['global_min']:.4f}, {split_stats['global_max']:.4f}]")
        print(f"  Entropy (R,G,B): ({split_stats['entropy_per_channel']['R']:.2f}, "
              f"{split_stats['entropy_per_channel']['G']:.2f}, "
              f"{split_stats['entropy_per_channel']['B']:.2f})")
        print(f"  Class distribution: {split_stats['class_distribution']}")
        print()

    print("=" * 70)
    print(f"✓ Statistics saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
