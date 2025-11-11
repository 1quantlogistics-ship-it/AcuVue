#!/usr/bin/env python3
"""
Preprocessing Audit Script - Detect CLAHE Application
=====================================================

Analyzes sample images from each dataset to detect CLAHE preprocessing.

CLAHE (Contrast Limited Adaptive Histogram Equalization) has distinctive characteristics:
- Green channel histogram shows increased contrast with sharper peaks
- Histogram bins show more uniform distribution (equalization effect)
- Standard deviation of pixel intensities typically increases
- Histogram entropy increases

Usage:
    python scripts/verify_preprocessing.py

Output:
    reports/preprocessing_audit.json
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random


def compute_histogram_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Compute histogram statistics for green channel.

    CLAHE typically causes:
    - Higher standard deviation (increased contrast)
    - Higher entropy (more uniform distribution)
    - Flattened histogram (equalization)
    """
    # Extract green channel
    if image.ndim == 3:
        green = image[:, :, 1]
    else:
        green = image

    # Compute histogram
    hist, _ = np.histogram(green, bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()  # Normalize

    # Compute statistics
    mean_val = np.mean(green)
    std_val = np.std(green)

    # Compute entropy
    # Higher entropy = more uniform distribution (CLAHE effect)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    # Compute histogram uniformity (flatness)
    # Lower uniformity = sharper peaks (raw image)
    # Higher uniformity = flatter distribution (CLAHE)
    uniformity = 1 / (np.sum((hist_norm - 1/256)**2) + 1e-10)

    # Peak count (number of histogram bins with > mean count)
    peak_count = np.sum(hist > hist.mean())

    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'entropy': float(entropy),
        'uniformity': float(uniformity),
        'peak_count': int(peak_count)
    }


def detect_clahe(stats: Dict[str, float]) -> Tuple[bool, float]:
    """
    Detect CLAHE application based on histogram statistics.

    CLAHE indicators:
    - entropy > 6.5 (more uniform distribution)
    - std > 40 (increased contrast)
    - uniformity > 0.02 (flatter histogram)

    Returns:
        (clahe_detected, confidence_score)
    """
    clahe_score = 0.0
    max_score = 3.0

    # Entropy check (CLAHE typically increases entropy)
    if stats['entropy'] > 6.5:
        clahe_score += 1.0
    elif stats['entropy'] > 6.0:
        clahe_score += 0.5

    # Std check (CLAHE increases contrast)
    if stats['std'] > 45:
        clahe_score += 1.0
    elif stats['std'] > 35:
        clahe_score += 0.5

    # Uniformity check (CLAHE flattens histogram)
    if stats['uniformity'] > 0.025:
        clahe_score += 1.0
    elif stats['uniformity'] > 0.018:
        clahe_score += 0.5

    confidence = clahe_score / max_score
    detected = confidence > 0.5

    return detected, confidence


def analyze_dataset(dataset_path: Path, dataset_name: str, num_samples: int = 5) -> Dict:
    """Analyze sample images from a dataset."""

    images_dir = dataset_path / "images"

    if not images_dir.exists():
        return {
            'error': f"Dataset path not found: {images_dir}",
            'clahe_detected': False
        }

    # Get all PNG files
    image_files = list(images_dir.glob("*.png"))

    if len(image_files) == 0:
        return {
            'error': "No images found",
            'clahe_detected': False
        }

    # Sample random images
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    # Analyze each sample
    sample_stats = []
    clahe_detections = []
    confidences = []

    for img_file in sample_files:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Compute stats
        stats = compute_histogram_stats(img)
        detected, confidence = detect_clahe(stats)

        sample_stats.append({
            'filename': img_file.name,
            'stats': stats,
            'clahe_detected': detected,
            'confidence': confidence
        })

        clahe_detections.append(detected)
        confidences.append(confidence)

    # Aggregate results
    num_detected = sum(clahe_detections)
    avg_confidence = np.mean(confidences) if confidences else 0.0

    # Final decision: majority vote with confidence threshold
    clahe_present = num_detected >= (len(sample_stats) * 0.6)  # 60% threshold

    return {
        'dataset': dataset_name,
        'num_samples': len(sample_stats),
        'num_detected': num_detected,
        'detection_rate': num_detected / len(sample_stats) if sample_stats else 0.0,
        'avg_confidence': avg_confidence,
        'clahe_detected': clahe_present,
        'samples': sample_stats
    }


def main():
    # Define dataset paths
    base_path = Path("/Users/bengibson/AcuVue Depo/AcuVue/data/processed")

    datasets = {
        'rim_one': base_path / 'rim_one',
        'refuge2': base_path / 'refuge2',
        'g1020': base_path / 'g1020',
        'combined_v2': base_path / 'combined_v2'
    }

    print("=" * 60)
    print("Preprocessing Audit: CLAHE Detection")
    print("=" * 60)
    print()

    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"Analyzing {dataset_name}...")
        result = analyze_dataset(dataset_path, dataset_name, num_samples=5)
        results[dataset_name] = result

        if 'error' in result:
            print(f"  ✗ {result['error']}")
        else:
            clahe_str = "YES" if result['clahe_detected'] else "NO"
            confidence_pct = result['avg_confidence'] * 100
            print(f"  CLAHE detected: {clahe_str} (confidence: {confidence_pct:.1f}%)")
            print(f"  Detection rate: {result['detection_rate']*100:.1f}% ({result['num_detected']}/{result['num_samples']})")

            # Print sample statistics
            if result['num_samples'] > 0:
                avg_entropy = np.mean([s['stats']['entropy'] for s in result['samples']])
                avg_std = np.mean([s['stats']['std'] for s in result['samples']])
                print(f"  Avg entropy: {avg_entropy:.2f}, Avg std: {avg_std:.2f}")
        print()

    # Save results
    output_path = Path("/Users/bengibson/AcuVue Depo/AcuVue/reports/preprocessing_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"✓ Audit complete! Results saved to: {output_path}")
    print("=" * 60)
    print()

    # Summary
    print("SUMMARY:")
    for dataset_name, result in results.items():
        if 'error' not in result:
            status = "CLAHE DETECTED" if result['clahe_detected'] else "NO CLAHE"
            print(f"  {dataset_name:15s}: {status} ({result['avg_confidence']*100:.0f}% confidence)")

    # Decision
    print()
    print("RECOMMENDATION:")
    combined_result = results.get('combined_v2', {})
    if combined_result.get('clahe_detected', False):
        print("  → Dataset regeneration REQUIRED (CLAHE detected)")
        print("  → Proceed to Part 2: Remove CLAHE and regenerate datasets")
    else:
        print("  → Dataset OK (no CLAHE detected)")
        print("  → Skip Part 2, proceed to Part 3: ImageNet normalization")


if __name__ == '__main__':
    main()
