# Dataset Manifest: combined_v2

## Overview

**Dataset Name**: combined_v2
**Version**: Phase 03c.C Multi-Dataset Fusion
**Preprocessing Version**: v3_no_clahe_imagenet_norm
**Generated**: 2025-01-14
**Total Samples**: 1905
**Task**: Binary classification (Normal vs. Glaucoma)

## Description

This dataset combines three public fundus image datasets for glaucoma classification:
- **RIM-ONE r3**: Clinical fundus images from Spanish ophthalmology centers
- **REFUGE2**: REFUGE Challenge 2018 training set
- **G1020**: Large-scale Chinese multi-center glaucoma dataset

The dataset was created as part of Phase 03c.C to maximize training data diversity and improve model generalization across different imaging equipment, populations, and clinical settings.

## Dataset Composition

### Split Distribution

| Split | Samples | Normal | Glaucoma | Balance |
|-------|---------|--------|----------|---------|
| Train | 1394 | 1042 (74.8%) | 352 (25.2%) | 2.96:1 |
| Val   | 132  | 91 (68.9%)   | 41 (31.1%)  | 2.22:1 |
| Test  | 379  | 264 (69.7%)  | 115 (30.3%) | 2.30:1 |
| **Total** | **1905** | **1397 (73.3%)** | **508 (26.7%)** | **2.75:1** |

### Source Dataset Breakdown

#### Training Split (1394 samples)
- **RIM-ONE**: 281 samples
- **REFUGE2**: 400 samples
- **G1020**: 713 samples

#### Validation Split (132 samples)
- **RIM-ONE**: 30 samples
- **G1020**: 102 samples

#### Test Split (379 samples)
- **RIM-ONE**: 174 samples
- **G1020**: 205 samples

### Class Distribution

**Overall**: 73.3% Normal, 26.7% Glaucoma

**Recommended Class Weights** (for loss balancing):
```python
class_weights = [0.2525, 0.7475]  # [normal, glaucoma]
```

Formula: `weight = total / (2 * class_count)`, normalized to sum to 1

## Preprocessing Pipeline

### Raw Image Processing

1. **Load**: OpenCV `cv2.imread()` (PNG, 8-bit, 3-channel)
2. **Color Space**: BGR → RGB conversion
3. **Resize**: 512×512 pixels (bicubic interpolation)
4. **Normalization**: Pixel values [0, 255] → [0, 1]
5. **ImageNet Normalization** (optional, enabled in Phase 03e):
   - Mean (RGB): [0.485, 0.456, 0.406]
   - Std (RGB): [0.229, 0.224, 0.225]

### Data Augmentation (Training Only)

| Augmentation | Probability | Range |
|--------------|-------------|-------|
| Horizontal Flip | 50% | - |
| Vertical Flip | 50% | - |
| Rotation | 100% | ±5° |
| Brightness | 100% | ±10% |
| Contrast | 100% | ±10% |

### CLAHE Status

**CLAHE Removed in Phase 03e**: CLAHE (Contrast Limited Adaptive Histogram Equalization) was previously applied to the green channel during preprocessing in Phase 03c/03d. It was **removed** in Phase 03e after an audit revealed it was incompatible with ImageNet normalization and reduced transfer learning effectiveness.

**Current Status**: No CLAHE applied. All images processed with basic resizing and normalization only.

## File Structure

```
combined_v2/
├── MANIFEST.md              # This file
├── metadata.json            # Dataset metadata (class labels, splits, etc.)
├── splits.json              # Train/val/test split indices
├── images/                  # RGB fundus images (512×512 PNG)
│   ├── combined_v2_0000.png
│   ├── combined_v2_0001.png
│   └── ...
└── masks/                   # Segmentation masks (optional, for future use)
    ├── combined_v2_0000_mask.png
    ├── combined_v2_0001_mask.png
    └── ...
```

## Metadata Format

### metadata.json

Contains sample-level annotations:

```json
{
  "dataset": "combined_v2",
  "version": "Phase 03c.C Multi-Dataset Fusion",
  "source_datasets": ["rim_one", "refuge2", "g1020"],
  "num_samples": 1905,
  "num_train": 1394,
  "num_val": 132,
  "num_test": 379,
  "task": "classification",
  "has_masks": true,
  "has_labels": true,
  "label_distribution": {"normal": 1397, "glaucoma": 508},
  "class_names": ["normal", "glaucoma"],
  "image_size": 512,
  "seed": 42,
  "samples": [
    {
      "sample_id": 0,
      "image_filename": "combined_v2_0000.png",
      "mask_filename": "combined_v2_0000_mask.png",
      "label": 0,
      "label_name": "normal",
      "split": "train",
      "source_dataset": "rim_one",
      "original_sample_id": 123,
      "original_filename": "rimone_0123.png"
    },
    ...
  ]
}
```

### splits.json

Contains split indices:

```json
{
  "train": [0, 1, 2, ..., 1393],
  "val": [1394, 1395, ..., 1525],
  "test": [1526, 1527, ..., 1904]
}
```

## Batch Statistics

Computed on training split (BEFORE ImageNet normalization):

| Channel | Mean | Std | Entropy |
|---------|------|-----|---------|
| Red     | 0.541 | 0.312 | 7.69 |
| Green   | 0.258 | 0.156 | 7.16 |
| Blue    | 0.129 | 0.088 | 6.42 |

**Value Range**: [0.000, 1.000] (after pixel normalization)

## Source Datasets

### RIM-ONE r3

- **Source**: [RIM-ONE Database](http://medimrg.webs.ull.es/)
- **Samples Used**: 485 total (281 train, 30 val, 174 test)
- **Characteristics**:
  - Clinical fundus images
  - Spanish population
  - High-quality optic disc imaging
  - Expert annotations

### REFUGE2

- **Source**: [REFUGE Challenge 2018](https://refuge.grand-challenge.org/)
- **Samples Used**: 400 training samples
  - **Note**: Validation and test sets excluded (labels unavailable)
- **Characteristics**:
  - Challenge dataset
  - Multiple imaging centers
  - Clinical-grade fundus images
  - Expert glaucoma annotations

### G1020

- **Source**: [G1020 Dataset](https://github.com/cuhk-aim-g1020/g1020)
- **Samples Used**: 1020 total (713 train, 102 val, 205 test)
- **Characteristics**:
  - Large-scale Chinese multi-center dataset
  - Diverse imaging equipment
  - Real-world clinical variability
  - Expert consensus labels

## Usage Example

```python
from data.fundus_dataset import FundusDataset

# Load training set with ImageNet normalization (Phase 03e)
dataset = FundusDataset(
    data_root='data/processed/combined_v2',
    split='train',
    task='classification',
    image_size=512,
    augment=True,
    use_imagenet_norm=True
)

print(f"Dataset size: {len(dataset)}")  # 1394
image, label = dataset[0]
print(f"Image shape: {image.shape}")    # (3, 512, 512)
print(f"Label: {label}")                # 0 (normal) or 1 (glaucoma)
```

## Quality Control

### Phase 03e Preprocessing Audit

**Date**: 2025-01-14

**Validation Tests**:
1. ✅ CLAHE removal confirmed (preprocessing scripts audited)
2. ✅ ImageNet normalization validated (value range: [-1.83, 2.18])
3. ✅ Formula verification (max diff: 0.000000)
4. ✅ Labels consistency check passed

**Batch Statistics**: Computed on all splits (see [reports/batch_statistics.json](../../reports/batch_statistics.json))

**CLAHE Audit**: False positives detected due to natural fundus image variance (see [reports/preprocessing_audit.json](../../reports/preprocessing_audit.json))

## Known Issues

### Class Imbalance

The dataset has a 2.75:1 imbalance in favor of normal samples. Recommended mitigation strategies:

1. **Class Weighting**: Use recommended weights [0.2525, 0.7475]
2. **Focal Loss**: Use `gamma=2.0` to down-weight easy examples
3. **Balanced Sampling**: Sample with probabilities proportional to inverse class frequencies

### Missing REFUGE2 Validation/Test Sets

REFUGE2 validation and test sets are excluded because labels are not publicly available (competition holdout sets). This reduces the validation set size.

## Changelog

### Phase 03c.C (2025-01-12)
- Initial dataset fusion
- Combined RIM-ONE + REFUGE2 + G1020
- Applied CLAHE preprocessing

### Phase 03e (2025-01-14)
- **BREAKING**: Removed CLAHE from preprocessing
- Regenerated all datasets from scratch
- Added ImageNet normalization support
- Created validation and audit scripts
- Updated preprocessing version to v3_no_clahe_imagenet_norm

## References

1. Fumero, F., Alayón, S., Sanchez, J. L., Sigut, J., & Gonzalez-Hernandez, M. (2011). RIM-ONE: An open retinal image database for optic nerve evaluation. *Proc. CBMS*, 116-121.

2. Orlando, J. I., et al. (2020). REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs. *Medical Image Analysis*, 59, 101570.

3. Liu, H., et al. (2021). Development and validation of a deep learning system to detect glaucomatous optic neuropathy using fundus photographs. *JAMA Ophthalmology*, 137(12), 1353-1360.

## License

This combined dataset inherits the licenses of its source datasets:
- **RIM-ONE r3**: [Check source website](http://medimrg.webs.ull.es/)
- **REFUGE2**: [REFUGE Challenge terms](https://refuge.grand-challenge.org/)
- **G1020**: [Check repository](https://github.com/cuhk-aim-g1020/g1020)

**Usage**: Research and educational purposes only. Commercial use requires checking individual dataset licenses.

## Contact

For questions about this dataset or preprocessing:
- See [docs/preprocessing_pipeline.md](../../docs/preprocessing_pipeline.md)
- Check [reports/imagenet_norm_validation.txt](../../reports/imagenet_norm_validation.txt)

---

**Last Updated**: 2025-01-14
**Maintained By**: AcuVue Development Team
**Preprocessing Version**: v3_no_clahe_imagenet_norm
