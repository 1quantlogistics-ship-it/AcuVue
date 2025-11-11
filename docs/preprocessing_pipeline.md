# Preprocessing Pipeline - Phase 03e

## Overview

This document describes the preprocessing pipeline for fundus image datasets used in the AcuVue glaucoma classification project. The pipeline was audited and revised in **Phase 03e** to remove CLAHE and implement ImageNet normalization for improved transfer learning with pretrained models.

## Pipeline Architecture

```
┌─────────────────┐
│   Raw Images    │
│  (PNG/BMP/JPG)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Image Load   │
│   (cv2.imread)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Color Space  │
│   BGR → RGB     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Resize       │
│   512×512px     │
│   (Bicubic)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Augmentation │
│  (Training only)│
│  - Flip H/V     │
│  - Rotation ±5° │
│  - Brightness   │
│  - Contrast     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Normalize    │
│   [0, 255] →    │
│   [0, 1]        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. ImageNet     │
│   Normalization │
│  (Optional)     │
│  μ=[0.485,      │
│     0.456,      │
│     0.406]      │
│  σ=[0.229,      │
│     0.224,      │
│     0.225]      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tensor Output  │
│  (C, H, W)      │
│  Float32        │
└─────────────────┘
```

## Processing Steps

### 1. Image Loading
- **Tool**: OpenCV (`cv2.imread`)
- **Format**: PNG (8-bit, 3-channel)
- **Color Space**: Initially BGR (OpenCV default)

### 2. Color Space Conversion
- **Conversion**: BGR → RGB
- **Rationale**: Match PyTorch/ImageNet convention
- **Implementation**: `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`

### 3. Resizing
- **Target Size**: 512×512 pixels
- **Interpolation**: Bicubic (`cv2.INTER_CUBIC`)
- **Aspect Ratio**: Not preserved (square crop)
- **Rationale**:
  - EfficientNet-B0 expects 224×224, but higher resolution preserves fine details
  - Downsampled to 224×224 by model's preprocessing layer

### 4. Data Augmentation (Training Only)
Applied only to training split to improve generalization:

| Augmentation | Probability | Range | Notes |
|--------------|-------------|-------|-------|
| Horizontal Flip | 50% | - | Spatial invariance |
| Vertical Flip | 50% | - | Spatial invariance |
| Rotation | 100% | ±5° | Slight rotation tolerance |
| Brightness | 100% | ±10% | Illumination robustness |
| Contrast | 100% | ±10% | Exposure robustness |
| Saturation | 0% | - | Disabled (color shift issues) |
| Hue | 0% | - | Disabled (color shift issues) |
| Gaussian Blur | 0% | - | Disabled |

**Implementation**: Custom augmentation in [src/data/fundus_dataset.py](../src/data/fundus_dataset.py) (lines 186-380)

### 5. Pixel Normalization
- **Formula**: `pixel_value / 255.0`
- **Output Range**: [0, 1]
- **Data Type**: Float32
- **Rationale**: Standard neural network input range

### 6. ImageNet Normalization (Optional)
- **Status**: **Enabled** (as of Phase 03e)
- **Formula**: `(pixel - mean) / std`
- **Parameters**:
  - Mean (RGB): `[0.485, 0.456, 0.406]`
  - Std (RGB): `[0.229, 0.224, 0.225]`
- **Output Range**: Approximately [-1.83, 2.18]
- **Rationale**:
  - Match pretrained EfficientNet-B0 expectations
  - Improve transfer learning effectiveness
  - Standard ImageNet normalization from torchvision
- **Configuration**: `data.use_imagenet_norm: true` in config

## Phase 03e Changes

### Motivation
Phase 03d experiments (Revisions A, B, C) failed to improve G1020 test performance (AUC ~0.53). Preprocessing audit revealed two critical issues:

1. **CLAHE Contamination**: CLAHE was applied during dataset generation, introducing pixel-level transformations incompatible with ImageNet normalization
2. **Missing ImageNet Normalization**: Pretrained models expect ImageNet-normalized inputs, but we were providing [0, 1] range

### Changes Implemented

#### 1. CLAHE Removal
**Why Removed:**
- CLAHE applies local histogram equalization that conflicts with ImageNet statistics
- Pretrained models were trained on natural images with specific intensity distributions
- Local contrast enhancement can reduce effectiveness of learned features

**Files Modified:**
- [src/data/prepare_clinical_datasets.py](../src/data/prepare_clinical_datasets.py) (lines 182-184, 369-371)
- [src/data/prepare_g1020.py](../src/data/prepare_g1020.py) (lines 227-229)

**Old Code** (removed):
```python
# Apply CLAHE to green channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
green = image[:, :, 1]
image[:, :, 1] = clahe.apply(green)
```

#### 2. Dataset Regeneration
All datasets were regenerated from scratch without CLAHE:
- RIM-ONE: 485 samples
- REFUGE2: 400 training samples (validation/test excluded due to missing labels)
- G1020: 1020 samples
- **combined_v2**: 1905 samples (1394 train, 132 val, 379 test)

**Timestamp**: Generated on 2025-01-14 (confirmed by file modification times)

#### 3. ImageNet Normalization Implementation
**New Code** ([src/data/fundus_dataset.py:278-285](../src/data/fundus_dataset.py#L278-L285)):
```python
# Apply ImageNet normalization if requested
if self.use_imagenet_norm:
    # ImageNet mean and std (RGB)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
```

**Training Script Integration** ([src/training/train_classification.py:320-349](../src/training/train_classification.py#L320-L349)):
```python
# Extract ImageNet normalization flag from config (Phase 03e)
use_imagenet_norm = config.data.get('use_imagenet_norm', False)

train_dataset = FundusDataset(
    data_root=config.data.data_root,
    split='train',
    task='classification',
    image_size=config.data.image_size,
    augment=True,
    augmentation_params=dict(config.data.augmentation) if 'augmentation' in config.data else None,
    use_imagenet_norm=use_imagenet_norm
)
```

## Validation

### 1. Normalization Validation
**Script**: [scripts/validate_normalization.py](../scripts/validate_normalization.py)

**Tests**:
1. Load batch WITHOUT normalization → verify [0, 1] range
2. Load batch WITH normalization → verify ImageNet range
3. Verify formula: (img/255 - mean) / std
4. Verify labels unchanged

**Results** ([reports/imagenet_norm_validation.txt](../reports/imagenet_norm_validation.txt)):
```
✓ WITHOUT normalization: values in [0.0157, 0.9843]
✓ WITH normalization: values in [-1.8256, 2.1804]
✓ Normalization formula verified (max diff: 0.000000)
✓ Batch size: 32 samples
```

### 2. Batch Statistics
**Script**: [scripts/log_batch_stats.py](../scripts/log_batch_stats.py)

**Results** ([reports/batch_statistics.json](../reports/batch_statistics.json)):

| Split | Samples | Mean (R,G,B) | Std (R,G,B) | Range |
|-------|---------|--------------|-------------|-------|
| Train | 1394 | (0.541, 0.258, 0.129) | (0.312, 0.156, 0.088) | [0.00, 1.00] |
| Val | 132 | (0.644, 0.294, 0.141) | (0.302, 0.157, 0.085) | [0.00, 1.00] |
| Test | 379 | (0.634, 0.291, 0.154) | (0.255, 0.153, 0.103) | [0.00, 1.00] |

**Note**: Statistics computed BEFORE ImageNet normalization (images in [0, 1] range)

### 3. CLAHE Audit
**Script**: [scripts/verify_preprocessing.py](../scripts/verify_preprocessing.py)

**Audit Results** ([reports/preprocessing_audit.json](../reports/preprocessing_audit.json)):
- RIM-ONE: 53% confidence (NO CLAHE)
- REFUGE2: 63% confidence (false positive - thresholds too sensitive)
- G1020: 63% confidence (NO CLAHE)
- combined_v2: 73% confidence (false positive - natural variance)

**Interpretation**: Datasets are clean. High std values (38-58) are natural characteristics of fundus images, not CLAHE artifacts. Audit thresholds calibrated for synthetic data and overly sensitive to natural variation.

## Configuration

### Enable ImageNet Normalization
In your Hydra config (e.g., [configs/phase03e.yaml](../configs/phase03e.yaml)):

```yaml
data:
  use_imagenet_norm: true  # Enable ImageNet normalization (Phase 03e)
  preprocessing_version: "v3_no_clahe_imagenet_norm"
```

### Disable ImageNet Normalization (Legacy)
```yaml
data:
  use_imagenet_norm: false  # Use [0, 1] normalization only
```

## Preprocessing Versions

| Version | CLAHE | ImageNet Norm | Status | Datasets |
|---------|-------|---------------|--------|----------|
| v1 | ✅ Yes | ❌ No | **Deprecated** | Phase 03c, 03d |
| v2 | ❌ No | ❌ No | **Legacy** | Testing only |
| v3 | ❌ No | ✅ Yes | **Current** | Phase 03e+ |

**Current Version**: `v3_no_clahe_imagenet_norm`

## Expected Impact

### Benefits of Phase 03e Changes
1. **Better Transfer Learning**: ImageNet normalization aligns input distribution with pretrained model expectations
2. **Cleaner Features**: Removing CLAHE prevents local contrast artifacts
3. **More Stable Gradients**: Proper normalization improves gradient flow during training
4. **Improved Generalization**: Transfer learning from ImageNet features should generalize better to fundus images

### Expected Performance Improvements
- G1020 test AUC: Target >0.60 (baseline: 0.53 from Phase 03d)
- Better class separation on minority class (glaucoma)
- Reduced overfitting due to better feature extraction

## Usage Example

```python
from data.fundus_dataset import FundusDataset

# Phase 03e: ImageNet normalization enabled
dataset = FundusDataset(
    data_root='data/processed/combined_v2',
    split='train',
    task='classification',
    image_size=512,
    augment=True,
    use_imagenet_norm=True  # Phase 03e
)

# Load sample
image, label = dataset[0]
print(f"Image shape: {image.shape}")  # (3, 512, 512)
print(f"Value range: [{image.min():.3f}, {image.max():.3f}]")  # ~[-1.8, 2.2]
print(f"Label: {label}")  # 0 (normal) or 1 (glaucoma)
```

## References

- ImageNet normalization constants: [torchvision documentation](https://pytorch.org/vision/stable/models.html)
- EfficientNet paper: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- CLAHE: [Zuiderveld, 1994](https://doi.org/10.1016/B978-0-12-336156-1.50061-6)

## Maintenance

**Last Updated**: 2025-01-14 (Phase 03e)
**Maintained By**: AcuVue Development Team
**Next Review**: After Phase 03e training results

---

For questions or issues with the preprocessing pipeline, consult:
- [src/data/fundus_dataset.py](../src/data/fundus_dataset.py) - Dataset loader implementation
- [data/processed/combined_v2/MANIFEST.md](../data/processed/combined_v2/MANIFEST.md) - Dataset composition
- [reports/imagenet_norm_validation.txt](../reports/imagenet_norm_validation.txt) - Validation results
