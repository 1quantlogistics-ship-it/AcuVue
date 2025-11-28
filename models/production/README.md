# Production Model Weights

## Overview

This directory contains the production model weights for AcuVue glaucoma detection.

## Model Files

| File | Size | Description |
|------|------|-------------|
| `glaucoma_efficientnet_b0_v1.pt` | ~47MB | Production model weights (not in git) |
| `training_history_v1.json` | ~4KB | Training metrics history |
| `dataset_metadata_v1.json` | ~1KB | Dataset and splitting info |

## Setup Instructions

The model weights file (`glaucoma_efficientnet_b0_v1.pt`) is too large for git and must be obtained separately.

### Option 1: Copy from training results

If you have access to the original training output:

```bash
cp /path/to/training/results/rimone_efficientnet_b0_best.pt \
   models/production/glaucoma_efficientnet_b0_v1.pt
```

### Option 2: Download from team storage

Contact the team for the model weights file location.

### Option 3: Retrain

Use the training configuration in `configs/production_training_v1.yaml` to retrain:

```bash
# Training command (requires RIMONE dataset)
python src/training/train_classification.py --config configs/production_training_v1.yaml
```

## Verification

After obtaining the weights, verify the file:

```python
import torch
checkpoint = torch.load("models/production/glaucoma_efficientnet_b0_v1.pt")
print(f"Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'state_dict'}")
```

## Model Details

- **Architecture**: EfficientNet-B0 (timm)
- **Input**: 224x224 RGB, ImageNet normalized
- **Output**: 2 classes ['normal', 'glaucoma']
- **Test AUC**: 93.7% (hospital-based split)
- **Training**: Hospital-based splitting (r2/r3 train, r1 test)
