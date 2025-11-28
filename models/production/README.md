# Production Model Weights

## Overview

This directory contains production model weights for AcuVue glaucoma detection.
These are **expert head** models - trained on specific dataset domains and called
by the multi-head pipeline based on domain routing results.

## Multi-Head Architecture

```
Image -> Domain Router -> Expert Head Selection -> Diagnosis
         (routing/)       (production/)
```

The domain router (in `models/routing/`) classifies which dataset an image
belongs to, then routes to the appropriate expert head in this directory.

## Expert Heads

| Model | Size | Domain | Test AUC | SHA256 |
|-------|------|--------|----------|--------|
| `glaucoma_efficientnet_b0_v1.pt` | 46.4 MB | RIM-ONE | 93.7% | *see below* |

*Future heads will be added as training completes for REFUGE2, G1020*

## Domain Router

The domain router model is stored in `models/routing/`:

| Model | Size | Purpose |
|-------|------|---------|
| `domain_classifier_v1.pt` | ~5 MB | Route images to expert heads |

## Download

Model weights are too large for git. Use the download script:

```bash
# List available models
python scripts/download_weights.py --list

# Download all weights
python scripts/download_weights.py --all

# Download specific model
python scripts/download_weights.py --model glaucoma_efficientnet_b0_v1

# Show checksums for verification
python scripts/download_weights.py --list --checksums
```

## Manual Setup

If download URLs are not configured, obtain weights manually:

### Option 1: Copy from training results

```bash
cp /path/to/training/results/rimone_efficientnet_b0_best.pt \
   models/production/glaucoma_efficientnet_b0_v1.pt
```

### Option 2: Contact team

Contact the team for model weights location.

### Option 3: Retrain

```bash
python src/training/train_classification.py --config configs/production_training_v1.yaml
```

## Verification

```python
import torch
checkpoint = torch.load("models/production/glaucoma_efficientnet_b0_v1.pt")
print(f"Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'state_dict'}")
```

## Model Details

### glaucoma_efficientnet_b0_v1

- **Architecture**: EfficientNet-B0 (timm)
- **Input**: 224x224 RGB, ImageNet normalized
- **Output**: 2 classes ['normal', 'glaucoma']
- **Domain**: RIM-ONE dataset family
- **Test AUC**: 93.7%
- **Test Accuracy**: 76.5%
- **Test Sensitivity**: 74.4%
- **Test Specificity**: 91.7%
- **Training**: Hospital-based splitting (r2/r3 train, r1 test)

## Pipeline Usage

```python
from src.inference import MultiHeadPipeline

# Load full pipeline (router + heads)
pipeline = MultiHeadPipeline.from_config("configs/pipeline_v1.yaml")

# Predict with automatic routing
result = pipeline.predict("path/to/fundus.png")
print(f"Domain: {result.routed_domain} -> Head: {result.head_used}")
print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
```
