# Domain Router Weights

## Overview

This directory contains the domain classification model weights.
The domain router identifies which dataset family a fundus image belongs to,
enabling the multi-head pipeline to route to the appropriate expert head.

## Model Files

| File | Size | Description |
|------|------|-------------|
| `domain_classifier_v1.pt` | ~5 MB | MobileNetV3-Small domain classifier |

## Domain Classification

The router classifies images into 4 domains:

| Domain | Description |
|--------|-------------|
| `rimone` | RIM-ONE dataset family |
| `refuge2` | REFUGE2 challenge images |
| `g1020` | G1020 dataset |
| `unknown` | Unrecognized domain |

## Architecture

- **Backbone**: MobileNetV3-Small (~2.5M params)
- **Input**: 224x224 RGB, ImageNet normalized
- **Output**: 4-class classification
- **Purpose**: Fast routing decisions (not diagnosis)

## Download

```bash
python scripts/download_weights.py --model domain_classifier_v1
```

## Training

Train the domain classifier using:

```bash
python scripts/train_domain_classifier.py --config configs/router_training_v1.yaml
```

## Usage

```python
from src.routing import DomainRouter

router = DomainRouter("models/routing/domain_classifier_v1.pt")
result = router.route("path/to/fundus.png")
print(f"Domain: {result.domain} ({result.confidence:.1%})")
```
