# Multi-Head Domain Routing Architecture

This document describes the multi-head inference architecture for AcuVue, which routes fundus images to domain-specific expert models for improved glaucoma detection accuracy.

## Overview

The multi-head architecture separates two concerns:

1. **Domain Classification** (Router): Identifies WHERE an image came from (device/dataset)
2. **Disease Diagnosis** (Expert Heads): Specialized models optimized for each domain

This separation allows each expert head to be optimized for the specific characteristics of images from different acquisition sources, while the lightweight router handles real-time routing decisions.

## Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Fundus Image Input          │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │          Domain Router              │
                    │    (MobileNetV3-Small, ~2.5M)       │
                    │                                     │
                    │  Classifies: rimone, refuge2,       │
                    │              g1020, unknown         │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────┼────────────────────┐
                    │                │                    │
                    ▼                ▼                    ▼
           ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
           │  RIM-ONE Head │ │ REFUGE2 Head  │ │  G1020 Head   │
           │ (EfficientNet)│ │ (EfficientNet)│ │ (EfficientNet)│
           └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │       Glaucoma Prediction           │
                    │   (normal/glaucoma + confidence)    │
                    └─────────────────────────────────────┘
```

## Key Principle

**The router does NOT diagnose glaucoma** - it only identifies the image domain. Domain classification is about learning acquisition characteristics:

- Device artifacts and color profiles
- Image resolution and quality patterns
- Preprocessing signatures from different hospitals

The expert heads then perform the actual glaucoma diagnosis, optimized for their specific domain's characteristics.

## Components

### 1. Domain Router (`src/routing/`)

The domain router is a lightweight classifier that routes images to the appropriate expert head.

```python
from src.routing import DomainRouter

router = DomainRouter("models/routing/domain_classifier_v1.pt")
result = router.route(image)
print(f"Domain: {result.domain} ({result.confidence:.1%})")
# Output: Domain: rimone (94.2%)
```

**Architecture:**
- Backbone: MobileNetV3-Small (~2.5M parameters)
- Input: 224x224 RGB images (ImageNet normalized)
- Output: 4 domain classes (rimone, refuge2, g1020, unknown)
- Inference time: <10ms on GPU

**Files:**
- `src/routing/router.py` - DomainRouter high-level interface
- `src/routing/domain_classifier.py` - PyTorch model definition

### 2. Expert Heads (`src/inference/`)

Each expert head is an EfficientNet-B0 model trained specifically for glaucoma detection on its target domain.

```python
from src.inference.predictor import GlaucomaPredictor

predictor = GlaucomaPredictor.from_checkpoint("models/production/glaucoma_rimone_v1.pt")
result = predictor.predict(image)
print(f"{result.prediction}: {result.confidence:.1%}")
# Output: glaucoma: 87.3%
```

**Architecture:**
- Backbone: EfficientNet-B0 (~5.3M parameters)
- Input: 224x224 RGB images (ImageNet normalized)
- Output: Binary classification (normal/glaucoma)

### 3. Multi-Head Pipeline (`src/inference/pipeline.py`)

The pipeline orchestrates routing and prediction:

```python
from src.inference.pipeline import MultiHeadPipeline

pipeline = MultiHeadPipeline.from_config("configs/pipeline_v1.yaml")
result = pipeline.predict(image)

print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
print(f"Routed via: {result.routed_domain} -> {result.head_used}")
# Output:
# Prediction: glaucoma (87.3%)
# Routed via: rimone -> glaucoma_rimone_v1
```

**Features:**
- Automatic domain routing
- Fallback handling for unknown domains
- Ensemble mode for uncertain cases
- JSON-serializable results

### 4. Head Registry (`src/inference/head_registry.py`)

Manages available expert heads and domain mappings:

```python
from src.inference.head_registry import get_head_for_domain, get_available_heads

# Get head for a domain
config = get_head_for_domain('rimone')
print(f"Using: {config.name}")

# List all available heads
heads = get_available_heads()  # ['glaucoma_rimone_v1', ...]
```

## Domain Dataset & Training

### Domain Classification Dataset

The domain dataset learns to classify images by their source:

```python
from src.data.domain_dataset import create_domain_dataloaders

dataloaders = create_domain_dataloaders(
    data_roots={
        'rimone': 'data/processed/rim_one',
        'refuge': 'data/processed/refuge2',
        'g1020': 'data/processed/g1020',
    },
    batch_size=32,
)
```

**Key files:**
- `src/data/domain_labels.py` - Domain extraction from filenames
- `src/data/domain_dataset.py` - PyTorch Dataset classes

### Router Training

Train the domain router with:

```bash
python src/training/train_router.py --config configs/router_training_v1.yaml
```

**Training characteristics:**
- Fast training: ~20 epochs, <1 hour
- Light augmentation to preserve domain signatures
- Target: >90% domain classification accuracy

**Configuration (`configs/router_training_v1.yaml`):**
```yaml
model:
  backbone: mobilenetv3_small_100
  num_classes: 3  # rimone, refuge, g1020

training:
  epochs: 20
  batch_size: 32
  lr: 0.001
  optimizer: adamw
  scheduler: cosine
```

## Pipeline Configuration

Example pipeline config (`configs/pipeline_v1.yaml`):

```yaml
router:
  checkpoint: models/routing/domain_classifier_v1.pt

heads:
  glaucoma_rimone_v1:
    checkpoint: models/production/glaucoma_efficientnet_b0_v1.pt
  # Future heads:
  # glaucoma_refuge_v1:
  #   checkpoint: models/production/glaucoma_refuge_v1.pt

domain_mapping:
  rimone: glaucoma_rimone_v1
  refuge2: glaucoma_rimone_v1  # Fallback until trained
  g1020: glaucoma_rimone_v1   # Fallback until trained
  unknown: glaucoma_rimone_v1
```

## Usage Examples

### Basic Prediction

```python
from src.inference.pipeline import MultiHeadPipeline

# Load pipeline
pipeline = MultiHeadPipeline.from_config("configs/pipeline_v1.yaml")

# Single prediction
result = pipeline.predict("patient_fundus.png")
print(f"Diagnosis: {result.prediction}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Domain: {result.routed_domain}")
```

### Routing Only

```python
# Get routing info without running diagnosis
routing = pipeline.get_routing_info(image)
print(f"Domain: {routing.domain} ({routing.confidence:.1%})")
print(f"All scores: {routing.all_scores}")
```

### Ensemble Prediction

```python
# Use multiple heads for uncertain cases
result = pipeline.predict_with_ensemble(
    image,
    head_names=['glaucoma_rimone_v1', 'glaucoma_refuge_v1']
)
print(f"Ensemble prediction: {result.prediction}")
```

### Batch Processing

```python
from src.inference.predictor import GlaucomaPredictor

predictor = GlaucomaPredictor.from_checkpoint(checkpoint_path)
results = predictor.predict_batch(image_paths, batch_size=16)

for result in results:
    print(f"{result.image_path}: {result.prediction}")
```

## Testing

Run the routing tests:

```bash
# Unit tests
pytest tests/unit/test_domain_router.py -v
pytest tests/unit/test_multi_head_pipeline.py -v

# Integration tests
pytest tests/integration/test_routing_pipeline.py -v
```

## Performance Considerations

| Component | Parameters | Inference Time (GPU) |
|-----------|------------|---------------------|
| Router    | ~2.5M      | <10ms               |
| Expert Head| ~5.3M     | ~20ms               |
| **Total** | ~7.8M      | ~30ms               |

The lightweight router adds minimal overhead while enabling domain-specific optimization.

## Future Work

1. **Additional Expert Heads**: Train specialized heads for REFUGE2 and G1020 domains
2. **Confidence Thresholding**: Automatic ensemble when routing confidence is low
3. **New Domain Detection**: Flag images that don't match known domains
4. **Model Distillation**: Compress expert heads for edge deployment
